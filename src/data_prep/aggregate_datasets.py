"""
Combine given datasets into a single, deduplicated annotation file
"aggregated_annotations.csv", then check & reconcile species names. Store
species names not found in the "official" taxonomic file in
"species_name_mapping.{csv | xlsx}" for the user to specify how to handle those
species labels (rename/map, remove, or retain).
"""

import os
import logging
import coloredlogs
import argparse
import re
from glob import glob
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process, fuzz

from utils.utils_metadata import (
    load_dataset_metadata_config,
    store_dataset_metadata,
    collect_dataset_annotations,
)

# from utils.utils_functions import assign_codes_ordered_by_frequency

# initialize tqdm
tqdm.pandas()


# =============================================
# FUNCTIONS
# =============================================
def download_species_file(download_files=False) -> pd.DataFrame:
    """Load Orthoptera and Cicadidae species names from the corresponding TaxonWorks servers to check against.
    Either load a cached local species file or download it (for replicability).
    """

    df_simple = None
    # these seem to be public links, so putting them here should be fine
    cicadidae_url = "https://sfg.taxonworks.org/api/v1/otus/5974/inventory/dwc.csv?project_token=ZEJhFp9sq8kBfks15qAbAg"
    orthoptera_url = "https://sfg.taxonworks.org/api/v1/otus/805980/inventory/dwc.csv?project_token=3oerVKf82_196cIECvHYNg"
    if download_files:
        try:
            # a simple progress bar, because it takes a moment...
            with tqdm(total=2, desc="downloading species files") as pbar:
                pbar.set_postfix_str("downloading Orthoptera file")
                df_orthoptera = pd.read_csv(
                    orthoptera_url, sep="\t", low_memory=False
                )
                pbar.update(1)
                pbar.set_postfix_str("downloading Cicadidae file")
                df_cicadidae = pd.read_csv(
                    cicadidae_url, sep="\t", low_memory=False
                )
                pbar.update(1)
            logging.info("Downloaded species file.")
            # df cleanup - keep only the relevant columns
            # call it neoptera, since it's the common taxon of Orthoptera & Cicadidae
            df_neoptera = pd.concat([df_orthoptera, df_cicadidae])

            # throw out unused columns
            df_simple = df_neoptera[
                [
                    "country",
                    "genus",
                    "order",
                    "previousIdentifications",
                    "scientificName",
                    "scientificNameAuthorship",
                    "specificEpithet",
                ]
            ].copy()
            # get the species names into the same nomenclature as
            # InsectSet47 and 66 (<genus><species>, no space or underscore)
            df_simple["genus"] = df_simple["genus"].astype(str)
            df_simple["specificEpithet"] = df_simple["specificEpithet"].astype(
                str
            )
            df_simple["scientificName"] = (
                df_simple["genus"].str.strip()
                + " "
                + df_simple["specificEpithet"].str.strip()
            )
            df_simple["scientificNameNoSpace"] = df_simple[
                "scientificName"
            ].str.replace(" ", "")

            # remove duplicates species names such that we keep track of synonyms/prev. identifcations
            df_simple = df_simple.drop_duplicates(
                subset=["previousIdentifications", "scientificName"]
            )

            # add a column for previous identifications (normalized to <genus><species>)
            df_simple["prev_id_norm"] = df_simple[
                "previousIdentifications"
            ].apply(normalize_taxon)

            # save the species file
            species_file_dirpath = "./data/aggregation/species_file/"
            os.makedirs(species_file_dirpath, exist_ok=True)
            species_file_fname = "species_file_simple.csv"
            species_file_fpath = os.path.join(
                species_file_dirpath, species_file_fname
            )
            df_simple.to_csv(species_file_fpath, index=False)
            logging.info(f"Stored updated species file {species_file_fpath}")

        except Exception as e:
            logging.warning("Failed to download species file data.")
            logging.warning(e)
            logging.info("Falling back on cached copy.")

    if df_simple is None:
        # download_files was false, or download failed
        species_file_cached_path = (
            "./data/aggregation/species_file/cached_species_file_simple.csv"
        )
        df_simple = pd.read_csv(species_file_cached_path)
        logging.info("Loaded cached copy of the species files.")

    return df_simple


def apply_previous_mapping(mapping_to_use_fpath, mapping_new):
    # load old mapping
    if mapping_to_use_fpath.endswith(".xlsx"):
        mapping_use = pd.read_excel(mapping_to_use_fpath)
    elif mapping_to_use_fpath.endswith(".csv"):
        mapping_use = pd.read_csv(mapping_to_use_fpath)
    else:
        raise UserWarning("--use-mapping must be an Excel or CSV file.")
    # remove columns that will be replaced from old maping
    mapping_new.drop(labels=["user_mapping", "comment"], axis=1, inplace=True)
    # merge useful columns onto new mapping
    mapping_out = mapping_new.merge(
        mapping_use[["class_name", "user_mapping", "comment"]],
        on="class_name",
        how="left",
    )
    return mapping_out


def remove_duplicates_hashes(df, output_path):
    # find duplicate recordings based on hashes
    # this showed that all of i47 is in i66 - we can drop i47 entirely
    # 21 of i32 are in i66 - just deduplicate them
    if not df["hash"].is_unique:
        logging.info("Duplicate hashes found.")

        # get the unique files
        df_unique = df.drop_duplicates(subset="hash")

        # find duplicates and store them in a csv
        df_duplicates_all = df[df.duplicated(subset="hash", keep=False)]
        df_duplicates_all = df_duplicates_all.sort_values("hash")

        logging.info(
            f"A total of {len(df_duplicates_all)} duplicate hashes found in:"
        )
        dupe_vcs = df_duplicates_all["dataset_name"].value_counts()
        logging.info(f"\n{dupe_vcs}")

        duplicates_path = os.path.join(
            output_path, "aggregated_duplicates_hashes.csv"
        )

        df_duplicates_all.to_csv(duplicates_path, index=False)
        assert df_unique["hash"].is_unique, "Hashes are still not unique"

        df = df_unique.copy()

        logging.info(
            f"Duplicate hashes removed and stored in {duplicates_path}"
        )
    else:
        logging.info("No duplicate hashes found.")

    return df


def remove_duplicates_filenames(df, output_path):
    # duplicate filenames can hint at identical files, even though hashes
    # might differ (possible reasons: different encoding, bit depths)
    if not df["file_name"].is_unique:
        # get the unique files based on file names
        df_unique = df.drop_duplicates(subset="file_name")

        # find duplicates and store them in a csv
        df_duplicates_all = df[df.duplicated(subset="file_name", keep=False)]
        df_duplicates_all = df_duplicates_all.sort_values("file_name")

        logging.info(
            f"A total of {len(df_duplicates_all)} duplicate file names found in:"
        )
        dupe_vcs = df_duplicates_all["dataset_name"].value_counts()
        logging.info(f"\n{dupe_vcs}")

        duplicates_path = os.path.join(
            output_path, "aggregated_duplicates_filenames.csv"
        )

        df_duplicates_all.to_csv(duplicates_path, index=False)
        assert df_unique[
            "file_name"
        ].is_unique, "File names are still not unique"

        df = df_unique.copy()

        logging.info(
            f"Duplicate file names removed and stored in {duplicates_path}"
        )
    else:
        logging.info("No duplicate file names found.")

    return df


def check_species_names(df_all, df_species, args):
    # class names are expected to not contain any spaces
    if df_all["class_name"].str.contains(" ").any():
        logging.warning(
            "Column 'class_name' contains space characters. Remove spaces for optimal matching!"
        )
    # extract species names that cannot be found in the official nomenclatures
    df_not_conform = df_all[
        ~df_all["class_name"].isin(df_species["scientificNameNoSpace"])
    ]

    # get the source dataset for these non-conform species
    df_sources = (
        df_not_conform.groupby("class_name")["dataset_name"]
        .unique()
        .apply(lambda x: ", ".join(sorted(x)))
        .reset_index(name="source_datasets")
    )

    # get a value count to see which classes are most affected
    df_not_conform = df_not_conform["class_name"].value_counts().reset_index()
    df_not_conform.columns = ["class_name", "file_count"]
    logging.info(
        f"Found {len(df_not_conform)} species names that cannot be found in the species file ({sum(df_not_conform.file_count)} files total)."
    )

    df_not_conform = pd.merge(df_not_conform, df_sources, on="class_name")
    # drop NAs and duplicates in the prev_id_norm column
    # we can use this to map from previous identifications / species names
    # to current official nomenclature
    df_prev_names = df_species.drop_duplicates("prev_id_norm")

    # fuzzy match to scientific name
    df_not_conform["fuzzy_match_sci_name"] = df_not_conform["class_name"].apply(
        lambda x: fuzzy_match_taxa(
            x, df_species["scientificNameNoSpace"].tolist(), args.cutoff
        )
    )

    # log results
    num_matches_sci_name = len(df_not_conform["fuzzy_match_sci_name"].dropna())
    logging.info(
        f"Found {num_matches_sci_name} matches for {len(df_not_conform)} unknown scientfic names in current scientific names."
    )

    # fuzzy match to "previous names"
    df_not_conform["fuzzy_match_prev_name"] = df_not_conform[
        "class_name"
    ].apply(
        lambda x: fuzzy_match_taxa(
            x, df_prev_names["prev_id_norm"].tolist(), args.cutoff
        )
    )
    # log results
    num_matches_prev_sci_name = len(
        df_not_conform["fuzzy_match_prev_name"].dropna()
    )
    logging.info(
        f"Found {num_matches_prev_sci_name} matches for {len(df_not_conform)} unknown scientfic names in previous scientific names."
    )

    # add columns for user to provide manual mapping and comments
    df_not_conform["user_mapping"] = pd.NA
    # bypass the need for user_mapping to be filled in by providing identical values as "class_name"
    if args.keep_classnames is True:
        df_not_conform["user_mapping"] = df_not_conform["class_name"]
    df_not_conform["comment"] = pd.NA

    if args.use_mapping is not None:
        logging.info(f"Applying previous mapping {args.use_mapping}")
        df_not_conform = apply_previous_mapping(
            args.use_mapping, df_not_conform
        )

    # store species mapping file
    output_path = os.path.join(args.output_path, "species_name_mapping")
    if args.output_format == "xlsx":
        df_not_conform.to_excel(
            f"{output_path}.xlsx",
            index=False,
        )
    else:
        df_not_conform.to_csv(
            f"{output_path}.csv",
            index=False,
        )

    logging.info(f"Stored species name mappings file in {output_path}")
    logging.info(
        "Edit this file to specify how to handle unknown species names."
    )


def normalize_taxon(name):
    if pd.isna(name):
        return ""
    # Remove subgenus in parentheses, authors, dates, and punctuation
    name = re.sub(r"\w+, \d{4}$", "", name)  # removes "author, yyyy"
    name = re.sub(r"\(.*?\)", "", name)  # removes anything in brackets
    name = re.sub(r"[^a-zA-Z]", "", name)  # removes spaces, commas, dots, etc.
    return name


def fuzzy_match_taxa(query, choices, threshold=60):
    match = process.extractOne(query, choices, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0]
    return pd.NA


# =============================================
# Parse CMD arguments
# =============================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-nd",
        "--no-download",
        dest="redownload_file",
        action="store_false",
        default=True,
        help="Do not re-download the species file from the taxonworks servers of www.orthoptera.speciesfile.org and www.hoppers.speciesfile.org, use the cached copy (data/aggregation/species_file/cached_species_file_simple.csv) instead",
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=60,
        help="Cutoff value for the fuzzy matching of unknown species names to the species file",
    )

    parser.add_argument(
        "-m",
        "--dataset-metadata-path",
        default="./data/aggregation/dataset_metadata.yml",
        help="Metadata config file to use (specifies which datasets are included in the aggregation)",
    )

    parser.add_argument(
        "-op",
        "--output-path",
        default="./data/aggregation/initial/",
        help="Directory where the output is stored.",
    )

    parser.add_argument(
        "-of",
        "--output-format",
        choices=["xlsx", "csv"],
        default="xlsx",
        help="Format of the output file for editing of non-conform species names",
    )

    parser.add_argument(
        "-um",
        "--use-mapping",
        type=str,
        help="Path to an existing species_name_mapping to be reused here.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="More verbose output",
    )

    parser.add_argument(
        "-nk",
        "--no-keep-classnames",
        dest="keep_classnames",
        action="store_false",
        default=True,
        help="Do not inherit class names unchanged from the source datasets being aggregated. Species names not found in the species file are then dropped unless mapped manually in species_name_mapping.xlsx",
    )

    return parser.parse_args()


# =============================================
# MAIN
# =============================================


def main(args: argparse.Namespace):
    # make sure the species mapping file does not exist yet because it will be
    # overwritten otherwise, potentially destroying users work
    output_path = args.output_path

    spec_name_mapping_files = glob(
        os.path.join(output_path, "species_name_mapping.*")
    )
    assert (
        len(spec_name_mapping_files) == 0
    ), f"{' '.join(spec_name_mapping_files)} exists already. Remove or rename it and run this script again."

    os.makedirs(output_path, exist_ok=True)

    # load yaml file with paths to dataset annotations
    dataset_metadata = load_dataset_metadata_config(args.dataset_metadata_path)
    logging.info("Loaded dataset metadata path file.")

    # get all annotations into a single df
    df_all = collect_dataset_annotations(dataset_metadata, args.verbose)
    logging.info(f"Loaded all datasets ({len(df_all)} recordings).")

    # remove duplicates based on hashes and file names
    df_all = remove_duplicates_hashes(df_all, output_path)
    df_all = remove_duplicates_filenames(df_all, output_path)

    logging.info(f"Got {len(df_all['hash'])} unique recordings in total.")

    # load Orthoptera & Cicadidae species files
    df_species = download_species_file(args.redownload_file)

    # check species names and save an excel file for the user to provide mappings
    os.makedirs(output_path, exist_ok=True)
    check_species_names(df_all, df_species, args)

    # stores: df with annotations (do not update metadata yaml and do not
    # save a class_to_type dict / label mappings)

    store_dataset_metadata(
        "init_aggregated",
        output_path,
        df_all,
        class_to_type_dict=None,
        update_metadata_yml=False,
    )


# =============================================
# ENTRY
# =============================================
if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # configure logging
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    coloredlogs.install(level=logging_level)

    # Run main script
    main(args)

    # finalize
    logging.info("")
    logging.info(40 * "=")
    logging.info("DONE")
    logging.info(40 * "=")
