"""
This script provides metadata for InsectSet(47, 66, 459) audio files and stores them as a csv for later use.
"""

import argparse
import os
import logging
import coloredlogs
import pandas as pd
from tqdm import tqdm

from utils.utils_functions import assign_codes_ordered_by_frequency
from utils.utils_metadata import (
    add_recordings_info,
    store_dataset_metadata,
    compute_hash,
)

tqdm.pandas()

# =============================================
# Global Variables
# =============================================

# maps number identifier to folder name
DATASETS = {
    32: "InsectSet32",
    47: "InsectSet47_Train_Val_Test",
    66: "InsectSet66_Train_Val_Test",
    459: "InsectSet459_Train_Val_Test",
}


# =============================================
# Functions
# =============================================


def __log_config():
    logging_level = logging.INFO
    coloredlogs.install(level=logging_level)


def load_insect_annotations(insect_set_name, root_directory, subdir):
    """Load the InsectSet annotations from their respective folders and make
    some slight changes to column names and nomenclatures to harmonize them for later
    """

    def read_csv_in_subdir(filename):
        return pd.read_csv(os.path.join(root_directory, subdir, filename))

    def read_csv_at_root(filename):
        return pd.read_csv(os.path.join(root_directory, filename))

    if insect_set_name == "InsectSet459":
        annotation_file = f"{insect_set_name}_Train_Val_Test_Annotation.csv"
        df = read_csv_in_subdir(annotation_file)

        # also, the species label nomenclature differs:
        # in 459 it is <genus>_<species
        # in other two it is <genus><species> (no underscore)
        # make "back up" column with the original species name
        df.rename({"species_name": "species_name_orig"}, axis=1, inplace=True)

        # remove separating underscore and insert new column directly next
        # to "species_name_orig"
        df.insert(
            df.columns.get_loc("species_name_orig") + 1,
            "species_name",
            df["species_name_orig"].str.replace("_", ""),
        )

    elif insect_set_name == "InsectSet32":
        files = ["Orthoptera.csv", "Cicadidae.csv"]
        dfs = []
        for f in files:
            df = read_csv_in_subdir(f)

            label = os.path.splitext(f)[0]
            # Extract "Orthoptera" or "Cicadidae"
            df["taxon_group"] = label
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # rename column to harmonize with other datasets
        df.rename({"data_set": "subset"}, axis=1, inplace=True)

    else:
        # 47 and 66
        annotation_file = f"{subdir}_Annotation.csv"
        df = read_csv_at_root(annotation_file)

    logging.info(f"Found {len(df)} annotations for {insect_set_name}")

    return df


def add_directory(insect_set, root_directory, subdir, df_insects):

    # again, file paths are different for 459 than 47 & 66
    if insect_set == 459:
        # audio files are in "Train" and "Validation" subdirs, respectively
        df_insects["directory"] = df_insects["subset"].apply(
            lambda x: os.path.join(root_directory, subdir, x)
        )

    elif insect_set == 32:
        # audio files are in "Orthoptera" and "Cicadidae" subdirs, respectively
        df_insects["directory"] = df_insects.apply(
            lambda row: os.path.join(
                root_directory, subdir, row["taxon_group"], row["species"]
            ),
            axis=1,
        )

    else:
        # all audio files are in the same directory for 47 & 66
        df_insects["directory"] = os.path.join(root_directory, subdir)

    # Rearrange the columns
    df_insects = df_insects[
        ["insect_set", "directory"]
        + [
            col
            for col in df_insects.columns
            if col not in ["insect_set", "directory"]
        ]
    ]

    return df_insects


# =============================================
# Parse CMD arguments
# =============================================
def parse_args(args=None) -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--insect",
        type=int,
        choices=[32, 47, 66, 459],
        default=459,
        help="Select InsectSet 32, 47, 66, or 459",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="./data/InsectSet/",
        help="path of the data directory.",
    )

    return parser.parse_args(args)


# =============================================
# MAIN
# =============================================


def main(args: argparse.Namespace):
    """Main script code"""

    # insect set 32, 47, 66, or 459
    insect_set = args.insect
    insect_set_name = "InsectSet" + str(insect_set)

    # directory
    root_directory = args.directory

    # Map the insect_set argument to the corresponding dataset
    subdir = DATASETS[insect_set]

    # load the dataset annotations
    df_insects = load_insect_annotations(
        insect_set_name, root_directory, subdir
    )

    # Add the insect set name
    df_insects["insect_set"] = insect_set_name

    # ...and the file directory to the dataframe
    df_insects = add_directory(insect_set, root_directory, subdir, df_insects)

    # add the file path to the audio files
    df_insects["fpath_audio"] = df_insects.apply(
        lambda row: os.path.join(row["directory"], row["file_name"]), axis=1
    )

    # Add the recordings info (torchinfo metadata)
    df_insects = add_recordings_info(df_insects)
    logging.info("Added recordings info.")

    # column name for the species differs between 459 and 32, 47 & 66
    insect_set_to_label_colname = {
        32: "species",
        47: "species",
        66: "species",
        459: "species_name",
    }
    label_column = insect_set_to_label_colname[insect_set]

    # Add the class column and create a dictionary to map the class to the species
    df_insects, class_to_type_dict = assign_codes_ordered_by_frequency(
        df_insects, label_column, "class", verbose=True
    )
    # this is in case the datasets are to be used on their own - will get overridden when aggregating
    df_insects["class_name"] = df_insects[label_column]

    # compute hashes for later
    df_insects["hash"] = df_insects["fpath_audio"].progress_apply(compute_hash)
    logging.info("Computed hashes.")

    # stores: df with annotations, class dict, a yml file with locations of both
    store_dataset_metadata(
        insect_set_name, root_directory, df_insects, class_to_type_dict
    )


# =============================================
# ENTRY
# =============================================

if __name__ == "__main__":
    __log_config()
    # Parse the command line arguments
    args = parse_args()

    # Run main script
    main(args)

    # finalize
    logging.info(40 * "=")
    logging.info("DONE")
    logging.info(40 * "=")
