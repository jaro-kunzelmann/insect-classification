"""
Apply user-defined species name handling ("species_name_mapping.{csv | xlsx}")
to taxonomically unclear species names in initial dataset aggregation
(init_aggregated_annotations.csv).
- TODO make analyze_freq an optional step beforehand to find out which classes to kick out.
"""

import os
import logging
import coloredlogs
import argparse
import pandas as pd
from utils.utils_metadata import store_dataset_metadata
from utils.utils_functions import (
    assign_codes_ordered_by_frequency,
    apply_user_filters,
    apply_user_mappings,
    load_config,
    save_config,
)


# =============================================
# Parse CMD arguments
# =============================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--annotations-path",
        default="./data/aggregation/initial/meta/annotations_init_aggregated.csv",
        help="Path to initial annotations file to be processed",
    )

    parser.add_argument(
        "-m",
        "--mappings-path",
        default="./data/aggregation/initial/species_name_mapping.xlsx",
        help="Path to species name mappings",
    )

    parser.add_argument(
        "-fc",
        "--filter-config-path",
        help="""Path to a yaml config file containing filters to apply to init_aggregated_annotations.csv
        The filename (without file extension) will serve as the name of this filter.
        By default, no filters are applied, which is useful for creating an initial overview.
        """,
    )

    parser.add_argument(
        "-s",
        "--strict",
        action="store_true",
        help="Pass this flag to run in strict mode which will produce more verbose output and warnings on filtering (default False)",
    )

    parser.add_argument(
        "-ow",
        "--overwrite",
        action="store_true",
        help="Pass this flag to overwrite previously produced output (default False). An Error will be raised if set to False and output already exists",
    )

    return parser.parse_args()


# =============================================
# MAIN
# =============================================


def main(args: argparse.Namespace):
    root_directory = "./data/aggregation/filtered/"
    if args.filter_config_path:
        filter_config = load_config(args.filter_config_path)
        # parse the filter name from the filter config file
        filter_name = os.path.splitext(
            os.path.basename(args.filter_config_path)
        )[0]
    else:
        filter_name = "no_filters"

    # create path to folder for this scripts output
    filter_out_path = os.path.join(root_directory, filter_name)

    # check if the output path exists before calling creatiing it, store_dataset_metadata
    if os.path.isdir(filter_out_path) and not args.overwrite:
        raise FileExistsError(
            f"Annotation folder {filter_out_path} already exists. Pass --overwrite to overwrite output."
        )
    os.makedirs(filter_out_path, exist_ok=True)

    # load the files, apply label mappings
    df_aggregated = pd.read_csv(args.annotations_path, low_memory=False)
    logging.info(
        f"Loaded annotations_init_aggregated.csv with {len(df_aggregated)} rows ({df_aggregated['class_name'].nunique()} classes)."
    )
    df_clean = apply_user_mappings(df_aggregated, args.mappings_path)

    # apply filters to the df, if specified
    if filter_name == "no_filters":
        df_out = df_clean
        logging.info("No filter applied.")

    else:
        # filter the df using the filters in the yaml
        df_out = apply_user_filters(
            df_clean, filter_config.copy(), args.strict
        ).copy()
        logging.info(
            f"Applied filter '{filter_name}' at {args.filter_config_path}"
        )

        logging.info(
            f"Filtered annotations contain {len(df_out)} rows ({df_out['class_name'].nunique()} classes)."
        )

    # Store updated annotations
    # Add the class column and create a dictionary to map the class to the species
    df_out, class_to_type_dict = assign_codes_ordered_by_frequency(
        df_out.copy(), "class_name", "class", verbose=True
    )
    # store label mappings and annotation csv
    store_dataset_metadata(
        filter_name,
        filter_out_path,
        df_out,
        class_to_type_dict,
        update_metadata_yml=False,
    )

    if not filter_name == "no_filters":
        # store the used config with the output
        config_fname = filter_name + ".yml"
        save_config(filter_config, filter_out_path, config_fname)
        logging.info(f"Saved filtering config at {filter_out_path}")

    # fstring voodoo
    logging.info(
        f"Saved {(not(bool(filter_out_path)))*'un'}filtered annotations with {len(df_out)} rows ({df_out['class_name'].nunique()} classes) at {filter_out_path}"
    )


# =============================================
# ENTRY
# =============================================
if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # configure logging
    logging_level = logging.INFO
    coloredlogs.install(level=logging_level)

    # Run main script
    main(args)

    # finalize
    logging.info("")
    logging.info(40 * "=")
    logging.info("DONE")
    logging.info(40 * "=")
