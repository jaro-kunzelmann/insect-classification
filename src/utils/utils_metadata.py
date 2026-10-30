import os
import hashlib
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import torchaudio
from tqdm import tqdm
from typing import Dict, Any
import yaml
import logging


def extract_xml_info(xml_path):

    if xml_path == "nan" or xml_path == "None" or isinstance(xml_path, float):
        return {
            "Date": None,
            "Time": None,
            "Duration": None,
            "Temperature": None,
            "Latitude": None,
            "Longitude": None,
        }

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract the DateTime, Duration, and Temperature information
    DateTime = root.find(".//DateTime").text
    Date, Time = DateTime.split(" ")
    Duration = root.find(".//Duration").text
    Duration = float(Duration.split(" ")[0])
    Temperature = root.find(".//Temperature").text
    Temperature = float(Temperature.split(" ")[0])

    # Extract the GPS Position information
    GPS = root.find(".//GPS")
    Valid = GPS.find(".//Valid").text
    if Valid == "yes":
        Position = GPS.find(".//Position").text
        Latitude, Longitude = Position.split(" ")
        Latitude = float(Latitude)
        Longitude = float(Longitude)
    else:
        Latitude = None
        Longitude = None

    # Return a dictionary with the extracted information
    return {
        "Date": Date,
        "Time": Time,
        "Duration": Duration,
        "Temperature": Temperature,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }


def add_recordings_info(df):
    """
    adds torchaudio.info metadata to a df containing a column "fpath_audio" with the
    path to the files
    """

    tqdm.pandas()

    with pd.option_context("mode.chained_assignment", None):
        df.loc[:, "info"] = df.progress_apply(
            lambda row: torchaudio.info(row["fpath_audio"]),
            axis=1,
        )

    # all MP3 files (e.g. in InsectSet459) will have encoding "UNKNOWN"
    # according to torchaudio.info and all "UNKNOWN" encodings are MP3 files,
    # and bits per sample is returned as 0.
    df_temp = pd.DataFrame(
        {
            "bits_per_sample": df["info"].apply(lambda x: x.bits_per_sample),
            "encoding": df["info"].apply(lambda x: x.encoding),
            "num_channels": df["info"].apply(lambda x: x.num_channels),
            "num_frames": df["info"].apply(lambda x: x.num_frames),
            "sample_rate": df["info"].apply(lambda x: x.sample_rate),
        }
    )
    # replace 0 with NA
    df_temp["bits_per_sample"] = df_temp["bits_per_sample"].replace(0, np.nan)

    # compute audio length and return concatenated df
    df_temp["audio_length"] = df_temp["num_frames"] / df_temp["sample_rate"]
    df = pd.concat((df, df_temp), axis=1)
    df.drop("info", axis=1, inplace=True)

    return df


def add_recordings_info_swiss(df_audio, column_name="xml"):
    """
    adds torchaudio.info metadata to swiss dataset and also extracts xml info
    from the recording devices
    """
    tqdm.pandas()

    df_metadata = df_audio.apply(
        lambda row: pd.Series(extract_xml_info(row[column_name])), axis=1
    )
    df_audio["info"] = df_audio["wav"].progress_apply(torchaudio.info)

    df = pd.DataFrame(
        {
            "bits_per_sample": df_audio["info"].apply(
                lambda x: x.bits_per_sample
            ),
            "encoding": df_audio["info"].apply(lambda x: x.encoding),
            "num_channels": df_audio["info"].apply(lambda x: x.num_channels),
            "num_frames": df_audio["info"].apply(lambda x: x.num_frames),
            "sample_rate": df_audio["info"].apply(lambda x: x.sample_rate),
        }
    )

    df_audio = pd.concat((df_audio, df_metadata, df), axis=1)
    df_audio.drop("info", axis=1, inplace=True)
    df_audio["audio_length"] = df_audio["num_frames"] / df_audio["sample_rate"]
    df_audio.insert(len(df_audio.columns) - 1, "type", df_audio.pop("type"))

    return df_audio


def compute_hash(file_path: str, algorithm: str = "md5", audio_only=True):
    """
    Computes the hash of a file using the specified hash algorithm.

    Args:
        file_path (str): The path to the file.
        algorithm (str): The hash algorithm to use (e.g., 'md5', 'sha256'). Defaults to 'md5'.
        audio_only (bool): Loads the raw audio content of the file, thereby ignoring the header content. Default True.

    Returns:
        str: The hexadecimal hash of the file.

    Raises:
        ValueError: If the specified hash algorithm is not supported.
        IOError: If there is an error reading the file.
    """
    try:
        # Dynamically get the hash constructor from hashlib
        hash_func = getattr(hashlib, algorithm)()
    except AttributeError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    try:
        if audio_only:
            # slower, less elegant, but should be robust to header changes
            waveform, sr = torchaudio.load(file_path)
            hash_func.update(waveform.numpy().tobytes())

        else:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hash_func.update(chunk)
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")

    return hash_func.hexdigest()


def load_dataset_metadata_config(
    path_dataset_metadata: str = "./data/aggregation/dataset_metadata.yml",
) -> Dict[str, Dict[str, Any]]:
    """Load dataset metadata from a YAML file.

    Returns an empty dictionary if the file does not exist or is invalid.
    """
    if not os.path.isfile(path_dataset_metadata):
        return {}

    try:
        with open(path_dataset_metadata, "r") as file:
            return yaml.safe_load(file) or {}
    except yaml.YAMLError:
        return {}


def store_dataset_metadata(
    dataset_name: str,
    root_directory: str,
    df: pd.DataFrame,
    class_to_type_dict: Dict[str, Dict[str, Any]],
    update_metadata_yml=True,
):
    """Save class label mapping (yml file), annotations_* (csv with annotations)
    and update the paths to both in data/aggregation/dataset_metadata.yml
    """
    # save .yml and .csv files in a separate folder to prevent cluttering
    # of data folders
    meta_dirpath = os.path.join(root_directory, "meta")
    os.makedirs(meta_dirpath, exist_ok=True)
    # Save the class mapping to a YAML file
    path_dict = None
    if class_to_type_dict is not None:
        path_dict = os.path.join(
            meta_dirpath, f"label_mappings_{dataset_name}.yml"
        )
        with open(path_dict, "w") as file:
            yaml.dump(class_to_type_dict, file)

        logging.info(f"Saved label mapping dict as {path_dict}.")

    # Store the dataframe as a csv file
    path_csv = os.path.join(meta_dirpath, f"annotations_{dataset_name}.csv")
    df.to_csv(path_csv, index=False)
    logging.info(f"Saved metadata csv-file to {path_csv}.")

    # update the dataset metadata, if desired
    if update_metadata_yml:
        # Store paths to annotation and class files in dataset_annotation_files
        new_dataset_metadata = {
            dataset_name: {"classes": path_dict, "labels": path_csv}
        }
        path_dataset_metadata = "./data/aggregation/dataset_metadata.yml"
        os.makedirs(os.path.dirname(path_dataset_metadata), exist_ok=True)
        dataset_metadata = load_dataset_metadata_config()
        dataset_metadata.update(new_dataset_metadata)

        with open(path_dataset_metadata, "w") as file:
            yaml.dump(dataset_metadata, file)

        logging.info(
            f"Updated dataset metadata ({path_dataset_metadata}) with paths to dataset annotation file and label mapping dict"
        )

    return path_csv, path_dict


def collect_dataset_annotations(
    dataset_metadata: Dict[str, Dict[str, Any]], debug: bool = False
) -> pd.DataFrame:
    """Collects all dataset annotations and metadata into a single dataframe"""
    dfs = []

    for dataset_name, paths in dataset_metadata.items():
        logging.debug(f"Dataset name: {str(dataset_name)}")
        # class_to_type file
        classes_path = paths["classes"]
        # annotation csv
        labels_path = paths["labels"]

        # load class_to_type dict
        with open(classes_path, "r") as file:
            class_to_type_dict = yaml.safe_load(file) or {}

        # read the annotation dataframe
        df = pd.read_csv(labels_path)
        df["dataset_name"] = dataset_name

        # add class name from the dict
        df["class_name"] = df["class"].map(class_to_type_dict)

        dfs.append(df)

    # only keep the common columns for now, otherwise it's a mess of 44 columns
    # with many NAs
    # maybe include "hash" as well, which has already been computed for
    # swiss recordings and we don't want to do it again...
    # Compute all unique columns
    common_columns = set.intersection(*(set(df.columns) for df in dfs))

    # we don't need "hash" in ESC-50, so we add it back in
    common_columns = list(common_columns) + ["rec_device", "directory"]

    # ensure all dfs have the same columns
    dfs = [df.reindex(columns=common_columns, fill_value=pd.NA) for df in dfs]
    df_combined = pd.concat(dfs, ignore_index=True)

    # strip underscores from class names to get a more uniform nomenclature
    # (InsectSet47 and 66 do not have them)
    df_combined["class_name"] = df_combined["class_name"].str.replace("_", "")

    # lower case class names
    # df_combined["class_name"] = df_combined["class_name"].str.lower()

    # Train -> train etc. (harmonize datasets)
    df_combined["subset"] = df_combined["subset"].str.lower()

    # check if all files have a subset split
    # the split would have to be added during metadata_prep*
    if df_combined["subset"].hasnans:
        df_subset_na = df_combined[df_combined["subset"].isna()]
        logging.warning(
            "NAs encountered in column 'subset' of dataset (number of files):"
        )
        vc = df_subset_na["dataset_name"].value_counts()
        logging.warning(vc)

    # extract file extensions (only .wav and .mp3 are supported)
    unique_extensions = (
        df_combined["file_name"]
        .apply(lambda x: os.path.splitext(x)[1].lower())
        .unique()
    )

    # check that not other file extension exist in the dataframe
    other_extensions = set(unique_extensions) - {".wav", ".mp3"}
    assert (
        len(other_extensions) == 0
    ), f"Found unexpected file extensions: {other_extensions}"

    # add the lossy column for .mp3s
    df_combined["is_lossy"] = df_combined["file_name"].str.endswith(".mp3")

    # check dirs and files exist
    if debug:
        # note: "directory" is not stored for every dataset, so NAs are dropped
        # here and only the datasets that have one are checked.
        dirs = list(df_combined["directory"].dropna().unique())
        assert all(
            [os.path.isdir(d) for d in dirs]
        ), "Not all dataset folders found"

        assert all(
            df_combined["fpath_audio"].apply(os.path.isfile)
        ), "Not all dataset files found"

        logging.debug("All dataset files found.")

    return df_combined
