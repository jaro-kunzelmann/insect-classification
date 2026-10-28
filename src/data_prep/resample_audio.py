"""
Resample audio files to target sample rate and save resampled files.
"""

import logging
import coloredlogs
import argparse
import os
import gc
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample
import scipy
import librosa
import math
from tqdm import tqdm

from multiprocessing import Process, Queue
from more_itertools import chunked
from utils.utils_metadata import add_recordings_info, store_dataset_metadata
from utils.utils_functions import assign_codes_ordered_by_frequency

import psutil

# =============================================
# GLOBALS
# =============================================
LOG_MEM = False


# =============================================
# FUNCTIONS
# =============================================
def log_mem(tag):
    global LOG_MEM
    if not LOG_MEM:
        return
    # log memory usage by process ID with a tag
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e6
    logging.debug(f"[{tag}] PID {os.getpid()} memory usage: {mem:.2f} MB")


def get_resampler(orig_sr: int, target_sr: int, method_name: str):
    """
    Provides a torchaudio resample transform for the given parameters (sample rates, method)
    """
    if method_name == "default":
        # default is a librosa-style "kaiser_best" method (see https://pytorch.org/audio/master/tutorials/audio_resampling_tutorial.html)
        return Resample(
            orig_sr,
            target_sr,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
            dtype=torch.float32,
        )
    elif method_name == "torchaudio":
        return Resample(orig_sr, target_sr)
    elif method_name == "scipy":
        return lambda x: scipy.signal.resample_poly(x, target_sr, orig_sr)
    elif method_name == "librosa":
        # this is what was implicitly used in IS459 paper (soxr hq)
        # (https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet/blob/useIS459/notebooks/01_preprocess_waves.ipynb)
        return lambda x: librosa.resample(
            x, orig_sr=orig_sr, target_sr=target_sr
        )
    else:
        raise ValueError(f"Unknown resampling method: {method_name}")


def resample(args):
    """
    Load, resample, save an audio file using torchaudio functions. Possible that
    the memory issue is caused in here.
    """
    # unpack args
    orig_fpath, resample_fpath, sr_target, resampling_method = args
    resampled = False
    resampling_status = "failed"
    resampling_error = None
    log_mem("Initial")
    try:
        # load the audio file
        waveform_orig, sr_orig = torchaudio.load(orig_fpath)
        log_mem("After file load")

        # make sure we are always in float32 format
        if waveform_orig.dtype != torch.float32:
            # logging.info(f"Cast wave to float32 ({orig_fpath})")
            waveform_orig = waveform_orig.to(torch.float32)

        # resample only if the file does not already have the target SR
        if sr_orig != sr_target:
            # logging.info(f"Resampling wave ({orig_fpath})")
            # get a resampler and apply it to the file
            resampler = get_resampler(
                sr_orig, sr_target, method_name=resampling_method
            )

            # librosa needs the wave to be a numpy array
            if resampling_method == "librosa":
                waveform_orig = waveform_orig.numpy()
                # and back to torch tensor
                waveform_out = torch.from_numpy(resampler(waveform_orig))
            else:
                waveform_out = resampler(waveform_orig)

            resampled = True
            del resampler  # trying to avoid caching

        else:
            # no resampling, but we will still save the wave in the standard
            # format
            waveform_out = waveform_orig

        torchaudio.save(
            resample_fpath,
            waveform_out,
            sr_target,
            bits_per_sample=16,
            encoding="PCM_S",
        )

        log_mem("After file save")
        # make sure the memory is freed. not sure this is needed nor works.
        del waveform_orig, waveform_out
        # call the Python garbage collector. This should remove objects that
        # are no longer used from memory.
        gc.collect()

        # log_mem("Before return (try block)")
        resampling_status = "ok"
        resampling_error = None

    except Exception as e:
        # "collect" also in the exception case and report failure via the
        # returned dict
        gc.collect()
        logging.warning(f"Failed to resample {orig_fpath}: {e}")
        # log_mem("Before return (Exception block)")
        resampling_status = "failed"
        resampling_error = str(e)

    # ensures we always return (=put in queue) in the exact same format
    finally:
        result = {
            "path": orig_fpath,
            "resampled": resampled,
            "resampling_status": resampling_status,
            "resampling_error": resampling_error,
        }
        return result


def resample_with_return(args, queue):
    # wrapper function for resample to be used with "Process"
    result = resample(args)
    queue.put(result)


def resample_dataset(df, target_sr, max_workers, resampling_method):
    """
    Resample the entire dataset in parallel (using multiprocessing.Pool)
    """
    global LOG_MEM
    # Prepare tasks (basically function args)
    # (orig_fpath, resample_fpath, sr_target, resampling_method)
    tasks = [
        (
            row["fpath_audio"],
            row["fpath_resampled"],
            target_sr,
            resampling_method,
        )
        for i, row in df.iterrows()
    ]
    # chunked parallelism with capped concurrency so torchaudio and it's backend
    # are unable to accumulate memory.
    # divides our tasks into chunks of length max_workers, which will correspond
    # the number of processes that will be started
    chunks = list(chunked(tasks, max_workers))

    logging.info(
        f"Tasks created. Starting resampling with {max_workers} worker{'s'*(max_workers>1)} ({len(chunks)} chunks) using method '{resampling_method}'"
    )

    # add tqdm bar only if we are not logging mem
    if not LOG_MEM:
        chunk_iter = tqdm(chunks, total=math.ceil(len(tasks) / max_workers))
    else:
        chunk_iter = chunks

    results = []
    for chunk in chunk_iter:
        queue = Queue()
        procs = []
        # a process is started for every task (=file to resample)
        for args in chunk:
            p = Process(target=resample_with_return, args=(args, queue))
            p.start()
            procs.append(p)

        # wait for all to end
        # this is probably a bit of bottleneck if one task takes longer than
        # all the others, but at least we don't get memory bloat
        # a more dynamic system would be possible, but needs more code which
        # decreases readability & maintainability
        for p in procs:
            p.join()

        # empty the queue into results list
        while not queue.empty():
            results.append(queue.get())

    # join the results onto the original dataset
    df_results = pd.DataFrame(results)
    df = df.merge(
        df_results, left_on="fpath_audio", right_on="path", how="left"
    )

    cols_to_rename = {
        "directory": "orig_directory",
        "bits_per_sample": "orig_bits_per_sample",
        "num_frames": "orig_num_frames",
        "audio_length": "orig_audio_length",
        "num_channels": "orig_num_channels",
        "fpath_audio": "orig_fpath_audio",
        "sample_rate": "orig_sample_rate",
        "encoding": "orig_encoding",
        "hash": "orig_hash",
        "file_name": "orig_file_name",
        "is_lossy": "orig_is_lossy",
        "fpath_resampled": "fpath_audio",
    }
    df.rename(cols_to_rename, axis=1, inplace=True)

    logging.info("Done resampling. Adding recordings info.")

    df = add_recordings_info(df)
    logging.info("Added recordings info")

    return df


# =============================================
# Parse CMD arguments
# =============================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--annotation-file",
        type=str,
        help="Path to annotation file to use",
        required=True,
    )

    parser.add_argument(
        "-sr",
        "--target-sample-rate",
        type=int,
        help="The sample rate to resample the audio to",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Path to output directory where resampled files will be stored",
        default="./data/aggregation/resampled_files/",
    )

    parser.add_argument(
        "-w",
        "--max-workers",
        type=int,
        default=1,
        help="Number of workers/process to use in parallel for resampling. To be safe, do not assign more workers than you have GBs of RAM available. Defaults to 1 (no parallelized resampling)",
    )

    parser.add_argument(
        "-rm",
        "--resampling-method",
        choices=["default", "torchaudio", "scipy", "librosa"],
        default="default",
        help="The resampling method to use. Default is a librosa-style 'kaiser_best' method (using torchaudio.transforms.Resampling). The other methods are the default methods of the respective libraries.",
    )

    parser.add_argument(
        "-l",
        "--log-memory",
        action="store_true",
        help="Log memory usage of each worker process at different stages of the resampling. This will set the logging level to DEBUG.",
    )

    return parser.parse_args()


# =============================================
# MAIN
# =============================================


def main(args: argparse.Namespace):
    # raise an error if the output directory already exists - this is to prevent
    # overwriting of files...
    global LOG_MEM
    if args.log_memory:
        LOG_MEM = True
    output_dir = args.output_dir
    output_dir_files = os.path.join(output_dir, "files")

    if os.path.isdir(output_dir):
        raise FileExistsError(
            f"Output directory {output_dir} already exists! Rename the existing directory or pass another directory to --output-dir"
        )
    os.makedirs(output_dir_files)
    # read the annoation file
    df = pd.read_csv(args.annotation_file)

    logging.info(
        f"Loaded {args.annotation_file} with {len(df)} rows ({df['class_name'].nunique()} classes)."
    )

    # create filenames and paths of resampled files (to be created) and do some sanity checks
    assert all(
        df["fpath_audio"].apply(os.path.isfile)
    ), "Some audio files referenced in the annotation file were not found!"
    logging.info(
        "All audio files referenced in the annotation file were found."
    )

    df["orig_fname_no_ext"] = df["file_name"].apply(
        lambda x: os.path.splitext(x)[0]
    )
    assert df["orig_fname_no_ext"].notna().all(), "Empty file names encountered"
    df["fname_resampled"] = df["orig_fname_no_ext"] + ".wav"
    df["fpath_resampled"] = df["fname_resampled"].apply(
        lambda x: os.path.join(output_dir_files, x)
    )
    df = df.drop(columns=["orig_fname_no_ext"])
    assert df[
        "fpath_resampled"
    ].is_unique, "File paths to resampled files are not unique. Some files will be overwritten & lost."

    logging.info("File paths to resampled files constructed.")

    # resample entire dataset
    df = resample_dataset(
        df,
        target_sr=args.target_sample_rate,
        max_workers=args.max_workers,
        resampling_method=args.resampling_method,
    )

    # check if the resampled files exist and warn if not
    df["resampled_exists"] = df["fpath_audio"].apply(os.path.isfile)

    if not all(df["resampled_exists"]):
        logging.warning("Not all resampled files found!")

    else:
        logging.info("All resampled files found")

    # Store updated annotations
    # Add the class column and create a dictionary to map the class to the species
    df_out, class_to_type_dict = assign_codes_ordered_by_frequency(
        df, "class_name", "class", verbose=True
    )
    # store label mappings and annotation csv
    store_dataset_metadata(
        "resampled",
        output_dir,
        df_out,
        class_to_type_dict,
        update_metadata_yml=False,
    )


# =============================================
# ENTRY
# =============================================
if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # configure logging
    if args.log_memory:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    coloredlogs.install(level=logging_level)

    # Run main script
    main(args)

    # finalize
    logging.info("")
    logging.info(40 * "=")
    logging.info("DONE")
    logging.info(40 * "=")
