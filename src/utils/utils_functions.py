# standard modules
import os
import time
import logging
import platform
import random
import io
import contextlib
import subprocess
import warnings
from pathlib import Path
from pprint import pformat

# DEBUG
import psutil
from pympler import asizeof

# yaml for config files
import yaml

# typehinting
from typing import Dict, Optional, Mapping, Sequence

# torch
from torch import optim
import torch
from torchinfo import summary

# numpy, sklearn and pandas
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# wandb for logging
import wandb

# custom modules
from utils.utils_plotting import create_cm

from utils.utils_config import load_config
from utils.utils_metrics import (
    prepare_labels,
    EvalOutputs,
    AGGREGATION_REGISTRY,
    aggregate_file_prob_mean,
)

#: Repository root (this file lives at <root>/src/utils/utils_functions.py)
REPO_ROOT = Path(__file__).resolve().parents[2]

# =============================================


def apply_balanced_subset_sampling(
    df: pd.DataFrame, subset_ratio: float, random_seed: int
) -> pd.DataFrame:
    """
    Applies stratified random sampling to the DataFrame based on the 'subset' column.

    Args:
        df: The input DataFrame containing the 'subset' column.
        subset_ratio: The fraction of data to sample (0 < ratio <= 1).
        random_seed: The seed to use for reproducibility.

    Returns:
        The subsampled DataFrame.
    """
    if not 0 < subset_ratio <= 1:
        raise ValueError(
            f"Subset ratio must be between 0 and 1, got {subset_ratio}"
        )

    # Standardize the subset column before sampling
    df["subset"] = df["subset"].str.lower()

    try:
        # Perform stratified sampling using the 'subset' column
        df_sampled = df.groupby("subset", group_keys=False).sample(
            frac=subset_ratio,
            random_state=random_seed,
            replace=False,
        )
    except ValueError as e:
        logging.error(f"Could not perform balanced sampling: {e}")
        logging.warning("Falling back to unstratified random sampling.")
        df_sampled = df.sample(frac=subset_ratio, random_state=random_seed)

    df_sampled.reset_index(drop=True, inplace=True)
    logging.info(f"Total samples after subsetting: {len(df_sampled)}.")

    return df_sampled


def log_sizeof(tag: str, obj) -> None:
    """
    Function useful for debugging memory issues.
    Log the approximate memory footprint of an object in MB.
    Uses correct methods for tensors, numpy arrays, dataframes, and generic objects.
    """
    try:
        # --- PyTorch tensors ----------------------------------------
        if isinstance(obj, torch.Tensor):
            size_bytes = obj.element_size() * obj.nelement()
            size_mb = size_bytes / 1024**2
            logging.info(
                "[SIZEOF %s] Tensor: %.2f MB (%d elements, %d bytes each)",
                tag,
                size_mb,
                obj.nelement(),
                obj.element_size(),
            )

        # --- NumPy arrays --------------------------------------------
        if isinstance(obj, np.ndarray):
            size_mb = obj.nbytes / 1024**2
            logging.info("[SIZEOF %s] NumPy array: %.2f MB", tag, size_mb)

        # --- Pandas DataFrames ---------------------------------------
        if isinstance(obj, pd.DataFrame):
            size_bytes = obj.memory_usage(deep=True).sum()
            size_mb = size_bytes / 1024**2
            logging.info("[SIZEOF %s] DataFrame: %.2f MB", tag, size_mb)

        # --- Python list of tensors ----------------------------------
        if (
            isinstance(obj, list)
            and len(obj) > 0
            and isinstance(obj[0], torch.Tensor)
        ):
            size_bytes = sum(t.element_size() * t.nelement() for t in obj)
            size_mb = size_bytes / 1024**2
            logging.info(
                "[SIZEOF %s] List[Tensor]: %.2f MB (%d tensors)",
                tag,
                size_mb,
                len(obj),
            )

        # --- Python dict of tensors ----------------------------------
        if isinstance(obj, dict) and len(obj) > 0:
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                size_bytes = sum(
                    v.element_size() * v.nelement() for v in obj.values()
                )
                size_mb = size_bytes / 1024**2
                logging.info(
                    "[SIZEOF %s] Dict[str, Tensor]: %.2f MB (%d tensors)",
                    tag,
                    size_mb,
                    len(obj),
                )

        # --- Fallback for everything else -----------------------------
        size_mb = asizeof.asizeof(obj) / 1024**2
        logging.info(
            "[SIZEOF %s] %s: %.2f MB", tag, obj.__class__.__name__, size_mb
        )

    except Exception as e:
        logging.warning("[SIZEOF %s] failed to measure size: %s", tag, e)


def log_memory(tag: str = "") -> None:
    """
    Function useful for debugging memory issues.
    Log memory used by the CPU (main process + children) as well as
    GPU CUDA allocated and reserved memory
    """
    root = psutil.Process(os.getpid())
    procs = [root] + root.children(recursive=True)

    total_rss = 0
    for p in procs:
        try:
            total_rss += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass

    total_gb = total_rss / 1024**3

    msg = f"[MEM {tag}] CPU total (proc + children): {total_gb:.2f} GB"
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        msg += (
            f" | GPU alloc: {alloc_gb:.2f} GB, reserved: {reserved_gb:.2f} GB"
        )
    logging.info(msg)


def assign_codes_based_on_previous_mapping(
    df,
    class_to_type_dict,
    num_classes,
    label_column="class_name",
    code_column="class",
    reassign_unknown=False,
):
    """
    Update the class codes [0...num_classes-1] in the dataset according to
    a previous class_to_type_dict, and perform some sanity checks
    """
    assert num_classes is not None, "Missing 'num_classes' in model checkpoint."
    assert len(class_to_type_dict) == num_classes, (
        f"Mismatch: {len(class_to_type_dict)} classes in class_to_type_dict, "
        f"but checkpoint says {num_classes} classes"
    )

    # check if species sets differ between class_to_type_dict and df
    dataset_labels = set(df[label_column].unique())
    mapping_labels = set(class_to_type_dict.values())
    missing = mapping_labels - dataset_labels
    extra = dataset_labels - mapping_labels

    if missing:
        logging.warning(
            f"Some labels in class mapping not present in dataset: {missing}"
        )

    if extra:
        logging.warning(f"Some dataset labels not in class mapping: {extra}")
        if reassign_unknown:
            df.loc[df[label_column].isin(extra), label_column] = "other_insect"

    # assign code column based on class_to_type dict:
    # switch classes and labels
    type_to_class_dict = {v: k for k, v in class_to_type_dict.items()}
    df[code_column] = df[label_column].map(type_to_class_dict)

    return df


def timeit(name="function"):
    """
    Time a passed function
    """

    def decorator(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            logging.info(f"[TIMER] {name} took {round(te - ts, 2)} seconds")
            return result

        return timed

    return decorator


def set_optimizer(
    # optimizer_name: str,
    model,
    config_training,
    freeze_base=False,
    head_name="classifier",
):
    """Sets the optimizer for the experiment"""
    optimizers = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "nadam": optim.NAdam,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
    }

    # grab config params
    weight_decay = config_training.get("weight_decay", 0.0)
    learning_rate = config_training.get("learning_rate", None)

    # optimizer_name from config
    optimizer_name = config_training.get("optimizer", None)

    # get named parameters
    named_params = list(model.named_parameters())
    # only pass unfrozen params to the optimizer
    if freeze_base:
        named_params = [(n, p) for n, p in named_params if p.requires_grad]

    try:
        optimizer_class = optimizers[optimizer_name]
        if isinstance(learning_rate, dict):
            # this implies there are two separate learning rates, one for head & one for base
            base_lr = learning_rate["base_lr"]
            head_lr = learning_rate["head_lr"]

            # separate base and head
            base_params = []
            head_params = []
            prefix = head_name + "."
            for name, param in named_params:
                if name.startswith(prefix):
                    head_params.append(param)
                else:
                    base_params.append(param)

            # update the learning rates for the model params
            model_params = [
                {"params": base_params, "lr": base_lr},
                {"params": head_params, "lr": head_lr},
            ]
            # initialize optimizer with two separate LRs
            optimizer = optimizer_class(model_params, weight_decay=weight_decay)
            logging.info(
                f"Set learning rate for model base (backbone): {base_lr}"
            )
            logging.info(
                f"Set learning rate for model head (classifier): {head_lr}"
            )

        elif isinstance(learning_rate, float):
            # initialize optimizer with a single LR
            params_only = [p for _, p in named_params]
            optimizer = optimizer_class(
                params_only, lr=learning_rate, weight_decay=weight_decay
            )
            logging.info(
                f"Set single learning rate for all unfrozen parameters: {learning_rate}"
            )

    except KeyError:
        logging.error(
            "Invalid optimizer name: %s. Available optimizers: %s",
            optimizer_name,
            list(optimizers.keys()),
        )
        raise  # re-raise the exception

    return optimizer


def set_device(device):
    """Sets the device for the experiment"""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return device


#: cuBLAS workspace settings that make cuBLAS GEMMs bit-reproducible on CUDA >= 10.2.
#: Must be present in the environment *before* the first cuBLAS handle is created.
VALID_CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """Seed every RNG stream used by the pipeline and optionally enforce determinism.

    Must be called *before* anything that draws random numbers -- in particular
    before ``initialize_model()``, whose freshly created classifier head is
    randomly initialised from the global torch RNG.

    Args:
        seed: Seed applied to ``random``, ``numpy`` and ``torch`` (CPU and CUDA).
        deterministic: If True, call ``torch.use_deterministic_algorithms`` so that
            any op without a deterministic CUDA kernel is reported instead of
            silently introducing run-to-run noise.
        warn_only: Forwarded to ``torch.use_deterministic_algorithms``. When True a
            nondeterministic op warns instead of raising. Use only to get a run
            through; the run is then *not* reproducible.

    Note:
        ``PYTHONHASHSEED`` cannot be set from inside a running interpreter -- string
        hashing is already randomised by then. It is exported here only so that
        forked/spawned children inherit it; the launcher must export it as well.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not deterministic:
        torch.use_deterministic_algorithms(False)
        logging.warning(
            "Deterministic algorithms are DISABLED. Runs at seed %s will not be "
            "bit-reproducible.",
            seed,
        )
        return

    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if (
        torch.cuda.is_available()
        and cublas_cfg not in VALID_CUBLAS_WORKSPACE_CONFIGS
    ):
        logging.warning(
            "CUBLAS_WORKSPACE_CONFIG is %r, expected one of %s. cuBLAS GEMMs may be "
            "nondeterministic. Export it before starting python.",
            cublas_cfg,
            VALID_CUBLAS_WORKSPACE_CONFIGS,
        )

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    logging.info(
        "Deterministic algorithms enabled (warn_only=%s), seed=%s",
        warn_only,
        seed,
    )


def seed_worker(worker_id):
    """
    Sets the seed for generating random numbers in a PyTorch worker process.
    It is used to ensure reproducibility in PyTorch data loaders.

    Args:
        worker_id (int): The ID for the worker process.

    Note:
        The seed for each worker is derived from the PyTorch initial seed using the worker ID,
        ensuring different seeds for different workers.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def initialize_wandb(config, extra_tags=None):
    """Initializes wandb for tracking the experiments and updates the config file"""

    wandb_logging = config["wandb"]["logging"]
    wandb_run_name = None

    entity = os.getenv(
        "WANDB_ENTITY", None
    )  # get entity from env variable if set

    if wandb_logging:
        project = config["wandb"]["project"]
        tags = config["wandb"].get("tags", [])
        if extra_tags:
            tags = list(set(tags + extra_tags))  # merge and deduplicate

        # notes and tags are optional
        run = wandb.init(
            project=project,
            entity=entity,
            notes=config["wandb"].get("notes", None),
            tags=tags,
            config=config,
        )
        wandb_run_name = run.name
        logging.info("Logging to wandb is enabled.")
    else:
        logging.info("Logging to wandb is disabled.")

    return wandb_run_name, wandb_logging


def log_wandb(wandb_logging: bool, log_data: Dict):
    # log Data is a Dict of metrics
    if wandb_logging:
        wandb.log(log_data)


def log_metrics(log_data: Dict):
    # log dict with metrics prettily
    for k, v in log_data.items():
        logging.info(f"{k:45s}: {v:.4f}")


def rename_model(path_model, wandb_run_name=None):
    """Rename model if it already exists"""
    base, ext = os.path.splitext(path_model)

    if wandb_run_name:
        new_path_model = f"{base}_{wandb_run_name}{ext}"
    else:
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        new_path_model = f"{base}_{counter}{ext}"
    return new_path_model


def handle_model_path(config, wandb_run_name, overwrite_model=False):
    """Handle the model path based on the configuration and arguments"""
    path_model = config["training"]["path_to_save_model"]
    if os.path.exists(path_model):
        logging.info("Model file already exists at %s.", path_model)
        if not overwrite_model:
            path_model = rename_model(path_model, wandb_run_name)
            # Update config file with the new path
            config["training"]["path_to_save_model"] = path_model
            logging.info("Model will be renamed to %s.", path_model)
        else:
            logging.info("Model will be overwritten.")
    return config


def save_command(sys_args, path_results, filename="run_command.sh"):
    # save the used command as a sh script so recreating an experiment is
    # more transparent & convenient
    command_str = " ".join(sys_args)
    command_str = "python " + command_str
    filepath = os.path.join(path_results, filename)

    with open(filepath, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(command_str + "\n")

    logging.info(f"Command saved to {filepath}")


def save_pred_csv(
    df: pd.DataFrame, preds: np.ndarray, path: str = "./results/df.csv"
) -> pd.DataFrame:
    """
    Adds predicted labels to a dataframe and saves it to a CSV file.

    Args:
        df (pd.DataFrame): The input dataframe to add predicted labels to.
        preds (torch.Tensor): The predicted labels as a PyTorch tensor.
        path (str, optional): The directory to save the CSV file in. Defaults to "./results".

    Returns:
        pd.DataFrame: The input dataframe with the predicted labels added.

    """
    # preds = torch.sigmoid(preds.cpu()).argmax(dim=1).numpy()
    df = df.assign(pred=preds)
    df.to_csv(path, index=False)


def save_cm_table(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    labels,
    target_names,
    path,
    name="conf_matrix",
):
    """
    Save a labeled confusion matrix to an Excel file.

    Parameters:
    - labels (list or array): Ground truth labels (as indices).
    - preds (list or array): Predicted labels (as indices).
    - categories (list): List of class names (ordered by index).
    - path (str): Directory to save the Excel file in.
    - name (str): Base name of the output file (no extension).
    """

    # predicted probabilities
    # pred_probs = torch.softmax(pred_logits)
    # # predicted labels
    # pred_labels = pred_probs.argmax(dim=1)
    # create report: classwise precision, recall, f1, support
    # note: the IS459 paper uses zero_division=1
    report = classification_report(
        true_labels,
        pred_labels,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=np.nan,
    )
    df_report = pd.DataFrame(report).transpose()

    # compute the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    # wrap in DataFrame for labeling
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)

    # row-wise normalization
    df_cm_norm = df_cm.div(df_cm.sum(axis=1), axis=0).fillna(0)

    # most-confused classes (top 50)
    confusion_totals = df_cm.sum(axis=1) - df_cm.values.diagonal()
    top_confused = confusion_totals.sort_values(ascending=False).head(50).index
    df_cm_top = df_cm.loc[top_confused, top_confused]

    # make dfs
    dfs_to_save = {
        "Confusion matrix": df_cm,
        "Confusion matrix norm.": df_cm_norm,
        "Confusion matrix top 50": df_cm_top,
        "Classwise metrics": df_report,
    }

    # save to Excel sheets
    output_file = os.path.join(path, f"{name}.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df_out in dfs_to_save.items():
            df_out.to_excel(writer, sheet_name=sheet_name)

    logging.info(f"Confusion matrix saved to {output_file}")


def save_file_probabilities(
    outputs: EvalOutputs, mode: str, path_results: str, class_to_type_dict: Dict
) -> str:
    """Write per-file mean probabilities to ``probs_file_{mode}.csv``.

    The chunk-level dumps written by :func:`save_results` only contain hard class
    predictions, which forces downstream analyses to approximate file-level scores
    with a majority vote. Saving the ``file_prob_mean`` probability vectors makes an
    exact evaluation-noise bootstrap possible.

    Args:
        outputs: Chunk-level :class:`~utils.utils_metrics.EvalOutputs` holding logits
            (``prediction_type='logits'``, one row per chunk, ``paths`` populated).
        mode: Split name, used in the file name (``train`` / ``valid`` / ``test``).
        path_results: Directory to write into.
        class_to_type_dict: Mapping ``class index -> class name``.

    Returns:
        The path of the written CSV file.
    """
    file_outputs = aggregate_file_prob_mean(outputs)

    probs = file_outputs.to_probs().detach().cpu().numpy()  # [F, C], softmaxed
    labels_true = file_outputs.y_true.detach().cpu().numpy()  # [F]
    labels_pred = probs.argmax(axis=1)

    columns = [
        f"prob_{idx}_{class_to_type_dict[idx]}" for idx in range(probs.shape[1])
    ]
    df_probs = pd.DataFrame(probs, columns=columns)
    df_probs.insert(0, "fpath_audio", file_outputs.paths)
    df_probs.insert(1, "label_true", labels_true)
    df_probs.insert(2, "label_pred", labels_pred)

    output_file = os.path.join(path_results, f"probs_file_{mode}.csv")
    df_probs.to_csv(output_file, index=False)
    logging.info(
        "Saved per-file mean probabilities for %s files to %s",
        len(df_probs),
        output_file,
    )
    return output_file


def collect_run_provenance(num_workers: Optional[int] = None) -> Dict:
    """Collect the code/environment fingerprint needed to tell drift from nondeterminism.

    Two runs of the same config at the same seed can still differ if the code or the
    environment moved underneath them. Recording this alongside each run makes that
    distinguishable after the fact.

    Args:
        num_workers: DataLoader worker count, which changes the augmentation RNG
            streams and therefore has to match for a run to be reproducible.

    Returns:
        A flat dictionary of provenance fields, safe to push into the W&B config.
    """

    def _git(*cmd: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *cmd],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, OSError) as exc:
            logging.warning(
                "Could not read git provenance (%s %s): %s", *cmd, exc
            )
            return None

    provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # --untracked-files=no: only *tracked* modifications mean the commit no longer
        # describes the code that ran. Untracked scratch files (papers, analyses) would
        # otherwise pin this to True forever and make the flag useless.
        "git_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no")
        ),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "num_workers": num_workers,
    }

    if provenance["git_dirty"]:
        logging.warning(
            "Working tree is dirty: the recorded git commit does not fully describe "
            "the code this run executed."
        )

    logging.info("Run provenance:\n%s", pformat(provenance))
    return provenance


def save_results(
    df: pd.DataFrame,
    outputs: EvalOutputs,
    mode: str,
    path_results: str | Path,
    class_to_type_dict: Mapping[int, str],
    aggregations: Sequence[str],
) -> None:
    """
    Save predictions and confusion matrices for multiple aggregation methods.

    For each requested aggregation, this function:

    1. Aggregates the chunk-level evaluation outputs.
    2. Converts the aggregated predictions to hard class labels.
    3. Saves a CSV containing the predictions.
    4. Saves a plotted confusion matrix.
    5. Additionally saves the confusion matrix as an Excel table when the
       number of classes exceeds 66.

    Parameters
    ----------
    df:
        Source dataframe passed to ``save_pred_csv``. It must be compatible
        with the samples produced by the selected aggregation.
    outputs:
        Chunk-level evaluation outputs containing predictions, labels, and
        file paths.
    mode:
        Evaluation mode used in output filenames, such as ``"train"``,
        ``"val"``, or ``"test"``.
    path_results:
        Directory in which result files are stored.
    class_to_type_dict:
        Mapping from integer class indices to class names.
    aggregations:
        Names of aggregation functions registered in
        ``AGGREGATION_REGISTRY``.

    Raises
    ------
    KeyError
        If an aggregation name is not registered.
    ValueError
        If the aggregated output cannot be aligned with the dataframe used
        by ``save_pred_csv``.
    """
    path_results = Path(path_results)
    path_results.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(class_to_type_dict.items())
    labels = [class_idx for class_idx, _ in sorted_items]
    target_names = [class_name for _, class_name in sorted_items]

    for aggregation_name in aggregations:
        try:
            aggregation_fn = AGGREGATION_REGISTRY[aggregation_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown aggregation {aggregation_name!r}. Available "
                f"aggregations: {sorted(AGGREGATION_REGISTRY)}"
            ) from exc

        aggregated_outputs = aggregation_fn(outputs)

        labels_true, labels_pred = prepare_labels(
            aggregated_outputs.y_true,
            aggregated_outputs.y_pred,
        )

        result_name = f"{mode}_{aggregation_name}"

        if aggregated_outputs.paths is None:
            raise ValueError(
                f"Aggregation {aggregation_name!r} did not return paths."
            )

        # note: this only preserves path, true label, and prediction for file-level results.
        # Other file-level metadata from df is not retained.
        if len(df) == len(labels_pred):
            df_predictions = df.copy()
        else:
            df_predictions = pd.DataFrame(
                {
                    "path": aggregated_outputs.paths,
                    "label": labels_true.numpy(),
                }
            )

        save_pred_csv(
            df_predictions,
            labels_pred,
            path=path_results / f"df_{result_name}.csv",
        )

        xyticks = True

        if len(labels) > 66:
            logging.info(
                "The confusion matrix for %s contains %d classes. "
                "Saving an additional Excel table and disabling plot labels.",
                aggregation_name,
                len(labels),
            )

            save_cm_table(
                labels_true,
                labels_pred,
                labels,
                target_names,
                path=path_results,
                name=f"conf_matrix_{result_name}",
            )

            xyticks = False

        create_cm(
            labels_true,
            labels_pred,
            categories=target_names,
            path=path_results,
            name=f"conf_matrix_{result_name}",
            show_fig=False,
            xyticks=xyticks,
        )


def save_config(config, path_results, config_name="config.yml"):
    # Store config file
    with open(os.path.join(path_results, config_name), "w") as f:
        yaml.dump(config, f)


def load_model_cp(path_model, device):
    # load a model checkpoint while ignoring warnings
    logger = logging.getLogger()
    with warnings.catch_warnings():
        if logger.getEffectiveLevel() <= logging.INFO:
            warnings.simplefilter("ignore", category=FutureWarning)
            best_model_cp = torch.load(
                path_model, map_location=torch.device(device)
            )
    return best_model_cp


def assign_codes_ordered_by_frequency(
    df, label_column, code_column, verbose=False
):
    """
    Assign codes to the label_column in the order of most samples per class. The codes are assigned such that
    the class with the most samples gets the code 0, the class with the second most samples gets the code 1, and so on.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_column (str): The column name of the labels.
        code_column (str): The column name to store the codes.
        verbose (bool): If True, log the DataFrame with the class number, the class of the label, and the number of samples per class.

    Returns:
        df (pd.DataFrame): The DataFrame with the new codes column.
        class_to_type_dict (dict): A dictionary where the keys are the codes and the values are the corresponding labels.
    """
    class_counts = df[label_column].value_counts()
    class_ordered = class_counts.index
    class_to_type_dict = {i: type for i, type in enumerate(class_ordered)}
    df[code_column] = pd.Categorical(
        df[label_column], categories=class_ordered, ordered=True
    ).codes

    if verbose:
        logging_df = class_counts.reset_index()
        logging_df.columns = [label_column, "Number of samples"]
        logging_df.insert(0, "Class Number", range(1, 1 + len(logging_df)))
        logging.info("\n" + logging_df.to_string(index=False))

    return df, class_to_type_dict


def calculate_class_weights(
    df, var_name, method="inverse_frequency", beta=0.999
):
    """
    Calculate class weights to handle class imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the samples (typically: training set).
    var_name : str
        Column name containing the integer class codes (e.g. 0..C-1).
    method : str
        Weighting method. One of:
            - "none" / None : return None (no class weighting)
            - "inverse_frequency"       : w_c ∝ 1 / n_c
            - "inverse_sqrt_frequency"  : w_c ∝ 1 / sqrt(n_c)
            - "effective_num_samples"   : (1 - beta) / (1 - beta**n_c)
            - "InsectEffNet"           : w_c ∝ 1 - f_c  (f_c = n_c / N)
    beta : float
        Hyperparameter used for "effective_num_samples".

    Returns
    -------
    torch.Tensor or None
        Tensor of shape [num_classes] with class weights, or None if method is "none".
    """
    if method in [None, "none"]:
        return None

    # FIX: .sort_index() ensures weights align with Class 0, 1, 2... in case the classes are not sorted by size
    class_counts = df[var_name].value_counts().sort_index()
    counts = torch.tensor(class_counts.values, dtype=torch.float)
    num_classes = counts.numel()
    N = float(counts.sum())
    # num_classes = len(counts)

    if method == "inverse_frequency":
        weights = 1.0 / counts
    elif method == "inverse_sqrt_frequency":
        weights = 1.0 / torch.sqrt(counts)
    elif method == "effective_num_samples":
        # CVPR 2019 method for large imbalance
        effective_num = 1.0 - torch.pow(
            torch.tensor(beta, dtype=counts.dtype), counts
        )
        weights = (1.0 - beta) / effective_num
    elif method == "InsectEffNet":
        freq = counts / N
        return 1.0 - freq
    else:
        raise ValueError(f"Unknown class weight method: {method}")
        # return torch.ones(num_classes)

    # NORMALIZE: Keep sum equal to num_classes to stabilize Learning Rate
    weights = weights / weights.sum() * num_classes
    return weights


def log_tensor_stats(name: str, t: torch.Tensor) -> None:
    """
    Log basic statistics (mean, std, min, max, range, max/min ratio) of a tensor.

    Args:
        name (str): Descriptive name for the tensor (e.g. "Sampler weights").
        t (torch.Tensor): 1D or ND tensor whose stats will be logged.
    """
    if t.numel() == 0:
        logging.info(f"{name}: tensor is empty, no statistics to log.")
        return

    t = t.to(torch.float64)

    mean = t.mean()
    std = t.std()
    t_min = t.min()
    t_max = t.max()
    value_range = t_max - t_min

    # Avoid division by zero for max/min ratio
    if t_min != 0:
        max_min_ratio = float(t_max / t_min)
    else:
        max_min_ratio = float("inf")

    logging.info(
        f"{name} statistics:\n"
        f"  mean:          {mean:.6f}\n"
        f"  std:           {std:.6f}\n"
        f"  min:           {t_min:.6f}\n"
        f"  max:           {t_max:.6f}\n"
        f"  range:         {value_range:.6f}\n"
        f"  max/min ratio: {max_min_ratio:.2f}x"
    )


def log_class_weights_to_wandb(weights_for_logging, class_to_type_dict):
    """
    Log class weights to WandB with table and bar chart visualizations.

    Args:
        weights_for_logging (dict): Dictionary containing:
            - "type" (str): Either "sampler" or "loss"
            - "method" (str): The weighting method used
            - "weights" (list): List of class weights
        class_to_type_dict (dict): Mapping from class index to class name
    """
    import wandb
    import plotly.graph_objects as go

    if weights_for_logging is None:
        return

    weight_type = weights_for_logging["type"]
    weight_method = weights_for_logging["method"]
    weights_list = weights_for_logging["weights"]

    # Log method to wandb summary
    wandb.run.summary.update(
        {
            f"{weight_type}_weighting_method": weight_method,
        }
    )

    # Log weights as a list (for basic visualization)
    wandb.log({f"{weight_type}_class_weights": weights_list})

    # Create sorted class names list
    sorted_class_names = [
        class_to_type_dict[i] for i in range(len(weights_list))
    ]

    # Create table with class names and weights
    table_data = [
        [idx, class_name, weight]
        for idx, (class_name, weight) in enumerate(
            zip(sorted_class_names, weights_list)
        )
    ]

    weights_table = wandb.Table(
        data=table_data, columns=["class_idx", "class_name", "weight"]
    )

    wandb.log({f"{weight_type}_weights_table": weights_table})

    # Create custom Plotly bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=sorted_class_names,
                y=weights_list,
                marker=dict(
                    color=weights_list,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Weight"),
                ),
                text=[f"{w:.4f}" for w in weights_list],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Weight: %{y:.6f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"{weight_type.capitalize()} Class Weights ({weight_method})",
        xaxis_title="Class Name",
        yaxis_title="Weight",
        xaxis_tickangle=-45,
        height=max(
            600, len(weights_list) * 20
        ),  # Scale height with number of classes
        showlegend=False,
        template="plotly_white",
    )

    wandb.log({f"{weight_type}_weights_barchart": wandb.Plotly(fig)})

    logging.info(
        f"Logged {weight_type} class weights (method: {weight_method}) "
        f"to wandb as list, table, and bar chart"
    )


def log_class_weight_recommendations(
    weighted_sampler_method: Optional[str],
    weighted_loss_method: Optional[str],
    # optimizer_name: str,
    training_config: dict,
) -> None:
    """
    Log recommendations and sanity checks when using class weighting
    (either via sampler or loss).

    Parameters
    ----------
    weighted_sampler_method : str or None
        Method used for WeightedRandomSampler (e.g. "inverse_frequency"), or None.
    weighted_loss_method : str or None
        Method used for loss-based class weights (e.g. "effective_num_samples"), or None.
    # optimizer_name : str
    #     Name of the optimizer in use (e.g. "adam", "adamw").
    training_config : dict
        The 'training' section of the config, used to check things like gradient clipping.
    """
    # Only log recommendations if some form of class weighting is active
    if weighted_sampler_method is None and weighted_loss_method is None:
        return

    # Optimizer recommendation
    optimizer_name = training_config.get("optimizer", None)
    if optimizer_name != "adamw":
        logging.warning(
            "Using optimizer '%s' with class weights. "
            "AdamW is recommended for better stability.",
            optimizer_name,
        )

    # Gradient clipping recommendation
    grad_clip = training_config.get("gradient_clip_norm", None)
    if grad_clip is None:
        logging.warning(
            "Using class weights without gradient clipping. "
            "Consider adding 'gradient_clip_norm: 1.0' to the training config "
            "for more stable training."
        )
    else:
        logging.info(
            "✓ Using gradient clipping (max_norm=%.2f) with class weights",
            grad_clip,
        )


def process_species_dictionaries(
    yaml_file, num_classes, path_results, dump=False
):
    """
    Process species dictionaries by filtering based on the number of classes
    and save the filtered dictionaries to a YAML file.

    Parameters:
    - yaml_file (dict): dictionary for full
    - num_classes (int): Maximum number of classes to keep.
    - path_results (str): Path to the directory where results are stored.
    """
    # Read species dictionaries from YAML file
    with open(yaml_file, "r") as f:
        species_dictionaries = yaml.safe_load(f)

    # Filter species dictionaries based on number of classes
    species_dictionaries = {
        k: v for k, v in species_dictionaries.items() if int(k) < num_classes
    }
    categories = list(species_dictionaries.values())

    # Write filtered species dictionaries to a YAML file in path_results
    if dump == True:
        output_yaml_file = os.path.join(
            path_results, "species_dictionaries.yml"
        )
        with open(output_yaml_file, "w") as f:
            yaml.dump(species_dictionaries, f)

    return categories


def apply_user_filters(df, filters, strict=True):
    """
    Accepts a dict containing filters to apply to the annotation dataframe,
    applies them to the dataframe and returns a the filtered df.
    """
    original_keys = set(filters)
    if ("dataset_mappings" in original_keys) and strict:
        logging.warning(
            "'dataset_mappings' not supported here - will be omitted."
        )

    # set up filter handler registry
    FILTER_HANDLERS = {}

    def filter_handler(name, required=True):
        def decorator(func):
            FILTER_HANDLERS[name] = {
                "func": func,
                "required": required,
            }
            return func

        return decorator

    # filtering logic, the decorator is called upon function definition and
    # registers the implemented filters
    # filtering logic is executed sequentially, in the same order the functions
    # are defined here
    @filter_handler("min_sample_rate")
    def apply_min_sample_rate(df, value):
        # remove files with a sample rate below min_sample_rate
        return df[df["sample_rate"] >= value]

    @filter_handler("min_audio_length")
    # remove files below a certain audio duration
    def apply_min_audio_length(df, value):
        return df[df["audio_length"] >= value]

    @filter_handler("use_class_list")
    # remove files below a certain audio duration
    def apply_use_class_list(df, dict_of_values):

        list_path = dict_of_values["list_path"]
        ext = os.path.splitext(list_path)[-1].lower()
        if ext == ".yml":
            class_to_type_dict = load_config(list_path)
            species_list = list(class_to_type_dict.values())
        elif ext == ".csv":
            species_list = pd.read_csv(list_path)
            species_list = species_list["class_name"].to_list()
        else:
            raise ValueError(
                "Cannot handle file extension: {ext}, only .yml or .csv"
            )
        species_list += dict_of_values["classes_to_keep"]

        df_to_keep = df[df["class_name"].isin(species_list)]

        if dict_of_values["sample_other_class"] == True:
            # extract values from config
            n_samples_per_class = dict_of_values["n_samples_per_class"]
            top_n_classes_to_sample = dict_of_values["top_n_classes_to_sample"]
            # sample data into "other_class"
            df_to_drop = df[~df["class_name"].isin(species_list)]
            top_classes = (
                df_to_drop["class_name"]
                .value_counts()
                .nlargest(top_n_classes_to_sample)
                .index.tolist()
            )
            df_top_other = df_to_drop[
                df_to_drop["class_name"].isin(top_classes)
            ]
            df_to_add = df_top_other.groupby("class_name").apply(
                lambda g: g.sample(
                    min(len(g), n_samples_per_class), random_state=42
                )
            )
            df_to_add = df_to_add.reset_index(drop=True)
            new_class_name = dict_of_values["new_class_name"]
            df_to_add["class_name"] = new_class_name
            # just to make sure
            df_to_add["class"] = pd.NA
            # add samples "other_insect" into the df to keep
            df_to_keep = pd.concat([df_to_keep, df_to_add], ignore_index=True)

            logging.info(
                f"Sampled from {top_n_classes_to_sample} top classes up to {n_samples_per_class} files (total {len(df_to_add)} files sampled into class '{new_class_name}')"
            )

        return df_to_keep

    @filter_handler("reassign_class_below")  # required=False
    def apply_class_reassignment(df, dict_of_values):
        # reassign classes below a certain number of files / examples per class
        # to a new "other" class
        new_class_name = dict_of_values["new_class_name"]
        threshold = dict_of_values["reassign_below_n_examples"]

        # count the number of examples per class
        class_counts = df["class_name"].value_counts()

        # Identify classes below the threshold
        classes_to_reassign = class_counts[class_counts < threshold].index

        # Create a copy and reassign
        df_reassigned = df.copy()
        df_reassigned["class_name"] = df_reassigned["class_name"].apply(
            lambda x: new_class_name if x in classes_to_reassign else x
        )
        return df_reassigned

    @filter_handler("min_duration_per_class")
    # remove files/classes where the class has less than min_duration_per_class (in minutes) of total audio
    def apply_min_duration_per_class(df, value):
        duration_per_class = (
            df.groupby("class_name")["audio_length"]
            .sum()
            .sort_values(ascending=False)
        )
        duration_per_class /= 60  # seconds to minutes
        keep = duration_per_class[duration_per_class >= value].index
        return df[df["class_name"].isin(keep)]

    @filter_handler("min_files_per_class")  # required=False
    def apply_min_files_per_class(df, value):
        # remove files/classes that have less than min_files_per_class files
        vc = df["class_name"].value_counts()
        keep = vc[vc >= value].index
        return df[df["class_name"].isin(keep)]

    @filter_handler("min_files_per_subset")  # required=False
    def filter_per_subset(df, dict_of_values):
        keep_classes = set(df["class_name"])  # start with all classes

        for subset, min_count in dict_of_values.items():
            subset_counts = (
                df[df["subset"] == subset].groupby("class_name").size()
            )
            # logging.info(f"{subset_counts}")
            valid_classes = set(subset_counts[subset_counts >= min_count].index)
            keep_classes &= valid_classes  # intersection

        return df[df["class_name"].isin(keep_classes)]

    # apply known filters sequentially
    for name, meta in FILTER_HANDLERS.items():
        if name in filters:
            len_init = len(df)
            classes_init = set(df["class_name"])
            val = filters.pop(name)
            # function call
            df = meta["func"](df, val)
            n_rows_dropped = len_init - len(df)
            classes_dropped = classes_init - set(df["class_name"])
            logging.info(f"Filter {name} dropped {n_rows_dropped} rows.")
            logging.info(f"...and {len(classes_dropped)} classes")
            if classes_dropped:
                logging.info(classes_dropped)

    # warn about extra filters in filtering yaml
    if strict:
        if filters:
            logging.warning(
                "These filters were in the config but not recognized:"
            )
            for extra in filters:
                logging.warning(f"  - {extra}")

        # warn about implemented filters that weren't specified
        missing = [
            name
            for name, meta in FILTER_HANDLERS.items()
            if meta.get("required") and name not in original_keys
        ]

        if missing:
            logging.info(
                "These filters are available but were not provided in the config:"
            )
            for name in missing:
                logging.info(f"  - {name}")
    return df


def apply_user_mappings(df_aggregated, species_name_mapping_path):
    """Load annotations as well as class name mappings, apply said mappings to
    the annotations classes and return the clean dataframe."""

    # load user-specified class name mappings
    file_ext = os.path.splitext(species_name_mapping_path)[1]
    if file_ext == ".xlsx":
        df_species_name_mapping = pd.read_excel(species_name_mapping_path)
    else:
        df_species_name_mapping = pd.read_csv(species_name_mapping_path)

    num_total_mappings_labels = len(df_species_name_mapping)
    logging.info(
        f"Loaded {species_name_mapping_path} with {num_total_mappings_labels} rows (mappings)."
    )

    # drop species names / labels where "user_mapping" is empty
    species_to_drop = df_species_name_mapping.loc[
        df_species_name_mapping["user_mapping"].isna(), "class_name"
    ]
    species_to_drop = set(species_to_drop)
    df_clean = df_aggregated.loc[
        ~df_aggregated["class_name"].isin(species_to_drop)
    ]
    num_dropped_examples = len(df_aggregated) - len(df_clean)
    num_dropped_labels = len(species_to_drop)

    # logging
    if species_to_drop:
        logging.info(
            f"Dropped {num_dropped_labels} species labels (dropped {num_dropped_examples} rows):"
        )
        logging.info(species_to_drop)

    # apply mapping/rename species names according to user mappings
    # drop NAs and only map differing names...
    user_mapping = df_species_name_mapping[
        ["class_name", "user_mapping"]
    ].dropna()

    # count unchanged class names
    num_unchanged_labels = len(
        user_mapping[user_mapping["user_mapping"] == user_mapping["class_name"]]
    )
    logging.info(f"{num_unchanged_labels} species labels remained unchanged.")

    # get mappings where the labels differ
    user_mapping = user_mapping[
        user_mapping["user_mapping"] != user_mapping["class_name"]
    ]

    # create a dict for renaming
    user_mapping_dict = dict(
        zip(user_mapping["class_name"], user_mapping["user_mapping"])
    )

    # apply the mapping and count number of changed values
    mapped = df_clean["class_name"].map(user_mapping_dict)
    num_mapped_examples = mapped.notna().sum()
    num_mapped_labels = len(user_mapping_dict)
    logging.info(
        f"{num_mapped_examples} values were mapped ({num_mapped_labels} labels)"
    )

    # combine "mapped" Series with original column
    df_clean.loc[:, "class_name"] = mapped.fillna(df_clean["class_name"])

    # final sanity checks
    assert (
        num_dropped_labels + num_unchanged_labels + num_mapped_labels
        == num_total_mappings_labels
    ), "number of initially loaded and applied mappings differ"
    assert (
        not df_clean["class_name"].isin(species_to_drop).any()
    ), "labels that should have been dropped remain!"
    assert (
        not df_clean["class_name"].isin(user_mapping["class_name"]).any()
    ), "labels that should have been mapped remain!"

    logging.info(
        f"Cleaned annotations contain {len(df_clean)} rows ({df_clean['class_name'].nunique()} classes)."
    )

    return df_clean


def apply_user_dataset_config(df, dataset_config):

    # restrict to datasets listed under "use" in the filter_config
    df_out = None
    if "use" in dataset_config.keys():
        use_datasets = dataset_config["use"]
        df_out = df[df["dataset_name"].isin(use_datasets)].copy()
        logging.info("Using dataset(s):")
        logging.info(use_datasets)

        # reporting
        datasets_init = set(df["dataset_name"])
        classes_init = set(df["class_name"])
        n_rows_dropped = len(df) - len(df_out)
        classes_dropped = classes_init - set(df_out["class_name"])
        datasets_dropped = datasets_init - set(df_out["dataset_name"])
        logging.info(f"Dropped {len(datasets_dropped)} datasets")
        if datasets_dropped:
            logging.info(datasets_dropped)
        logging.info(f"Dropped {n_rows_dropped} rows.")
        logging.info(f"...and {len(classes_dropped)} classes")
        if classes_dropped:
            logging.info(classes_dropped)

    else:
        raise UserWarning("Specify which datasets to use in the filter_config.")

    # apply remapping for dataset and subset
    if "mappings" in dataset_config.keys():
        for dataset_name, mapping in dataset_config["mappings"].items():
            # select the source dataset
            mask = df_out["dataset_name"] == dataset_name
            # replace subset according to the mapping (e.g. train: test)
            # specified in the config
            df_out.loc[mask, "subset"] = df_out.loc[mask, "subset"].replace(
                mapping
            )

    return df_out


def create_datasets(df):
    """
    create dataset from df and user configs
    """

    # extract the train-valid-test split
    df["subset"] = df["subset"].str.lower()
    df_train = df[df["subset"] == "train"].reset_index(drop=True)
    df_valid = df[df["subset"] == "validation"].reset_index(drop=True)
    df_test = df[df["subset"] == "test"].reset_index(drop=True)

    return df_train, df_valid, df_test


def capture_summary(model, input_size):
    # get the complete torchinfo summary without it spamming to the console
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        summary(model, input_size=input_size, verbose=1)
    return buffer.getvalue()


def save_model_summary(summary_str, path_results, model_name):
    fpath_summary = os.path.join(path_results, f"{model_name}_summary.txt")
    with open(fpath_summary, "w") as f:
        f.write(summary_str)


def log_dataset_subsets(train_dataset, valid_dataset, test_dataset):
    """Log proportions of subsets"""
    n_samples_total = (
        len(train_dataset) + len(valid_dataset) + len(test_dataset)
    )

    # calculate percentages
    perc_samples_train = (len(train_dataset) / n_samples_total) * 100
    perc_samples_valid = (len(valid_dataset) / n_samples_total) * 100
    perc_samples_test = (len(test_dataset) / n_samples_total) * 100

    logging.info(
        f"Total number of samples (Dataset): {n_samples_total}, train: {perc_samples_train:.0f}%, validation: {perc_samples_valid:.0f}%, test: {perc_samples_test:.0f}%",
    )


def check_dataset(df, strict=True, custom_msg="", silent=True):
    num_classes = df["class"].nunique()

    # check if the same species are in all subsets
    if not silent:
        logging.info(
            f"{custom_msg}There are %s unique classes in the dataset.",
            num_classes,
        )

    if not df["class"].max() + 1 == num_classes:
        if strict:
            raise ValueError(
                "Invalid class labels in annotations (df['class'].max() + 1 != num_classes)"
            )
        else:
            logging.warning(
                "Invalid class labels in annotations (df['class'].max() + 1 != num_classes)"
            )

    # check if the subset column contains NAs and warn if true
    if df["subset"].hasnans:
        logging.debug("Annotation CSV: column 'subset' contains NAs")

    species_sets = {
        subset: set(df[df["subset"] == subset]["class_name"].unique())
        for subset in ["train", "validation", "test"]
    }

    # Check if all sets are equal
    all_equal = (
        species_sets["train"]
        == species_sets["validation"]
        == species_sets["test"]
    )
    if not all_equal:
        if strict:
            raise ValueError(
                "There are differences in species sets between train, validation and test datasets."
            )
        logging.warning(
            "There are differences in species sets between train, validation and test datasets."
        )
    else:
        logging.debug("Same species are in train, validation, test subsets.")


def handle_result_path(path_model, args):
    name = os.path.splitext(os.path.basename(path_model))[0]

    # name the result path based on the dataset config path
    if args.evaluate_test_only and args.dataset_config_path is not None:
        filter_name = os.path.splitext(
            os.path.basename(args.dataset_config_path)
        )[0]
        name += "_" + filter_name

    elif args.evaluate_test_only:
        name += "_evaluate_test"
    result_path = f"./results/{name}"

    if os.path.isdir(result_path):
        # append the first free counter, never reuse an existing folder
        counter = 1
        while os.path.isdir(f"{result_path}_{counter}"):
            counter += 1
        result_path_new = f"{result_path}_{counter}"
        logging.info(
            f"Result path '{result_path}' exists, renaming to: '{result_path_new}'"
        )
        result_path = result_path_new
        name = os.path.basename(result_path)

    os.makedirs(result_path)
    return name, result_path
