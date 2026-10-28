import argparse
import os
import sys
import signal
from pprint import pformat

# Deterministic cuBLAS GEMMs. cuBLAS reads this when it creates its handle, so it has
# to be in the environment before torch initialises CUDA. Setting it here covers runs
# whose launcher forgot to export it; an explicit export still wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# logging
import logging
import coloredlogs
import wandb

# dataframes and dictionaries
import pandas as pd

# torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
import torch

# Torch model info
from torchinfo import summary

# custom modules
from utils.utils_config import load_config, update_config
from utils.utils_functions import (
    log_class_weight_recommendations,
    log_tensor_stats,
    log_class_weights_to_wandb,
    timeit,
    set_optimizer,
    set_device,
    set_seed,
    seed_worker,
    initialize_wandb,
    handle_model_path,
    collect_run_provenance,
    save_results,
    save_file_probabilities,
    save_command,
    save_config,
    calculate_class_weights,
    create_datasets,
    save_model_summary,
    capture_summary,
    apply_user_filters,
    apply_user_dataset_config,
    check_dataset,
    assign_codes_ordered_by_frequency,
    log_dataset_subsets,
    load_model_cp,
    assign_codes_based_on_previous_mapping,
    handle_result_path,
    log_metrics,
    apply_balanced_subset_sampling,
)
from utils.utils_metadata import store_dataset_metadata
from utils.utils_metrics import compute_metrics
import custom_datasets
from custom_datasets import (
    create_transforms,
    make_collate_fn,
    verify_compose,
)
from utils.losses import FocalLoss
from models import initialize_model
from train_eval import train, evaluate_model
from dotenv import load_dotenv

# =============================================
# Load environment variables from a .env file
# =============================================

load_dotenv()

# =============================================
# Functions
# =============================================


def handle_sigterm(signum, frame):
    wandb.finish(exit_code=0)
    sys.exit(0)


# =============================================
# Parse CMD arguments
# =============================================
def float_or_none(x):
    if x.lower() == "none":
        return None
    return float(x)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path of the yml-configuration file for the pipeline (see configs/models/).",
    )

    parser.add_argument(
        "-t",
        "--trsf_config",
        type=str,
        required=True,
        help="Path of the yml-configuration file for the dataset transformations (see configs/transforms/).",
    )

    parser.add_argument(
        "-dc",
        "--dataset-config-path",
        type=str,
        default=None,
        help="""Path to a yaml config file specifying how the datasets are filtered and used.""",
    )

    parser.add_argument(
        "-d",
        "--device",
        choices=[None, "cuda", "mps", "cpu"],
        default=None,
        help="Specify the torch device to be used. If not specified, it will use 'cuda' or 'mps' if available and else 'cpu'.",
    )

    parser.add_argument(
        "-wo",
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the dataloader.",
    )

    parser.add_argument(
        "-ms",
        "--model_source",
        choices=["pann", "timm", None],
        default=None,
        help="Choose the source of the model. 'pann' for models from PANNs and 'timm' for models from PyTorch Image Models (timm). If not specified, the model specified in the config is used.",
    )

    parser.add_argument(
        "-pt",
        "--pretrain",
        choices=["none", "custom", "default"],
        default="default",
        help="Specify the pretrained weights to use. 'none' for no pretraining, 'custom' to load custom pre-trained weights specified in config file, 'default' to use the timm and PANN pre-trained weights, respectively. Defaults to 'default'.",
    )

    parser.add_argument(
        "-fb",
        "--freeze_base",
        action="store_true",
        help="Fine-tune only the classification head of the model.",
    )

    parser.add_argument(
        "-ep",
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train the model. If not specified, it will use the number of epochs specified in the config file.",
    )

    parser.add_argument(
        "-em",
        "--eval_model_path",
        type=str,
        default=None,
        help="Path to a trained model checkpoint (.pth) to evaluate without training. Implies --evaluate_only. Overrides ['model']['eval_model_path'] in the config.",
    )

    parser.add_argument(
        "-eo",
        "--evaluate_only",
        action="store_true",
        help="Evaluate the model only without training (on train, validation and test set). Provide the model with --eval_model_path or in the config under ['model']['eval_model_path'].",
    )

    parser.add_argument(
        "-eto",
        "--evaluate_test_only",
        action="store_true",
        help="Evaluate the model only on the TEST set without training. This will also set evaluate_only to True. Provide the model with --eval_model_path or in the config under ['model']['eval_model_path'].",
    )

    parser.add_argument(
        "-sr",
        "--store_results",
        action="store_false",
        default=True,
        help="Do not store result files for train, val and test set of each fold.",
    )

    parser.add_argument(
        "-sm",
        "--store_models",
        action="store_false",
        default=True,
        help="Do not store the pytorch models.",
    )

    parser.add_argument(
        "-om",
        "--overwrite_model",
        action="store_true",
        default=False,
        help="Overwrite the existing PyTorch model if it is found at the same path. By default, the model will not be overwritten.",
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        default=None,
        help="Provide a path to store the model. This will override any other loaded config.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed. Overrides global.seed in the config file.",
    )

    parser.add_argument(
        "--deterministic_warn_only",
        action="store_true",
        help="Warn instead of raising when an op has no deterministic implementation. "
        "Lets a run finish, but the run is then NOT reproducible.",
    )

    parser.add_argument(
        "--save_file_probs",
        action="store_true",
        default=False,
        help="Additionally write per-file mean-probability CSVs "
        "(probs_file_{test,valid}.csv) alongside the standard results. "
        "Needed for evaluation-noise / determinism analyses.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print additional information (logging level will be set to DEBUG). Default false.",
    )

    parser.add_argument(
        "-wb",
        "--wandb",
        action="store_true",
        default=False,
        help="Log results to Weights & Biases (requires 'wandb login'). Off by default.",
    )

    parser.add_argument(
        "-sw",
        "--sweep",
        action="store_true",
        default=False,
        help="""Pass this flag to join a running wandb sweep. 
        Make sure to start the wandb sweep using the command 'wandb sweep your_sweep_config.yml' beforehand,
        which will give you a sweep ID.
        Also, this script should be called by a wandb sweep agent using the command 'wandb agent <sweep_id>'""",
    )

    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        default=None,
        choices=["sgd", "adam", "adamw", "nadam", "rmsprop", "adagrad"],
        help="Optimizer for training. Overrides config file if specified.",
    )

    parser.add_argument(
        "-lte",
        "--log_test_eval",
        action="store_true",
        default=False,
        help="Log test dataset results during training (will evaluate on test set every epoch)",
    )

    parser.add_argument(
        "-ltr",
        "--log_train_eval",
        action="store_true",
        default=False,
        help="Log train dataset results during training (will evaluate on training set every epoch)",
    )

    parser.add_argument(
        "-wt",
        "--wandb_tags",
        type=str,
        nargs="*",
        default=None,
        help="Additional wandb tags to add (will be merged with tags from config file).",
    )

    parser.add_argument(
        "-po",
        "--p_override",
        type=float,
        default=None,
        help="Override probability for all augmentations where p != 1.0. Value must be between 0 and 1.",
    )

    parser.add_argument(
        "-was",
        "--wave_augmentations_scale",
        type=float,
        default=None,
        help="Scale all augmentations done on the wave by this number. Value must be between 0 and 1.",
    )

    parser.add_argument(
        "-ws",
        "--weighted_sampler",
        type=str,
        default=None,
        choices=[
            None,
            "none",
            "inverse_frequency",
            "inverse_sqrt_frequency",
            "effective_num_samples",
            "InsectEffNet",
        ],
        help="Use WeightedRandomSampler for the training set. Overrides config file if specified.",
    )

    parser.add_argument(
        "-wl",
        "--weighted_loss",
        type=str,
        default=None,
        choices=[
            None,
            "none",
            "inverse_frequency",
            "inverse_sqrt_frequency",
            "effective_num_samples",
            "InsectEffNet",
        ],
        help="Method for calculating class weights for loss. Overrides config file if specified.",
    )

    parser.add_argument(
        "-beta",
        "--effective_num_beta",
        type=float,
        default=None,
        help="Beta parameter for effective number of samples class weighting. Overrides config file if specified.",
    )

    parser.add_argument(
        "-srt",
        "--subset_ratio",
        type=float,
        default=None,
        help="Run the experiment with a random subset of the data for quick prototyping. Value must be between 0 and 1. If None, the full dataset is used.",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for the optimizer. Overrides config file if specified.",
    )

    parser.add_argument(
        "-gc",
        "--gradient_clip",
        type=float_or_none,
        default=None,
        help="Maximum norm for gradient clipping. Overrides config file if specified.",
    )

    parser.add_argument(
        "-fg",
        "--focal_gamma",
        type=float,
        default=None,
        help="Gamma parameter for focal loss. Overrides config file if specified.",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training and evaluation. Overrides config file if specified.",
    )

    parser.add_argument(
        "-tmax",
        "--lr_scheduler_t_max",
        type=int,
        default=None,
        help="T_max parameter for CosineAnnealingLR scheduler. Overrides config file if specified.",
    )

    parser.add_argument(
        "-dr",
        "--drop_rate",
        type=float,
        default=None,
        help="Dropout rate before classifier head. Overrides config file if specified.",
    )

    parser.add_argument(
        "-dpr",
        "--drop_path_rate",
        type=float,
        default=None,
        help="Drop path (stochastic depth) rate. Overrides config file if specified.",
    )

    parser.add_argument(
        "-ma",
        "--mixup_alpha",
        type=float_or_none,
        default=None,
        help="Mixup alpha parameter. Overrides config file if specified.",
    )

    return parser.parse_args()


# =============================================
# MAIN
# =============================================


@timeit(name="Pipeline")
def main(args: argparse.Namespace, sweep_params=None, wandb_run_name=None):
    """Main script code"""

    # evaluate_test_only and eval_model_path set evaluate_only to true
    if args.evaluate_test_only or args.eval_model_path is not None:
        args.evaluate_only = True

    logging.info(
        "------------------------- Load configs and annotations -------------------------"
    )
    # load configuration files for pipeline, transformations, data
    config = load_config(args.config)

    logging.info("Loaded config file %s.", args.config)
    config_trsf = load_config(args.trsf_config)
    logging.info("Loaded transform config file %s.", args.trsf_config)

    # load dataset filter config and dataset
    filter_config = None
    if args.dataset_config_path is not None:
        filter_config = load_config(args.dataset_config_path)
        logging.info("Loaded dataset config file %s.", args.dataset_config_path)

    # set device
    device = set_device(args.device)

    # update config with parser arguments (incl. 'epochs'), make sure metrics config is there
    config = update_config(config, args, sweep_params, device)

    # Seed before anything draws random numbers. In particular this has to precede
    # initialize_model(): timm gives the model a freshly initialised classifier head,
    # and seeding afterwards leaves that head drawn from an unseeded RNG.
    seed = config["global"]["seed"]
    set_seed(
        seed,
        deterministic=config["global"].get("deterministic", True),
        warn_only=config["global"].get("deterministic_warn_only", False),
    )
    logging.info(
        "Seed set to %s (before dataset, transform and model creation)", seed
    )

    # Initialize wandb
    if sweep_params is not None:
        # in case of sweep, wandb.init() is called before main()
        # run name and sweep params are passed into main
        wandb_logging = True
        logging.info(f"Running sweep '{wandb_run_name}'")
    else:
        wandb_run_name, wandb_logging = initialize_wandb(
            {**config, **config_trsf}, extra_tags=args.wandb_tags
        )

    # if wandb_logging:
    #     wandb.config.update({
    #         "weighted_sampler_method": args.weighted_sampler,
    #         "weighted_loss_method": args.weighted_loss,
    #         "gradient_clip_norm": args.gradient_clip,
    #         "optimizer": args.optimizer,
    #     }, allow_val_change=True)

    # Record what code and environment produced this run, so that a residual spread
    # between two same-seed runs can be attributed to drift rather than to the RNG.
    provenance = collect_run_provenance(num_workers=args.num_workers)
    if wandb_logging:
        wandb.config.update({"provenance": provenance}, allow_val_change=True)

    # update model path if required
    config = handle_model_path(config, wandb_run_name, args.overwrite_model)
    path_model = config["training"]["path_to_save_model"]

    # Load the CSV file with labels and paths to audios
    csv_file_path = config["audio_csv_file"]["name"]
    df = pd.read_csv(csv_file_path, low_memory=False)

    # checks & logging
    check_dataset(df, strict=False, custom_msg="Initial: ", silent=False)

    # Apply subset ratio for prototyping
    subset_ratio = args.subset_ratio
    if subset_ratio is not None:
        df = apply_balanced_subset_sampling(
            df, subset_ratio, config["global"]["seed"]
        )

    # apply filtering & remapping and create split datasets
    # apply dataset config and create subsets
    # pop the dataset mappings from the config dict
    if filter_config is not None:
        dataset_config = None
        if "datasets" in filter_config.keys():
            dataset_config = filter_config.pop("datasets")

        # apply the filters specified in the config
        df = apply_user_filters(df, filter_config)
        logging.info("Applied user filters.")

        # apply dataset mappings if existing
        if dataset_config is not None:
            df = apply_user_dataset_config(df, dataset_config)

    # =============================================
    # CLASS IMBALANCE HANDLING
    # =============================================
    # Two mutually exclusive strategies:
    # 1. Weighted Sampler: Oversample minority classes during training
    #    - Sets shuffle=False (sampler handles randomness)
    #    - No class weights in loss function
    # 2. Weighted Loss: Apply class weights to loss function
    #    - Normal shuffling in DataLoader
    #    - Weights applied via loss_fn parameter
    # =============================================

    # Determine strategy from command line or config
    weighted_sampler_method = config["training"].get(
        "weighted_sampler_method", None
    )
    weighted_loss_method = config["training"].get("weighted_loss_method", None)
    weights_for_logging = None  # Will be set later if weighting is used

    # Validate mutual exclusivity
    if weighted_sampler_method is not None and weighted_loss_method is not None:
        raise ValueError(
            "Cannot use both weighted_sampler and weighted_loss simultaneously. "
            f"Config has weighted_sampler_method='{weighted_sampler_method}' and "
            f"weighted_loss_method='{weighted_loss_method}'. Set one to 'none'."
        )

    # if we are reusing a trained model, we need to make sure to use the same
    # class - label mapping, load and check it before preparing datasets.
    if args.evaluate_only:
        # 1. check if model has a class-idx mapping in it's checkpoint
        path_model = config["model"].get("eval_model_path", None)
        if path_model is None:
            raise ValueError(
                "No model to evaluate. Pass --eval_model_path or set ['model']['eval_model_path'] in the config."
            )
        eval_model_cp = load_model_cp(path_model, device)
        # will need batch shape for later
        batch_shape = eval_model_cp["batch_shape"]

        model_class_to_type_dict = eval_model_cp.get("class_to_type_dict", None)
        num_classes_model = eval_model_cp.get("num_classes", None)

        if model_class_to_type_dict is None:
            # 2. if not, check if mapping is in config
            # path to a class_to_type_dict
            config_class_to_type_dict_path = config["model"].get(
                "class_to_type_dict", None
            )
            assert (
                config_class_to_type_dict_path is not None
            ), """Model checkpoint has no 
            class_to_type_dict nor was a path to a yml file containing it provided in the config under ['model']['class_to_type_dict'].
            Cannot be sure which label in the data will correspond to which model class, which makes evaluation impossible!
            """
            logging.info(
                "Applying class_to_type dict from config to the dataset"
            )
            # load the yml into a dict
            class_to_type_dict = load_config(config_class_to_type_dict_path)

        else:
            logging.info(
                "Applying class_to_type dict from model checkpoint to the dataset"
            )
            class_to_type_dict = model_class_to_type_dict

        df = assign_codes_based_on_previous_mapping(
            df,
            class_to_type_dict,
            num_classes_model,
            label_column="class_name",
            code_column="class",
            reassign_unknown=True,  # reassign classes the model does not have ot "other_insect"
        )
        logging.info(
            "Class codes (labels) in dataset reassigned to match model."
        )

    else:
        # if it's a new model, update class labels solely based on dataset
        df, class_to_type_dict = assign_codes_ordered_by_frequency(
            df, label_column="class_name", code_column="class"
        )
        logging.info("Reassigned class codes (labels) in dataset.")

    if args.store_results:
        name, result_path = handle_result_path(path_model, args)
        # store dataset metadata

        path_csv_updated, path_dict_updated = store_dataset_metadata(
            f"training_data_{name}",
            result_path,
            df,
            class_to_type_dict,
            update_metadata_yml=False,
        )

    # if there are still differences in species sets, let the script fail
    # only use strict mode if we are not evaluating
    # (during evaluation on a different dataset it could be that not all subsets have data)
    check_dataset(
        df, strict=(not args.evaluate_test_only), custom_msg="After filtering: "
    )

    df_train, df_valid, df_test = create_datasets(df)

    # we can do this since we shouldn't have NAs in the subset column
    log_dataset_subsets(df_train, df_valid, df_test)

    # final check
    check_dataset(
        pd.concat([df_train, df_valid, df_test]),
        strict=(not args.evaluate_test_only),
        custom_msg="Final check: ",
        silent=False,
    )

    logging.info(
        "------------------------- Initialize model -------------------------"
    )

    if args.evaluate_only:
        num_classes = num_classes_model

    else:
        num_classes = df["class"].nunique()

    # Load and initialize the model
    freeze_base = args.freeze_base
    logging.info(
        "Base model is freezed."
        if freeze_base
        else "Base model is not freezed."
    )
    model_source = config["model"]["model_source"]
    # when evaluating, the checkpoint replaces all weights: skip downloading
    # pretrained weights
    pretrain = "none" if args.evaluate_only else args.pretrain
    model = initialize_model(
        config,
        config_trsf,
        model_source,
        num_classes,
        freeze_base,
        pretrain,
    )
    model = model.to(device)

    # create model summary (layers, trainable params etc)
    summary_str = str(summary(model, verbose=0))
    logging.debug("Model summary:")
    logging.debug(summary_str)

    logging.info("Model initialized and moved to device %s.", device)

    # define the resizing size if the model has fixed input size
    if model_source == "timm" and model.default_cfg["fixed_input_size"]:
        resizing_size = model.default_cfg["input_size"][1:]
        logging.info(
            "Will add transformation `Resize` to shape %s to the augmentations.",
            resizing_size,
        )
    else:
        resizing_size = None

    logging.info(
        "------------------------- Create transforms and dataloaders -------------------------"
    )

    aug_sweep_params = {}
    wave_augmentations_scale = args.wave_augmentations_scale
    if sweep_params is not None and wave_augmentations_scale is not None:
        raise UserWarning(
            "Cannot use command line argument 'wave_augmentations_scale' during sweep!"
        )
    # in case of sweep, grab augmentation override params
    elif sweep_params is not None:
        aug_sweep_params = sweep_params["augmentations"]
    elif wave_augmentations_scale is not None:
        aug_sweep_params["waveform_aug_scale"] = wave_augmentations_scale

    # Get p_override from command line arguments
    p_override = args.p_override
    if p_override is not None and aug_sweep_params:
        raise UserWarning(
            "Cannot use command line argument 'p_override' during sweep or with command line argument 'wave_augmentations_scale'!"
        )
    elif p_override is not None:
        if not 0 <= p_override <= 1:
            raise ValueError(
                f"p_override must be between 0 and 1, got {p_override}"
            )
        logging.info(
            f"Using global augmentation probability override: {p_override}"
        )

    # define transformations
    # verify compose by calculating the number of expected transforms
    logging.info(
        "------------------------- Create TEST transforms -------------------------"
    )
    # Don't pass augmentation overrides
    augmentations_test = create_transforms(config_trsf, resizing_size, "test")
    test_total_p = verify_compose(augmentations_test)
    logging.info(
        f"augmentations_test: Total expected number of transforms : {test_total_p}"
    )

    if not args.evaluate_test_only:
        logging.info(
            "------------------------- Create TRAIN transforms -------------------------"
        )
        augmentations_train = create_transforms(
            config_trsf,
            resizing_size,
            "train",
            aug_sweep_params,
            df,
            p_override=p_override,
        )
        train_total_p = verify_compose(augmentations_train)
        train_non_test_p = round(train_total_p - test_total_p, 2)
        logging.info(
            f"augmentations_train: Total expected number of transforms : {train_total_p}"
        )
        logging.info(
            f"...expected number of non-test transforms : {train_non_test_p}"
        )

        # push to wandb for easy comparison
        if wandb_logging:
            wandb.log({"augmentation_p": train_non_test_p})
        logging.info(
            "------------------------- Create VALID transforms -------------------------"
        )
        augmentations_valid = create_transforms(
            config_trsf,
            resizing_size,
            "valid",
            aug_sweep_params,
            df,
            p_override=p_override,
        )
        valid_total_p = verify_compose(augmentations_valid)
        valid_non_test_p = round(valid_total_p - test_total_p, 2)
        logging.info(
            f"augmentations_valid: Total expected number of transforms : {valid_total_p}"
        )
        logging.info(
            f"...expected number of non-test transforms : {valid_non_test_p}"
        )

    # create the datasets
    dataset_config = config["training"].get("dataset").copy()
    dataset_class_str = dataset_config.pop("class")
    logging.info(f"Using dataset class {dataset_class_str}")
    dataset_class = getattr(custom_datasets, dataset_class_str)
    logging.info("Dataset config: %s", pformat(dataset_config))

    test_dataset = dataset_class(
        df_test, transform=augmentations_test, **dataset_config
    )
    train_dataset = pd.DataFrame()
    valid_dataset = pd.DataFrame()

    if not args.evaluate_test_only:
        train_dataset = dataset_class(
            df_train, transform=augmentations_train, **dataset_config
        )
        valid_dataset = dataset_class(
            df_valid, transform=augmentations_valid, **dataset_config
        )
        # update dfs (in case they are modified in the dataset_class)
        df_train = train_dataset.get_file_df()
        df_valid = valid_dataset.get_file_df()

    df_test = test_dataset.get_file_df()

    check_dataset(
        pd.concat([df_train, df_valid, df_test]),
        strict=True,  # (not args.evaluate_test_only),
        custom_msg="After dataset_class.get_file_df() call: ",
    )

    # repeat logging of subset percentages
    log_dataset_subsets(train_dataset, valid_dataset, test_dataset)

    # Dedicated generator for the dataloaders, so that batch order and the per-worker
    # augmentation seeds do not depend on how much of the global RNG stream model
    # construction happened to consume. (set_seed() already ran, see top of main().)
    g = torch.Generator()
    g.manual_seed(seed)

    # Create the dataloaders
    batch_size = config["training"]["batch_size"]
    logging.info("Using a batch size of %s", batch_size)

    mixup_alpha = config["training"].get("mixup_alpha", None)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=make_collate_fn(),
    )

    if not args.evaluate_test_only:

        # sampler = None
        # shuffle = True
        beta = config["training"].get("effective_num_beta", None)

        if weighted_sampler_method is not None:
            logging.info(
                f"Using WeightedRandomSampler with method: {weighted_sampler_method}"
            )
            if weighted_sampler_method == "effective_num_samples":
                logging.info(f"  beta = {beta}")

            # calculate class weights
            class_weights_sampler = calculate_class_weights(
                df_train, "class", method=weighted_sampler_method, beta=beta
            )

            # Store for logging
            weights_for_logging = {
                "type": "sampler",
                "method": weighted_sampler_method,
                "weights": class_weights_sampler.tolist(),
            }

            # for sampler class weights needs to be given for each item
            labels = torch.tensor(df_train["class"].values, dtype=torch.long)
            sampler_weights = class_weights_sampler[labels].to(torch.float64)
            log_tensor_stats("Sampler weights", sampler_weights)

            # create weighted sampler
            # generator=g: without it the sampler draws from the global torch RNG,
            # so the epoch composition depends on every other RNG consumer
            sampler = WeightedRandomSampler(
                weights=sampler_weights,
                num_samples=len(sampler_weights),
                replacement=True,
                generator=g,
            )
            shuffle = False  # Sampler handles the randomness
        else:
            logging.info("Not using WeightedRandomSampler.")
            sampler = None
            shuffle = True

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=make_collate_fn(mixup_alpha, num_classes),
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=make_collate_fn(),
        )
    logging.info("Initialized dataloader(s) with %s workers", args.num_workers)

    logging.info(
        "------------------------- Setting up loss function and optimizer -------------------------"
    )

    # get loss configuration
    loss_type = config["training"].get("loss_type", "cross_entropy")
    label_smoothing = config["training"].get("label_smoothing", 0.0)

    # Calculate class weights for loss function if specified
    class_weights_loss = None
    if weighted_loss_method is not None and not args.evaluate_test_only:
        logging.info(
            f"Using class weights in loss function with method: {weighted_loss_method}"
        )
        if weighted_loss_method == "effective_num_samples":
            logging.info(f"with beta={beta}")

        class_weights_loss = calculate_class_weights(
            df_train, "class", method=weighted_loss_method, beta=beta
        )

        # Store for logging
        weights_for_logging = {
            "type": "loss",
            "method": weighted_loss_method,
            "weights": class_weights_loss.tolist(),
        }

        if class_weights_loss is not None:
            class_weights_loss = class_weights_loss.to(device)
            log_tensor_stats("Loss class weights", class_weights_loss)
    else:
        logging.info("Not using class weights in loss function.")

    # Set up the loss function based on the configuration
    if loss_type == "focal":
        gamma = config["training"].get("focal_gamma", 2.0)
        normalize_by_alpha = config["training"].get(
            "focal_normalize_by_alpha", False
        )
        loss_fn = FocalLoss(
            alpha=class_weights_loss,
            gamma=gamma,
            reduction="mean",
            label_smoothing=label_smoothing,
            normalize_by_alpha=normalize_by_alpha,
        )
        logging.info(
            f"Using Focal Loss with gamma={gamma}, label_smoothing={label_smoothing}, normalize_by_alpha={normalize_by_alpha}"
        )
    elif loss_type == "cross_entropy":
        # Define loss function with weighted cross entropy
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights_loss,
            reduction="mean",
            label_smoothing=label_smoothing,
        )
        logging.info(
            f"Using Cross Entropy Loss with label_smoothing={label_smoothing}"
        )
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Choose 'focal' or 'cross_entropy'"
        )

    metrics_train = {}
    metrics_val = {}
    metrics_test = {}

    if not args.evaluate_only:
        if args.log_test_eval:
            logging.info(
                "Evaluating the model on the test set after each epoch."
            )
        if config["training"]["log_train_eval"]:
            logging.info(
                "Evaluating the model on the training set after each epoch."
            )

        logging.info(
            f"Using optimizer {config['training'].get('optimizer', None)}"
        )

        # create the optimizer
        optimizer = set_optimizer(
            # args.optimizer,
            model,
            config["training"],
            args.freeze_base,
        )

        # Log recommendations related to class weighting
        log_class_weight_recommendations(
            weighted_sampler_method=weighted_sampler_method,
            weighted_loss_method=weighted_loss_method,
            # optimizer_name=config["training"].get("optimizer", None),
            training_config=config["training"],
        )

        logging.info(
            "------------------------- Starting training -------------------------"
        )

        drop_rate = config["model"].get("drop_rate", 0)
        if drop_rate:
            logging.info("drop_rate: %s", drop_rate)
        drop_path_rate = config["model"].get("drop_path_rate", 0)
        if drop_path_rate:
            logging.info("drop_path_rate: %s", drop_path_rate)
        mixup_alpha = config["training"].get("mixup_alpha", None)
        if mixup_alpha:
            logging.info("mixup_alpha: %s", mixup_alpha)

        # enter main training loop
        train(
            model,
            optimizer,
            loss_fn,
            train_dataloader,
            valid_dataloader,
            class_to_type_dict,
            test_dataloader if args.log_test_eval else None,
            device,
            wandb_logging,
            config,
            config_trsf,
        )
        # load the best model checkpoint
        best_model_cp = load_model_cp(path_model, device)
        batch_shape = best_model_cp["batch_shape"]

        # load the state dictionary into the model
        model.load_state_dict(best_model_cp["model_state_dict"])
        logging.info("Loaded the best model from %s", path_model)

    else:
        # was already loaded earlier
        model.load_state_dict(eval_model_cp["model_state_dict"])
        logging.info(f"Skip training and evaluate the model at {path_model}")

    if not args.evaluate_test_only:
        logging.info(
            "------------------------- Evaluate final model on full training set ------------------------"
        )

        # get training metrics config. we can do this since update_config adds defaults if anything is missing
        metrics_config_train = config["metrics"]["train"]

        # Training set: evaluate model, compute metrics and log
        outputs_train = evaluate_model(model, train_dataloader, loss_fn, device)
        metrics_train = compute_metrics(
            outputs_train, metrics_config_train, mode="train_final"
        )
        log_metrics(metrics_train)

        logging.info(
            "------------------------- Evaluate final model on validation set ------------------------"
        )

        # Validation set: evaluate model, compute metrics and log
        metrics_config_val = config["metrics"]["val"]
        outputs_val = evaluate_model(model, valid_dataloader, loss_fn, device)
        metrics_val = compute_metrics(
            outputs_val, metrics_config_val, mode="val_final"
        )
        log_metrics(metrics_val)

    logging.info(
        "---------------------------- Evaluate on test set ---------------------------"
    )

    # Test set: evaluate model, compute metrics and log
    metrics_config_test = config["metrics"]["test"]
    outputs_test = evaluate_model(model, test_dataloader, loss_fn, device)
    metrics_test = compute_metrics(
        outputs_test, metrics_config_test, mode="test_final"
    )
    log_metrics(metrics_test)

    # log all metrics to wandb
    if wandb_logging:
        # metrics.update_wandb_summary()
        wandb.run.summary.update(
            {
                **metrics_train,
                **metrics_val,
                **metrics_test,
            }
        )
        logging.info("Updated wandb summary")

        # Log class weights if used
        log_class_weights_to_wandb(weights_for_logging, class_to_type_dict)

    # store config file and csv files for train, val and test set
    if args.store_results:
        logging.info(
            "---------------------------- Save results ---------------------------"
        )
        logging.info("Store results and config file at %s", result_path)

        save_results(
            df_test,
            outputs_test,
            "test",
            result_path,
            class_to_type_dict,
            aggregations=["file_prob_mean"],
        )
        if args.save_file_probs:
            # per-file `file_prob_mean` probabilities, not just hard chunk predictions
            save_file_probabilities(
                outputs_test, "test", result_path, class_to_type_dict
            )

        if not args.evaluate_test_only:
            # add predictions to csv-file, create and store confusion matrices
            save_results(
                df_train,
                outputs_train,
                "train",
                result_path,
                class_to_type_dict,
                aggregations=["file_prob_mean"],
            )

            save_results(
                df_valid,
                outputs_val,
                "valid",
                result_path,
                class_to_type_dict,
                aggregations=["file_prob_mean"],
            )
            if args.save_file_probs:
                save_file_probabilities(
                    outputs_val, "valid", result_path, class_to_type_dict
                )

        # save command used to execute the script + config files
        save_command(sys.argv, result_path)
        save_config(config, result_path, config_name="config.yml")
        save_config(config_trsf, result_path, config_name="config_trsf.yml")

        # save a more complete model summary
        summary_str = capture_summary(model, input_size=batch_shape)
        save_model_summary(
            summary_str, result_path, model_name=config["model"]["model_name"]
        )

    # never delete a model that was only evaluated
    if not args.store_models and not args.evaluate_only:
        # safely delete the model file again
        if os.path.isfile(path_model):
            os.remove(path_model)
            logging.info("Deleted model file again at %s", path_model)


# =============================================
# W&B SWEEP
# =============================================


def sweep_training(arguments):
    """Sweep over the training parameters"""

    # Initialize a new wandb run
    run = wandb.init()

    # Load hyperparameters from the current sweep run
    run_hyperparameters = run.config

    logging.info(
        f"Initializing sweep '{run.name}' training with configuration:"
    )
    logging.info(str(run_hyperparameters))
    # Run main function with command line arguments and hyperparameters
    main(arguments, run_hyperparameters, run.name)


# =============================================
# ENTRY
# =============================================
if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # configure logging
    logging_level = logging.DEBUG if args.verbose else logging.INFO

    # silence annoying matplotlib debug messages
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    coloredlogs.install(level=logging_level)

    if args.sweep:
        # gracefully handle wandb agents killing off runs:
        signal.signal(signal.SIGTERM, handle_sigterm)

        logging.info("Joining wandb sweep.")
        args.store_results = False
        args.store_models = False
        # start the training
        sweep_training(args)
    else:
        main(args)

    # finalize
    logging.info("")
    logging.info(40 * "=")
    logging.info("DONE")
    logging.info(40 * "=")
