import logging
import yaml
from utils.utils_metrics import DEFAULT_METRICS_BY_AGG

# =============================================


_NONE_LIKE = {"none", "null", "nil", "n/a", "na", "nan", ""}


def normalize_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _NONE_LIKE:
        return None
    return value


def normalize_config(obj):
    """
    Recursively normalize 'none-like' values to Python None.
    """
    if isinstance(obj, dict):
        return {k: normalize_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_config(v) for v in obj]
    return normalize_none(obj)


def load_config(config_file_path):
    """
    load yaml configuration file
    """
    try:
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            if config is None:
                logging.warning(f"Empty config loaded from {config_file_path}")
            return normalize_config(config)
    except FileNotFoundError:
        logging.error("Config file %s not found", config_file_path)
        raise
    except yaml.YAMLError as error:
        logging.error(
            "Error parsing config file %s: %s", config_file_path, error
        )
        raise


def update_config(config, args, sweep_params, device):
    """Updates the configuration with command line arguments for the experiment"""

    if args.wandb is not None:
        config["wandb"]["logging"] = args.wandb

    # ============================================================
    # Command line args override config
    # ============================================================
    if args.model_path is not None:
        config["training"]["path_to_save_model"] = args.model_path
        logging.info(
            "Path to save model set to %s via command line", args.model_path
        )

    if getattr(args, "eval_model_path", None) is not None:
        config["model"]["eval_model_path"] = args.eval_model_path
        logging.info(
            "Model to evaluate set to %s via command line", args.eval_model_path
        )

    config.setdefault("global", {})
    if getattr(args, "seed", None) is not None:
        config["global"]["seed"] = args.seed
        logging.info(f"Seed set to {args.seed} via command line")

    if getattr(args, "deterministic_warn_only", False):
        config["global"]["deterministic_warn_only"] = True
        logging.warning(
            "Determinism violations will only warn (run is not reproducible)"
        )

    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
        logging.info(f"Number of epochs set to {args.epochs} via command line")

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
        logging.info(f"Batch size set to {args.batch_size} via command line")

    # Override optimizer if specified via CLI
    if args.optimizer is not None:
        config["training"]["optimizer"] = args.optimizer
        logging.info(f"Optimizer set to {args.optimizer} via command line")

    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
        logging.info(
            f"Learning rate set to {args.learning_rate} via command line"
        )

    if args.lr_scheduler_t_max is not None:
        if "lr_scheduler" not in config["training"]:
            config["training"]["lr_scheduler"] = {}
        config["training"]["lr_scheduler"]["T_max"] = args.lr_scheduler_t_max
        logging.info(
            f"Override lr_scheduler T_max from config: {args.lr_scheduler_t_max}"
        )

    if args.gradient_clip is not None:
        config["training"]["gradient_clip_norm"] = args.gradient_clip
        logging.info(
            f"Gradient clipping set to {args.gradient_clip} via command line"
        )

    if args.drop_rate is not None:
        config["model"]["drop_rate"] = args.drop_rate
        logging.info(f"Drop rate set to {args.drop_rate} via command line")

    if args.drop_path_rate is not None:
        config["model"]["drop_path_rate"] = args.drop_path_rate
        logging.info(
            f"Drop path rate set to {args.drop_path_rate} via command line"
        )

    if args.mixup_alpha is not None:
        config["training"]["mixup_alpha"] = args.mixup_alpha
        logging.info(f"Mixup alpha set to {args.mixup_alpha} via command line")

    # Override weighted sampler method if specified
    if args.weighted_sampler is not None:
        config["training"]["weighted_sampler_method"] = args.weighted_sampler
        logging.info(
            f"Weighted sampler method set to {args.weighted_sampler} via command line"
        )

    # Override weighted loss method if specified
    if args.weighted_loss is not None:
        config["training"]["weighted_loss_method"] = args.weighted_loss
        logging.info(
            f"Weighted loss method set to {args.weighted_loss} via command line"
        )

    # Override beta if specified
    if args.effective_num_beta is not None:
        config["training"]["effective_num_beta"] = args.effective_num_beta
        logging.info(
            f"Effective number beta set to {args.effective_num_beta} via command line"
        )

    # Override focal gamma if specified
    if args.focal_gamma is not None:
        config["training"]["focal_gamma"] = args.focal_gamma
        logging.info(f"Focal gamma set to {args.focal_gamma} via command line")

    # ============================================================
    # Sweep params override anything else
    # ============================================================
    # TODO a better pattern for this...
    if sweep_params:
        if "learning_rate" in sweep_params:
            config["training"]["learning_rate"] = sweep_params["learning_rate"]

        if "epochs" in sweep_params:
            config["training"]["num_epochs"] = sweep_params["epochs"]
            # also have to adapt T_max (for CosineAnnealing)
            if (
                config["training"]
                .get("lr_scheduler", None)
                .get("scheduler", None)
                == "CosineAnnealingLR"
            ):
                config["training"]["lr_scheduler"]["T_max"] = sweep_params[
                    "epochs"
                ]

        if "batch_size" in sweep_params:
            config["training"]["batch_size"] = sweep_params["batch_size"]

        if "weight_decay" in sweep_params:
            config["training"]["weight_decay"] = sweep_params["weight_decay"]

        if "label_smoothing" in sweep_params:
            config["training"]["label_smoothing"] = sweep_params[
                "label_smoothing"
            ]

        if "drop_rate" in sweep_params:
            config["model"]["drop_rate"] = sweep_params["drop_rate"]

        if "drop_path_rate" in sweep_params:
            config["model"]["drop_path_rate"] = sweep_params["drop_path_rate"]

        if "mixup_alpha" in sweep_params:
            config["training"]["mixup_alpha"] = sweep_params["mixup_alpha"]

        if "optimizer" in sweep_params:
            config["training"]["optimizer"] = sweep_params["optimizer"]

        if "weighted_sampler_method" in sweep_params:
            config["training"]["weighted_sampler_method"] = sweep_params[
                "weighted_sampler_method"
            ]

        if "weighted_loss_method" in sweep_params:
            config["training"]["weighted_loss_method"] = sweep_params[
                "weighted_loss_method"
            ]

        if "effective_num_beta" in sweep_params:
            config["training"]["effective_num_beta"] = sweep_params[
                "effective_num_beta"
            ]

        if "focal_gamma" in sweep_params:
            config["training"]["focal_gamma"] = sweep_params["focal_gamma"]

        if "gradient_clip_norm" in sweep_params:
            config["training"]["gradient_clip_norm"] = sweep_params[
                "gradient_clip_norm"
            ]

    # ============================================================
    # Other config updates
    # ============================================================

    if args.model_source is not None:
        config["model"]["model_source"] = args.model_source

    if args.log_train_eval is not None:
        config["training"]["log_train_eval"] = args.log_train_eval

    # add device to config
    config["model"]["device"] = device

    # ensure the "metrics" section exists
    metrics_config = config.setdefault("metrics", {})

    # fill defaults for each subset
    metrics_config.setdefault("train", DEFAULT_METRICS_BY_AGG)
    metrics_config.setdefault("val", DEFAULT_METRICS_BY_AGG)
    metrics_config.setdefault("test", DEFAULT_METRICS_BY_AGG)

    config["metrics"] = metrics_config

    # normalize config again to handle any "none-like" values from CLI or sweep
    config = normalize_config(config)

    # Add raw ("unormalized") command line arguments to config
    config["cl_args"] = vars(args)

    return config
