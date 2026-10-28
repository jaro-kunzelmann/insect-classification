# standard modules
import time
import logging
from pprint import pformat
import numpy as np

# typehinting
from typing import Dict, Union, Callable

# torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

# import wandb
import wandb

# loading bars
from tqdm import tqdm

# custom modules
from utils.utils_functions import log_wandb, log_metrics
from utils.utils_metrics import EvalOutputs, compute_metrics

# default values
DEFAULT_NUM_EPOCHS = 5
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL_PATH = "./models/default_model.pth"
DEFAULT_MIN_LR = 0.00005
DEFAULT_WANDB_LOGGING = False


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: Union[str, torch.device] = "cpu",
) -> EvalOutputs:
    """
    Run a model over an entire DataLoader and collect per-sample outputs.

    This function performs a forward pass over all batches in the dataloader,
    accumulates the total loss, and returns an EvalOutputs object containing:

    - y_true: concatenated ground-truth labels   (shape [N])
      If MixUp was used and labels come in as soft labels (shape [N, C]),
      they are converted back to hard labels using argmax.

    - y_pred: concatenated raw model outputs (logits or probabilities)
              (shape [N, C])

    - prediction_type: What kind of prediction format we have ("logits",
        "probabilities",
        "scores",
        "one_hot")

    - loss: mean loss over all samples

    - paths: list of file paths corresponding to each sample, extracted from
             batch["fpath_audio"]

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.

    dataloader : DataLoader
        A dataloader yielding batches with keys "audio", "label", and "fpath_audio".

    loss_fn : Callable
        Loss function taking (y_pred, y_true) and returning a scalar tensor.

    device : str or torch.device, optional (default="cpu")
        Device to run inference on.

    Returns
    -------
    EvalOutputs
        A dataclass containing y_true, y_pred, prediction_type, loss, and paths for all samples.
    """
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []
    all_paths = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Evaluating model",
            leave=False,
            total=len(dataloader),
        ):
            audio = batch["audio"].to(device)
            y_true = batch["label"].to(device)
            y_pred = model(audio)

            loss = loss_fn(y_pred, y_true)
            batch_size = y_true.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pred_labels.append(y_pred.cpu())
            true_labels.append(y_true.cpu())
            all_paths.extend(batch["fpath_audio"])

    # Concatenate collected tensors
    preds = torch.cat(pred_labels, dim=0)
    labels = torch.cat(true_labels, dim=0)
    mean_loss = total_loss / total_samples

    # Convert MixUp soft labels to hard labels
    if labels.ndim == 2:
        labels = labels.argmax(dim=1)

    return EvalOutputs(
        y_true=labels,
        y_pred=preds,
        prediction_type="logits",
        loss=mean_loss,
        paths=all_paths,
    )


def save_model(
    model,
    optimizer,
    epoch,
    valid_loss,
    model_path,
    num_classes,
    class_to_type_dict,
    durations,
    batch_shape,
    input_shape,
    y_true_shape,
    y_pred_shape,
    config_trsf,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes,
            "class_to_type_dict": class_to_type_dict,
            "valid_loss": valid_loss,
            "durations": durations,
            "batch_shape": batch_shape,
            "input_shape": input_shape,
            "y_true_shape": y_true_shape,
            "y_pred_shape": y_pred_shape,
            "config_trsf": config_trsf,
        },
        model_path,
    )


def log_initial_batch(
    batch_shape, input_shape, y_true_shape, y_pred_shape, wandb_logging
):
    logging.info("Tensor shapes:")
    logging.info("Input batch shape:")
    logging.info(batch_shape)
    logging.info("Single input instance shape:")
    logging.info(input_shape)
    logging.info("y_true_shape shape:")
    logging.info(y_true_shape)
    logging.info("y_pred shape:")
    logging.info(y_pred_shape)
    logging.info("Input features:")
    num_features = batch_shape[-2] * batch_shape[-1]
    logging.info(num_features)
    if wandb_logging:
        wandb.log(
            {
                "spect_x": batch_shape[-1],
                "spect_y": batch_shape[-2],
                "input_shape_batch": str(batch_shape),
                "input_features": num_features,
            }
        )


# training loop
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: Callable,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    class_to_type_dict: Dict,
    test_dataloader: DataLoader = None,
    device: str = "cpu",
    wandb_logging: bool = False,
    config: Dict = None,
    config_trsf: Dict = None,
) -> None:
    """
    This function trains a given model using the specified optimizer and loss function on the provided training and validation data loaders.
    The training loop runs for a given number of epochs and logs the training and validation losses.
    The learning rate scheduler is used to adjust the learning rate during training.
    The best model (with the lowest validation loss) is saved during the training process.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (optim.Optimizer): The optimizer used for training.
        loss_fn (callable): The loss function used to calculate the loss between predicted and target values.
        train_dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        device (str, optional): The device to run the model on. Defaults to "cpu".
        wandb_logging (bool, optional): Whether to log the training process to wandb. Defaults to False.
        config (Dict, optional): Model and training configuration.

    Returns:
        None

    Example:
        model = MyModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        config = {"epochs": 5, "device": "cpu", "model_path": "./models/default_out.pth", "min_lr": 0.00005}
        train(model, optimizer, loss_fn, train_dataloader, val_dataloader, "cpu", True, config)
    """

    if config is None:
        config = dict()

    metrics_config = config.get("metrics", None)
    config = config.get("training", None)

    num_epochs = config.get("num_epochs", DEFAULT_NUM_EPOCHS)
    model_path = config.get("path_to_save_model", DEFAULT_MODEL_PATH)
    lr_scheduler_params = config.get("lr_scheduler", None)
    grad_clip_value = config.get("gradient_clip_norm", None)

    # Log gradient clipping configuration
    if grad_clip_value is not None:
        logging.info(
            f"Using gradient clipping with max norm: {grad_clip_value}"
        )
    else:
        logging.info("Gradient clipping disabled")

    # learning rate scheduler configuration
    # if not specified in the config, no scheduler is used (=constant LR)
    scheduler = None
    if lr_scheduler_params is not None:
        lr_scheduler_class = lr_scheduler_params.pop("scheduler", None)
        logging.info(
            "Using learning rate scheduler %s with configuration: %s",
            lr_scheduler_class,
            pformat(lr_scheduler_params),
        )
        # get the scheduler, pass optimizer and params
        scheduler = getattr(optim.lr_scheduler, lr_scheduler_class)(
            optimizer, **lr_scheduler_params
        )

    early_stopping_patience = config.get("early_stopping_patience", None)
    num_classes = getattr(model, "num_classes", None)
    log_train_eval = config.get("log_train_eval", None)

    if device is None:
        device = DEFAULT_DEVICE

    dur = []
    avg_time = 0
    best_valid_loss = float("inf")
    epoch_best_valid_loss = 0
    epochs_since_improvement = 0

    early_stopping_message = (
        f"with early stopping patience of {early_stopping_patience}"
        if early_stopping_patience is not None
        else "without early stopping"
    )

    start_train_message = (
        f"Start training for {num_epochs} epochs {early_stopping_message}"
    )
    logging.info(start_train_message)
    logging.info("Training configuration:")
    logging.info("%s", pformat(config))

    initial_batch = True
    for epoch in range(num_epochs):
        logging.info(
            "----------------------------- Training (epoch %i of %i) ----------------------------- ",
            epoch + 1,
            num_epochs,
        )
        t0 = time.time()
        model.train()

        train_losses = []
        true_labels = []
        pred_labels = []
        all_paths = []
        num_samples = 0

        for batch in train_dataloader:

            audio = batch["audio"]
            audio = audio.to(device)

            y_true = batch["label"]
            y_true = y_true.to(device)

            batch_size = y_true.size(0)
            num_samples += batch_size
            all_paths.extend(batch["fpath_audio"])

            optimizer.zero_grad()

            # raw, unnormalized scores (logits)
            y_pred = model(audio)

            # log input sizes in the first batch
            if initial_batch == True:
                batch_shape = audio.shape
                input_shape = batch_shape[1:]
                y_true_shape = y_true.shape
                y_pred_shape = y_pred.shape

                log_initial_batch(
                    batch_shape,
                    input_shape,
                    y_true_shape,
                    y_pred_shape,
                    wandb_logging,
                )

                initial_batch = False

            # expects raw logits and integer class indices
            loss = loss_fn(y_pred, y_true)
            loss.backward()

            # Log gradient norm before clipping (for debugging - only first few batches)
            if epoch == 0 and wandb_logging:
                batch_idx = num_samples // batch_size
                if batch_idx < 10:  # Only log first 10 batches
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    wandb.log(
                        {
                            "grad_norm_before_clip": total_norm,
                            "batch": batch_idx,
                        }
                    )

            # Gradient clipping
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_value
                )

            optimizer.step()

            # Log the sum training loss for each batch
            train_losses.append(
                loss.item() * batch_size
            )  # multiply by batch size

            # store for later metric calculation
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()
            pred_labels.append(y_pred)
            true_labels.append(y_true)

        # Compute the average training loss for the epoch
        logging.debug(f"num_samples: {num_samples}")
        logging.debug(f"sum(train_losses): {sum(train_losses)}")
        train_loss_avg = sum(train_losses) / num_samples

        labels_train = torch.cat(true_labels)
        preds_train = torch.cat(pred_labels)

        # convert back to hard labels (for metrics calculations)
        # this is the case when mixup was used
        if labels_train.ndim == 2:
            labels_train = labels_train.argmax(dim=1)

        # collect training outputs, compute metrics
        outputs_train = EvalOutputs(
            y_true=labels_train,
            y_pred=preds_train,
            prediction_type="logits",
            loss=train_loss_avg,
            paths=all_paths,
        )

        metrics_train = compute_metrics(
            outputs_train, metrics_config["train"], mode="train"
        )

        # evaluate on validation set, compute metrics
        outputs_val = evaluate_model(model, val_dataloader, loss_fn, device)
        metrics_val = compute_metrics(
            outputs_val, metrics_config["val"], mode="val"
        )

        # Get the current learning rate(s) from the optimizer
        lr_dict = {
            f"lr_group_{i}": group["lr"]
            for i, group in enumerate(optimizer.param_groups)
        }

        # Update the learning rate
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(metrics_val.mean_loss)
        elif scheduler is not None:
            # e.g. CosineAnnealing
            scheduler.step()

        # Log the training progress
        t = time.time() - t0
        dur.append(t)
        avg_time = (avg_time * epoch + t) / (epoch + 1)
        log_wandb(
            wandb_logging,
            {
                "epoch": epoch + 1,
                "epoch_time": t,
                "avg_epoch_time": avg_time,
                **metrics_train,
                **metrics_val,
                **lr_dict,
            },
        )

        # Compute metrics for train data
        if log_train_eval:
            outputs_train = evaluate_model(
                model, train_dataloader, loss_fn, device
            )
            metrics_train = compute_metrics(
                outputs_train,
                metrics_config["train"],
                mode="train",
                return_paths=True,
            )

            log_wandb(
                wandb_logging,
                {"epoch": epoch + 1, **metrics_train},
            )

        # Evaluate and compute metrics for test data
        if test_dataloader is not None:
            outputs_test = evaluate_model(
                model, test_dataloader, loss_fn, device, return_paths=True
            )
            metrics_test = compute_metrics(
                outputs_test, metrics_config["test"], mode="test"
            )

            log_wandb(
                wandb_logging,
                {"epoch": epoch + 1, **metrics_test},
            )

        # if validation loss improves, save the model
        if outputs_val.loss < best_valid_loss:
            best_valid_loss = outputs_val.loss
            epoch_best_valid_loss = epoch
            epochs_since_improvement = 0  # reset counter
            log_wandb(
                wandb_logging,
                {
                    "epoch": epoch + 1,
                    "best_valid_loss": best_valid_loss,
                },
            )
            if wandb_logging:
                wandb.run.summary["epoch_best_valid_loss"] = (
                    epoch_best_valid_loss
                )
            logging.info("Best validation loss: %.5f", best_valid_loss)
            logging.info(
                "Saving best model as %s for epoch: %s", model_path, epoch + 1
            )
            save_model(
                model,
                optimizer,
                epoch + 1,
                best_valid_loss,
                model_path,
                num_classes,
                class_to_type_dict,
                dur,
                batch_shape,
                input_shape,
                y_true_shape,
                y_pred_shape,
                config_trsf,
            )
        else:
            epochs_since_improvement += 1

        # log every 10th epoch
        if epoch % 10 == 0:
            logging.info(
                "Epoch %s, Time %.1f s, Validation metrics:", epoch + 1, t
            )
            log_metrics(metrics_val)

            if log_train_eval:
                logging.info("Training metrics: %")
                log_metrics(metrics_train)

            if test_dataloader is not None:
                logging.info("Test metrics: %s", metrics_test)
                log_metrics(metrics_test)

        # check if early stopping is needed
        if (
            early_stopping_patience is not None
            and epochs_since_improvement >= early_stopping_patience
        ):
            logging.info(
                "Early stopping triggered. No improvement in validation loss for %s epochs.",
                early_stopping_patience,
            )
            break

    logging.info(
        "----------------------------- Training finished -----------------------------"
    )
