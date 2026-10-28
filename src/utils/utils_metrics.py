from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Literal
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    classification_report,
)

PredictionType = Literal[
    "logits",
    "probabilities",
    "scores",
    "labels",
]

# =============================================
# Defaults
# =============================================
DEFAULT_METRICS_BY_AGG = {
    "chunks": ["accuracy", "balanced_accuracy", "f1_macro"],
    "file_logit_mean": ["accuracy", "balanced_accuracy", "f1_macro"],
    "file_prob_mean": ["accuracy", "balanced_accuracy", "f1_macro"],
    "file_prob_max": ["accuracy", "balanced_accuracy", "f1_macro"],
    "file_vote": ["accuracy", "balanced_accuracy", "f1_macro"],
}


# =============================================
# Functions
# =============================================


@dataclass
class EvalOutputs:
    """
    Dataclass for storing model outputs.
    y_pred are logits by default
    """

    y_true: torch.Tensor  # [N] (no soft labels)
    y_pred: torch.Tensor  # [N, C]  (logits or probs)
    prediction_type: PredictionType = "logits"
    loss: Optional[float] = None  # scalar (mean over N samples)
    paths: Optional[list[str]] = None  # length N, one per chunk/sample

    def to_probs(self) -> torch.Tensor:
        if self.y_pred.ndim != 2:
            raise ValueError("y_pred must have shape [N, C].")

        if self.prediction_type == "logits":
            return torch.softmax(self.y_pred, dim=1)

        if self.prediction_type == "probabilities":
            return self.y_pred

        raise ValueError(
            f"Cannot convert prediction type "
            f"{self.prediction_type!r} to probabilities "
            "without an explicit normalization policy."
        )

    def hard_preds(self) -> torch.Tensor:
        if self.prediction_type == "labels":
            if self.y_pred.ndim != 1:
                raise ValueError("Label predictions must have shape [N].")
            return self.y_pred.long()

        if self.y_pred.ndim != 2:
            raise ValueError("Score predictions must have shape [N, C].")

        return self.y_pred.argmax(dim=1)

    def hard_labels(self) -> torch.Tensor:
        """Return hard true labels [N]. (y_true is already hard.)"""
        if self.y_true.ndim != 1:
            raise ValueError("y_true must be hard labels [N].")
        return self.y_true


# aggregation functions
def group_indices_by_path(paths: list[str]) -> Dict[str, List[int]]:
    # helper function to group indices by path
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, p in enumerate(paths):
        groups[p].append(idx)
    return groups


def aggregate_identity(outputs: EvalOutputs) -> EvalOutputs:
    """No aggregation: chunk-level metrics."""
    return outputs


def aggregate_file_logit_mean(outputs: EvalOutputs) -> EvalOutputs:
    """Aggregate multiple chunks per file via logit mean."""
    if outputs.paths is None:
        raise ValueError("aggregate_file_logit_mean requires EvalOutputs.paths")

    groups = group_indices_by_path(outputs.paths)
    y_true = outputs.y_true
    logits = outputs.y_pred

    file_labels = []
    file_logits = []
    file_paths = []

    for path, idxs in groups.items():
        idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        chunk_logits = logits[idxs_tensor]  # [k, C]

        agg_logits = chunk_logits.mean(dim=0)  # [C]

        file_labels.append(
            y_true[idxs_tensor[0]]
        )  # assume one true label per file
        file_logits.append(agg_logits)
        file_paths.append(path)

    y_true_file = torch.stack(file_labels, dim=0)  # [F]
    logits_file = torch.stack(file_logits, dim=0)  # [F, C]

    return EvalOutputs(
        y_true=y_true_file,
        y_pred=logits_file,
        prediction_type="logits",
        paths=file_paths,
    )


def aggregate_file_prob_mean(outputs: EvalOutputs) -> EvalOutputs:
    """Aggregate multiple chunks per file via mean probability."""
    if outputs.paths is None:
        raise ValueError("aggregate_file_prob_mean requires EvalOutputs.paths")

    if outputs.prediction_type != "logits":
        raise ValueError(
            "aggregate_file_logit_mean requires prediction_type='logits', "
            f"got {outputs.prediction_type!r}."
        )

    groups = group_indices_by_path(outputs.paths)
    y_true = outputs.y_true

    probs = outputs.to_probs()

    file_labels = []
    file_probs = []
    file_paths = []

    for path, idxs in groups.items():
        idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        chunk_probs = probs[idxs_tensor]  # [k, C]

        agg_probs = chunk_probs.mean(dim=0)  # [C]

        file_labels.append(y_true[idxs_tensor[0]])
        file_probs.append(agg_probs)
        file_paths.append(path)

    y_true_file = torch.stack(file_labels, dim=0)  # [F]
    probs_file = torch.stack(file_probs, dim=0)  # [F, C]

    return EvalOutputs(
        y_true=y_true_file,
        y_pred=probs_file,
        prediction_type="probabilities",
        paths=file_paths,
    )


def aggregate_file_prob_max(outputs: EvalOutputs) -> EvalOutputs:
    """Aggregate multiple chunks per file via max probability over chunks."""
    if outputs.paths is None:
        raise ValueError("aggregate_file_prob_max requires EvalOutputs.paths")

    groups = group_indices_by_path(outputs.paths)
    y_true = outputs.y_true

    probs = outputs.to_probs()

    file_labels = []
    file_probs = []
    file_paths = []

    for path, idxs in groups.items():
        idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        chunk_probs = probs[idxs_tensor]  # [k, C]

        agg_probs, _ = chunk_probs.max(dim=0)  # [C]

        file_labels.append(y_true[idxs_tensor[0]])
        file_probs.append(agg_probs)
        file_paths.append(path)

    y_true_file = torch.stack(file_labels, dim=0)  # [F]
    probs_file = torch.stack(file_probs, dim=0)  # [F, C]

    return EvalOutputs(
        y_true=y_true_file,
        y_pred=probs_file,
        prediction_type="scores",
        paths=file_paths,
    )


def aggregate_file_vote(outputs: EvalOutputs) -> EvalOutputs:
    """Aggregate multiple chunks per file via majority vote of predicted classes."""
    if outputs.paths is None:
        raise ValueError("aggregate_file_vote requires EvalOutputs.paths")

    if outputs.y_pred.ndim != 2:
        raise ValueError(
            "aggregate_file_vote currently requires predictions with shape [N, C]."
        )

    num_classes = outputs.y_pred.shape[1]
    pred_classes = outputs.hard_preds()

    groups = group_indices_by_path(outputs.paths)
    y_true = outputs.y_true

    # per-chunk predicted classes
    pred_classes = outputs.hard_preds()

    file_labels = []
    file_paths = []

    file_predictions = []

    for path, idxs in groups.items():
        idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        chunk_preds = pred_classes[idxs_tensor]

        counts = torch.bincount(
            chunk_preds,
            minlength=num_classes,
        )
        winner = counts.argmax()

        file_labels.append(y_true[idxs_tensor[0]])
        file_predictions.append(winner)
        file_paths.append(path)

    return EvalOutputs(
        y_true=torch.stack(file_labels),
        y_pred=torch.stack(file_predictions),
        prediction_type="labels",
        paths=file_paths,
    )


AGGREGATION_REGISTRY = {
    "chunks": aggregate_identity,
    "file_logit_mean": aggregate_file_logit_mean,
    "file_prob_mean": aggregate_file_prob_mean,
    "file_prob_max": aggregate_file_prob_max,
    "file_vote": aggregate_file_vote,
}


# metrics
def prepare_labels(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert targets and predictions to hard class labels on CPU.

    Supports:
    - y_pred [N, C]: logits, probabilities or class scores
    - y_pred [N]: already-decoded class IDs
    """
    if y_true.ndim != 1:
        raise ValueError(
            f"y_true must have shape [N], got {tuple(y_true.shape)}"
        )

    if y_pred.ndim == 2:
        y_pred_hard = y_pred.argmax(dim=1)
    elif y_pred.ndim == 1:
        y_pred_hard = y_pred
    else:
        raise ValueError(
            "y_pred must have shape [N] or [N, C], "
            f"got {tuple(y_pred.shape)}"
        )

    return (
        y_true.detach().cpu().long(),
        y_pred_hard.detach().cpu().long(),
    )


def metric_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    y_true_cpu, y_pred_cpu = prepare_labels(y_true, y_pred)
    return float(accuracy_score(y_true_cpu.numpy(), y_pred_cpu.numpy()))


def metric_balanced_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    y_true_cpu, y_pred_cpu = prepare_labels(y_true, y_pred)
    return float(
        balanced_accuracy_score(y_true_cpu.numpy(), y_pred_cpu.numpy())
    )


def metric_f1_macro(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    y_true_cpu, y_pred_cpu = prepare_labels(y_true, y_pred)
    return float(
        f1_score(y_true_cpu.numpy(), y_pred_cpu.numpy(), average="macro")
    )


METRIC_REGISTRY = {
    "accuracy": metric_accuracy,
    "balanced_accuracy": metric_balanced_accuracy,
    "f1_macro": metric_f1_macro,
}


def compute_metrics(
    outputs: EvalOutputs,
    metrics_by_agg: Dict[str, List[str]],
    mode: str,
) -> Dict[str, float]:
    results: Dict[str, float] = {}

    for agg_name, metric_names in metrics_by_agg.items():
        agg_fn = AGGREGATION_REGISTRY[agg_name]

        # This flag describes the input to the aggregation.
        agg_outputs = agg_fn(outputs)

        for metric_name in metric_names:
            metric_fn = METRIC_REGISTRY[metric_name]

            value = metric_fn(agg_outputs.y_true, agg_outputs.y_pred)

            results[f"{mode}/{agg_name}/{metric_name}"] = value

    results[f"{mode}/loss"] = outputs.loss
    return results


def compute_classwise_metrics_by_aggregation(
    outputs,
    aggregation_names,
    *,
    labels=None,
    target_names=None,
    default_from_logits=True,
):
    reports = {}

    for agg_name in aggregation_names:
        agg_fn = AGGREGATION_REGISTRY[agg_name]

        agg_outputs = agg_fn(outputs)

        y_true, y_pred = prepare_labels(
            agg_outputs.y_true,
            agg_outputs.y_pred,
        )

        report = classification_report(
            y_true.numpy(),
            y_pred.numpy(),
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=np.nan,
        )

        reports[agg_name] = pd.DataFrame(report).transpose()

    return reports
