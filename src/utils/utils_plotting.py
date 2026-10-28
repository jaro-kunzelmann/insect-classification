import os
import logging
from typing import List, Optional, Tuple, Union, Any

import warnings

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from upsetplot import UpSet, from_indicators


def create_confusion_matrix(
    cf: np.ndarray,
    group_names: Optional[List[str]] = None,
    categories: Union[List[str], str] = "auto",
    count: bool = False,
    percent: bool = False,
    cbar: bool = True,
    xyticks: bool = True,
    xyplotlabels: bool = True,
    sum_stats: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "Blues",
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap diagram.
    Args:
        cf (np.ndarray): Confusion matrix to be passed in.
        group_names (Optional[List[str]]): List of strings that represent the labels row by row to be shown in each square.
            Default is None.
        categories (Union[List[str], str]): List of strings containing the categories to be displayed on the x, y axis.
            Default is "auto".
        count (bool): If True, show the raw number in the confusion matrix. Default is True.
        percent (bool): If True, show the proportions for each category. Default is True.
        cbar (bool): If True, show the color bar. The cbar values are based on the values in the confusion matrix.
            Default is True.
        xyticks (bool): If True, show x and y ticks. Default is True.
        xyplotlabels (bool): If True, show "True Label" and "Predicted Label2 on the figure. Default is True.
        sum_stats (bool): If True, display summary statistics below the figure. Default is True.
        figsize (Optional[Tuple[float, float]]): Tuple representing the figure size. Default is None.
        cmap (str): Colormap of the values displayed from matplotlib.pyplot.cm. Default is "Blues".
        title (Optional[str]): Title for the heatmap. Default is None.
        ax (Optional[Any]): Axes object to plot the heatmap on. If None, a new figure and axes will be created.
            Default is None.
    Returns:
        Tuple[plt.Figure, plt.Axes]: The Matplotlib figure and axes objects.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
        print(group_labels)
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.1%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    plt.rcParams.update({"font.size": 12})

    # MAKE THE HEATMAP VISUALIZATION
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = ax
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
        ax=ax,
    )

    if xyplotlabels:
        ax.set(
            ylabel="True label",
            xlabel="Predicted label" + stats_text,
        )
    else:
        ax.set(xlabel=stats_text)

    if title:
        ax.set(title=title)

    return fig, ax


def create_cm(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    figsize: Tuple[int, int] = (10, 10),
    path: Optional[str] = None,
    name: Optional[str] = None,
    show_fig: bool = True,
    **kwargs: Union[bool, Any],
) -> None:
    """
    Display confusion matrix.
    Args:
        true_labels (torch.Tensor): The true labels of the dataset.
        pred_logits (torch.Tensor): The predicted logits for the dataset.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (10, 10).
        path (str, optional): The file path to save the plot in. Defaults to None.
        name (str, optional): The name of the plot file. Defaults to None.
        show_fig (bool, optional): Whether to show the figure or not. Defaults to True.
    Returns:
        None
    """

    # predicted probabilities
    # pred_probs = torch.sigmoid(pred_logits)
    # # predicted labels
    # pred_labels = pred_probs.argmax(dim=1)

    # create and plot confusion matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    fig, ax = create_confusion_matrix(conf_mat, figsize=figsize, **kwargs)

    if name is None:
        name = "confusion_matrix"

    if path is not None:
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # save plot to file
        path_to_file = os.path.join(path, name + ".png")
        logging.info(f"Saving confusion matrix plot to {path_to_file}")
        fig.savefig(path_to_file)

    if show_fig:
        plt.show()


def create_dataset_plots(df, name, plot_dir):
    # TODO: add a plot for train/val/test split, add other plots from flash talk
    # class distribution plot (file count), top_n, with labels
    top_n = 30
    n_classes = df["class_name"].nunique()
    vc = df["class_name"].value_counts()
    top_classes = vc.head(top_n)

    # calculate the percentage of files shown on the plot
    num_files = vc.sum()
    perc_shown = top_classes.sum() / num_files * 100
    perc_shown = perc_shown.round(2)

    plt.figure(figsize=(10, 6))
    top_classes.plot(
        kind="bar",
        title=f"[{name}] Class distribution by files (top {top_n} classes, {perc_shown}% of all files)",
    )
    plt.xlabel("Class name")
    plt.ylabel("File count")
    plt.tight_layout()

    if plot_dir:
        plt.savefig(os.path.join(plot_dir, f"{name}_class_distr_by_top_n.png"))
    else:
        plt.show()
    plt.close()  # free up memory

    # Showing all classes
    plt.figure(figsize=(10, 6))
    # width=1.0: no gaps between bars
    plt.bar(range(len(vc)), vc.values, width=1.0)
    plt.title(
        f"[{name}] Class frequency by class rank (all n = {n_classes} classes, {num_files} files in total)"
    )
    plt.xlabel("Class rank")
    plt.ylabel("File count")
    # tiny gap before and after
    plt.xlim(-3, len(vc) + 3)

    # remove negative tick labels
    # get current tick positions
    ticks = plt.xticks()[0]

    # filter out negative ticks
    new_ticks = [tick for tick in ticks if tick >= 0]
    new_labels = [str(int(tick)) for tick in new_ticks]

    # apply filtered ticks and labels
    plt.xticks(ticks=new_ticks, labels=new_labels)

    plt.tight_layout()

    if plot_dir:
        plt.savefig(os.path.join(plot_dir, f"{name}_class_freq_by_rank.png"))

    else:
        plt.show()
    plt.close()  # free up memory

    # total duration per class
    duration_per_class = (
        df.groupby("class_name")["audio_length"]
        .sum()
        .sort_values(ascending=False)
    )
    # to hours
    duration_per_class /= 3600
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(duration_per_class)), duration_per_class.values, width=1.0
    )
    plt.title(
        f"[{name}] Total audio duration by class rank (all n = {n_classes} classes, {num_files} files in total)"
    )
    plt.xlabel("Class rank")
    plt.ylabel("Total duration (hours)")
    # plt.xticks([], [])
    plt.xlim(-3, len(duration_per_class) + 3)

    # remove negative tick labels
    # get current tick positions
    ticks = plt.xticks()[0]

    # filter out negative ticks
    new_ticks = [tick for tick in ticks if tick >= 0]
    new_labels = [str(int(tick)) for tick in new_ticks]

    # apply filtered ticks and labels
    plt.xticks(ticks=new_ticks, labels=new_labels)

    plt.tight_layout()

    if plot_dir:
        plt.savefig(
            os.path.join(plot_dir, f"{name}_duration_by_rank_top_n.png")
        )
    else:
        plt.show()
    plt.close()  # free up memory

    # CDF
    cdf = vc.cumsum() / vc.sum()
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cdf)), cdf.values)
    plt.title(
        f"[{name}] Cumulative distribution of class frequencies (all n = {n_classes} classes, {num_files} files in total)"
    )
    plt.xlabel("Class rank")
    # plt.xticks([], [])
    plt.ylabel("Cumulative proportion of files")
    # tiny gap before and after
    plt.xlim(-3, len(cdf) + 3)

    # remove negative tick labels
    # get current tick positions
    ticks = plt.xticks()[0]

    # filter out negative ticks
    new_ticks = [tick for tick in ticks if tick >= 0]
    new_labels = [str(int(tick)) for tick in new_ticks]

    # apply filtered ticks and labels
    plt.xticks(ticks=new_ticks, labels=new_labels)

    plt.tight_layout()

    if plot_dir:
        plt.savefig(os.path.join(plot_dir, f"{name}_duration_by_rank.png"))
    else:
        plt.show()
    plt.close()  # free up memory

    # sample rate
    # total duration per class
    duration_per_sr = (
        df.groupby("sample_rate")["audio_length"].sum().sort_index()
    )
    # to hours
    duration_per_sr /= 3600
    plt.figure(figsize=(10, 6))
    duration_per_sr.plot(kind="bar")
    # plt.bar(duration_per_sr.index, duration_per_sr.values, width=1.0)
    plt.title(
        f"[{name}] Total audio duration by sample rate (all n = {n_classes} classes, {num_files} files in total)"
    )
    plt.xlabel("Sample rate")
    plt.ylabel("Total duration (hours)")
    plt.tight_layout()

    if plot_dir:
        plt.savefig(os.path.join(plot_dir, f"{name}_duration_by_sr.png"))
    else:
        plt.show()
    plt.close()

    # distribution of file length (count)
    bin_size = 0.5
    max_durations = [None, 200, 10]
    # to hours
    durations = df["audio_length"].dropna()
    total_duration_all = durations.sum()

    for max_duration in max_durations:
        if max_duration:
            durations = durations[durations <= max_duration]

        if len(durations) == 0:
            # skip plotting in case we get an empty df here
            continue

        total_durations = durations.sum()

        durations_sum = total_durations.sum()
        prop_total_durations = int((durations_sum / total_duration_all) * 100)

        plt.figure(figsize=(10, 6))
        plt.hist(
            durations, bins=int(durations.max() / bin_size), edgecolor="black"
        )
        plt.xlabel("Audio duration (seconds)")
        plt.ylabel("File count")
        duration_str = f'{bool(max_duration) * ("(durations <= " + str(max_duration) + "s)" )}'
        plt.title(
            f"[{name}] Distribution of audio file lengths {duration_str} ({prop_total_durations}% of total duration shown)"
        )
        plt.tight_layout()

        if plot_dir:
            plt.savefig(
                os.path.join(
                    plot_dir,
                    f"{name}_count_by_file_length_{str(max_duration)}.png",
                )
            )
        else:
            plt.show()
        plt.close()

    # distribution of file length (duration)
    durations = df["audio_length"].dropna()

    # to hours
    total_duration_all = durations.sum() / 3600

    for max_duration in max_durations:
        if max_duration:
            durations = durations[durations <= max_duration]

        if len(durations) == 0:
            # skip plotting in case we get an empty df here
            continue

        # define bins
        bins = np.arange(0, durations.max() + bin_size, bin_size)
        hist, bin_edges = np.histogram(durations, bins=bins)

        # compute total duration per bin
        # use the midpoint of each bin multiplied by count
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        total_durations = hist * bin_midpoints
        # to hours
        total_durations /= 3600

        durations_sum = total_durations.sum()
        prop_total_durations = int((durations_sum / total_duration_all) * 100)

        # plot
        plt.figure(figsize=(10, 6))
        plt.bar(
            bin_midpoints,
            total_durations,
            width=bin_size * 0.9,
            edgecolor="black",
        )
        plt.xlabel("Audio duration (seconds)")
        plt.ylabel("Total audio time (hours)")
        duration_str = f'{bool(max_duration) * ("(durations <= " + str(max_duration) + "s)" )}'
        plt.title(
            f"[{name}] Total audio duration per bin {duration_str} ({prop_total_durations}% of total duration shown)"
        )
        plt.tight_layout()

        if plot_dir:
            plt.savefig(
                os.path.join(
                    plot_dir,
                    f"{name}_duration_by_file_length_{str(max_duration)}.png",
                )
            )
        else:
            plt.show()
        plt.close()


def plot_upset(summary_class_dataset_count, df_ch_species, output_path_plots):
    # suppress annoying pandas warning in UpSet calls
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # convert counts to binary presence/absence
        class_dataset_binary = summary_class_dataset_count > 0

        # create UpSet-compatible data
        upset_data = from_indicators(
            class_dataset_binary.columns.tolist(), class_dataset_binary
        )
        # plotting
        # UpSet overview globally
        plt.figure(figsize=(10, 6))
        UpSet(
            upset_data, show_counts=True, sort_categories_by="cardinality"
        ).plot()
        plt.title(
            "Global species coverage across datasets (UpSet plot)", y=1.02
        )

        if output_path_plots:
            # create file path if the plat is supposed to be saved
            output_fpath = os.path.join(output_path_plots, "upset_global.png")
            plt.savefig(output_fpath)
        else:
            plt.show()
        plt.close()

        # UpSet overview, Swiss
        df_binary_ch = class_dataset_binary[
            class_dataset_binary.index.isin(df_ch_species["class_name"])
        ]

        # create UpSet-compatible data
        upset_data_ch = from_indicators(
            df_binary_ch.columns.tolist(), df_binary_ch
        )

        plt.figure(figsize=(10, 6))
        UpSet(
            upset_data_ch, show_counts=True, sort_categories_by="cardinality"
        ).plot()
        plt.title("Swiss species coverage across datasets (UpSet plot)", y=1.02)

    if output_path_plots:
        # create file path if the plat is supposed to be saved
        output_fpath = os.path.join(output_path_plots, "upset_ch.png")
        plt.savefig(output_fpath)
    else:
        plt.show()
    plt.close()


def plot_count_heatmap_ch(
    summary_class_dataset_count, df_ch_species, output_path_plots
):

    # filter down to swiss species
    df_plot = summary_class_dataset_count[
        summary_class_dataset_count.index.isin(df_ch_species["class_name"])
    ]
    # manually drop env sound dataset
    df_plot = df_plot.drop(columns=["ESC-50", "ECOSoundSet_envsound_strong"])

    # sort by total
    total = df_plot.sum(axis=1).sort_values()
    df_plot = df_plot.reindex(total.index)

    # create plot
    plt.figure(figsize=(8, 20))
    sns.heatmap(
        df_plot,
        cmap="Blues",
        norm=LogNorm(vmin=1, vmax=df_plot.max().max()),
        annot=True,
        fmt="d",
        cbar=False,
    )
    plt.title("File counts per Swiss species per dataset")
    plt.xticks(rotation=45, ha="right")  # Rotate dataset labels
    for y in range(0, len(df_plot) + 1):
        plt.hlines(
            y,
            xmin=0,
            xmax=len(df_plot.columns),
            color="lightgray",
            linestyle="-",
            linewidth=0.7,
        )
    plt.xlabel("Dataset")
    plt.ylabel("Species")
    plt.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()

    if output_path_plots:
        # create file path if the plat is supposed to be saved
        output_fpath = os.path.join(output_path_plots, "count_heatmap_ch.png")
        plt.savefig(output_fpath)
    else:
        plt.show()

    plt.close()

    return df_plot


def plot_duration_heatmap_ch(
    summary_class_dataset_duration, df_ch_species, output_path_plots
):
    # filter down to swiss species
    df_plot = summary_class_dataset_duration[
        summary_class_dataset_duration.index.isin(df_ch_species["class_name"])
    ]

    # manually drop env sound dataset
    df_plot = df_plot.drop(columns=["ESC-50", "ECOSoundSet_envsound_strong"])

    # sort by total
    total = df_plot.sum(axis=1).sort_values()
    df_plot = df_plot.reindex(total.index)

    # create plot
    plt.figure(figsize=(8, 20))
    sns.heatmap(
        df_plot,
        cmap="Greens",
        norm=LogNorm(
            vmin=max(df_plot[df_plot > 0].min().min(), 1),  # avoid log(0)
            vmax=df_plot.max().max(),
        ),
        annot=True,
        fmt=".1f",
        cbar=False,
    )
    plt.title("Total duration (min) per Swiss species per dataset")
    plt.xticks(rotation=45, ha="right")
    for y in range(0, len(df_plot) + 1):
        plt.hlines(
            y,
            xmin=0,
            xmax=len(df_plot.columns),
            color="lightgray",
            linestyle="-",
            linewidth=0.7,
        )
    plt.xlabel("Dataset")
    plt.ylabel("Species")
    plt.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()

    if output_path_plots:
        # create file path if the plat is supposed to be saved
        output_fpath = os.path.join(
            output_path_plots, "duration_heatmap_ch.png"
        )
        plt.savefig(output_fpath)
    else:
        plt.show()

    plt.close()

    return df_plot


def plot_dataset_classes_heatmap(summary_dataset_classes, output_path_plots):
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(summary_dataset_classes > 0, cmap="Greys", cbar=False)

    # bring back spines (black box around the plot)
    for spine in ax.spines.values():
        spine.set_visible(True)

    for y in range(1, len(summary_dataset_classes)):
        ax.hlines(
            y, *ax.get_xlim(), color="lightgray", linestyle="-", linewidth=0.7
        )

    # # Set x-ticks to be class ranks, showing every 20th
    num_classes = summary_dataset_classes.shape[1]
    tick_positions = list(range(0, num_classes, 20))
    tick_labels = [str(i + 1) for i in tick_positions]

    # formatting
    plt.ylabel("Dataset")
    plt.xlabel("Class rank", labelpad=10)
    plt.title("Class presence across datasets", pad=10)
    plt.xticks(tick_positions, tick_labels, rotation=0)
    plt.tight_layout()

    if output_path_plots:
        # create file path if the plat is supposed to be saved
        output_fpath = os.path.join(
            output_path_plots, "class_presence_heatmap.png"
        )
        plt.savefig(output_fpath)
    else:
        plt.show()

    plt.close()


def plot_dataset_classes_heatmap_with_count(
    df, summary_dataset_classes, output_path_plots
):

    vc = df["class_name"].value_counts()

    fig, (ax_bar, ax_heatmap) = plt.subplots(
        2,
        1,  # 2 rows, 1 column
        sharex=True,  # share the x-axis between bar and dot plots
        figsize=(13, 8),
        gridspec_kw={"height_ratios": [1, 3]},  # optional: taller dot plot
    )

    # row 0: barplot
    # ax_bar.bar(range(len(vc)), vc.values, width=1.0, color="black")
    ax_bar.plot(range(len(vc)), vc.values, color="black")

    # # scale & labels
    ax_bar.set_yscale("log")

    # manual gridlines
    for y in [10, 100, 1000]:
        ax_bar.axhline(
            y, color="lightgray", linestyle="-", linewidth=0.7, zorder=1
        )

    ax_bar.set_title("Overall file count per class (log scale)", pad=10)
    ax_bar.set_ylabel("File count", labelpad=12)

    # row 1: heatmap
    sns.heatmap(
        summary_dataset_classes > 0, cmap="Greys", cbar=False, ax=ax_heatmap
    )

    # bring back spines (black box around the plot)
    for spine in ax_heatmap.spines.values():
        spine.set_visible(True)

    for y in range(1, len(summary_dataset_classes)):
        ax_heatmap.hlines(
            y,
            *ax_heatmap.get_xlim(),
            color="lightgray",
            linestyle="-",
            linewidth=0.7,
        )

    # # Set x-ticks to be class ranks, showing every 20th
    num_classes = summary_dataset_classes.shape[1]
    tick_positions = list(range(0, num_classes, 20))
    tick_labels = [str(i + 1) for i in tick_positions]

    # formatting
    ax_heatmap.set_ylabel("Dataset")
    ax_heatmap.set_xlabel("Class rank", labelpad=10)
    ax_heatmap.set_title("Class presence across datasets", pad=10)
    ax_heatmap.set_xticks(tick_positions, tick_labels, rotation=0)
    ax_bar.set_xticks(tick_positions, tick_labels, rotation=0)
    plt.tight_layout()

    if output_path_plots:
        # create file path if the plat is supposed to be saved
        output_fpath = os.path.join(
            output_path_plots, "class_presence_heatmap_with_counts.png"
        )
        plt.savefig(output_fpath)
    else:
        plt.show()

    plt.close()
