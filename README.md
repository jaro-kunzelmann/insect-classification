# Acoustic Classification of Insects

Code to reproduce the data preparation, training and evaluation of the insect sound classifiers (Orthoptera and Cicadidae) trained on [InsectSet459](https://zenodo.org/records/15675220).

## Requirements

* Linux (the download scripts use `bash`, `wget`, `md5sum` and `unzip`)
* An NVIDIA GPU with CUDA support. GPUs of the Blackwell generation (e.g. RTX 50xx) are **not** supported: they require a newer PyTorch than the one used here, which in turn requires patching `torch_audiomentations`.
* Disk space:
  * ~130 GB for InsectSet459 (~156 GB at peak during download and extraction)
  * ~72 GB for the resampled audio files

  Once the audio is resampled, the original InsectSet459 recordings are no longer needed for training (see [Data preparation](#2-prepare-the-data)).

## 1. Set up

### Clone the repository and create the environment

```bash
git clone https://github.com/jaro-kunzelmann/insect-classification.git
cd insect-classification
```

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (or use `mamba`/`conda`), then create and activate the environment and install the code of this repository. The environment uses PyTorch 2.5.1 with CUDA 12.4 from the `pytorch` channel, which needs flexible channel priority:

```bash
micromamba env create -f conda_environment.yml --channel-priority flexible
micromamba activate insects
pip install -e .
```

### Download the trained models

The trained models in `models/` are stored with [Git LFS](https://git-lfs.com/). `git-lfs` is part of the environment, so once it is activated run:

```bash
git lfs install
git lfs pull
```

## 2. Prepare the data

All commands are run from the repository root. Every script lists its options with `-h`, e.g. `python src/data_prep/resample_audio.py -h`. See [data/README.md](data/README.md) for details on the datasets and the files each step produces.

### Download the data

```bash
# Impulse responses (OpenAIR), used for data augmentation
./scripts/download_IRs.sh

# InsectSet459 (~130 GB, ~156 GB at peak)
./scripts/download_IS459.sh
```

If you get `Permission denied`, make the scripts executable first: `chmod +x scripts/*.sh`. Both scripts verify the MD5 checksums of the downloads. `download_IS459.sh` can be rerun after an interruption and continues where it stopped.

### Prepare annotations and resample

```bash
# 1. collect the InsectSet459 metadata
python src/data_prep/metadata_prep_insectset.py

# 2. aggregate the datasets and check the species names
python src/data_prep/aggregate_datasets.py

# 3. filter: drop recordings with a sample rate below 44.1 kHz and
#    classes with too few files per split
python src/data_prep/filter_aggregation.py --filter-config-path configs/dataset_filtering/initial_training.yml

# 4. resample all audio to 48 kHz (needs ~72 GB of additional disk space)
python src/data_prep/resample_audio.py \
    -a ./data/aggregation/filtered/initial_training/meta/annotations_initial_training.csv \
    --target-sample-rate 48000 --max-workers 6 \
    --output-dir ./data/aggregation/resampled_files/
```

`--max-workers` sets the number of parallel processes. Do not use more workers than you have GB of RAM. Warnings about "Illegal Audio-MPEG-Headers" during resampling can be ignored.

The resulting dataset (`data/aggregation/resampled_files/meta/annotations_resampled.csv`) has 24,950 recordings of 451 classes. The original recordings in `data/InsectSet/` are no longer needed for training and evaluation and can be deleted.

For transparency, `IS459_files_used.csv` (also as `IS459_files_used.xlsx`) in the repository root lists all InsectSet459 recordings used to produce the results and models, along with their split (`train`, `val`, `test`).

## 3. Train a model

Training is run with `src/run_experiment.py`, configured by three files:

| Option | Config | Content |
|---|---|---|
| `--config` | `configs/models/` | model, training and data loading parameters |
| `--trsf_config` | `configs/transforms/` | audio augmentations and spectrogram parameters, per split |
| `--dataset-config-path` | `configs/dataset_filtering/` | filters applied to the dataset |

Command line options override the values in the config files. List all options with `python src/run_experiment.py -h`.

### Quick test

Train for one epoch on a random 10% of the data to check that everything is set up correctly:

```bash
python -u src/run_experiment.py \
    --config configs/models/effnet_baseline.yml \
    --trsf_config configs/transforms/mel64_specaug0.yml \
    --dataset-config-path configs/dataset_filtering/initial_training.yml \
    --weighted_loss inverse_sqrt_frequency --gradient_clip none --mixup_alpha 0.25 \
    --model_path models/test_run.pth \
    -wo 8 --save_file_probs --deterministic_warn_only -srt 0.1 -ep 1
```

### Reproduce the models

Best model (EfficientNetV2-S, 64 mel bins, mixup, class-weighted loss):

```bash
python -u src/run_experiment.py \
    --config configs/models/effnet_best.yml \
    --trsf_config configs/transforms/mel64_specaug0.yml \
    --dataset-config-path configs/dataset_filtering/initial_training.yml \
    --weighted_loss inverse_sqrt_frequency --gradient_clip none --mixup_alpha 0.25 \
    -wo 8 --save_file_probs --deterministic_warn_only
```

Baseline model (EfficientNetV2-S, 128 mel bins, no mixup, no class weighting):

```bash
python -u src/run_experiment.py \
    --config configs/models/effnet_baseline.yml \
    --trsf_config configs/transforms/transforms_baseline.yml \
    --dataset-config-path configs/dataset_filtering/initial_training.yml \
    -wo 8 --deterministic_warn_only
```

The model is saved at the path given by `--model_path` or, if not given, by `training.path_to_save_model` in the config. If that file exists already, a suffix is added instead of overwriting it (`--overwrite_model` overwrites it), so retraining does not replace the provided models in `models/`.

### Output

Each run writes to `results/<model name>/` (a numeric suffix is added if the folder exists):

* `config.yml`, `config_trsf.yml`, `run_command.sh`: the full configuration and command of the run
* `meta/`: annotations and label mapping of the data used
* `df_{train,valid,test}_file_prob_mean.csv`: per-file predictions (chunk probabilities averaged per file)
* `conf_matrix_*`: confusion matrices
* `probs_file_{valid,test}.csv`: per-file class probabilities (only with `--save_file_probs`)

### Reproducibility

The random seed is set in the config (`global.seed`) or with `--seed`, and deterministic algorithms are enabled. Some operations have no deterministic CUDA implementation, so the commands above use `--deterministic_warn_only`. This lets the run finish, but two runs with the same seed may differ slightly.

### Logging to Weights & Biases

Logging to [Weights & Biases](https://wandb.ai/site) is off by default. To enable it, run `wandb login` once and add `--wandb` (`-wb`) to the command. The project name is set in the config (`wandb.project`). To log to a team, set `WANDB_ENTITY` in the environment or in a `.env` file in the repository root.

## 4. Evaluate a trained model

Pass a model checkpoint with `--eval_model_path` to evaluate it without training. With `--evaluate_test_only` only the test set is evaluated; without it, the training, validation and test sets are evaluated.

Two trained models are provided in `models/`:

| Model | Checkpoint | Config | Transform config |
|---|---|---|---|
| Best model | `models/best_model.pth` | `effnet_baseline_mod_sweep.yml` | `mel64_specaug0.yml` |
| Baseline model | `models/baseline_model.pth` | `effnet_baseline_mod.yml` | `transforms_baseline.yml` |

Best model:

```bash
python -u src/run_experiment.py \
    --config configs/models/effnet_best.yml \
    --trsf_config configs/transforms/mel64_specaug0.yml \
    --dataset-config-path configs/dataset_filtering/initial_training.yml \
    --eval_model_path models/best_model.pth \
    --evaluate_test_only -wo 8 --deterministic_warn_only
```

Baseline model:

```bash
python -u src/run_experiment.py \
    --config configs/models/effnet_baseline.yml \
    --trsf_config configs/transforms/transforms_baseline.yml \
    --dataset-config-path configs/dataset_filtering/initial_training.yml \
    --eval_model_path models/baseline_model.pth \
    --evaluate_test_only -wo 8 --deterministic_warn_only
```

Always use the transform config the model was trained with. The class labels are taken from the checkpoint. The results are written to `results/<model name>_<dataset config name>/`.

## Repository structure

```
configs/
    dataset_filtering/   dataset filters
    models/              model & training configs
    transforms/          augmentation & spectrogram configs
data/                    datasets (downloaded and generated, see data/README.md)
models/                  trained models (Git LFS)
scripts/                 download scripts
src/
    data_prep/           data preparation scripts
    utils/               helper modules
    run_experiment.py    training & evaluation entry point
    custom_datasets.py   datasets and audio transforms
    models.py            model loading (timm, PANNs)
    train_eval.py        training loop and evaluation
```

## Tools and libraries

* [Weights & Biases](https://wandb.ai/site) for experiment tracking
* [timm](https://github.com/huggingface/pytorch-image-models) for pretrained models
* [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations) for audio data augmentation on the GPU
* [Black](https://black.readthedocs.io/en/stable/index.html) for code formatting

## License

The code in this repository is released under the [MIT License](LICENSE). This does not cover the datasets used, which remain under the terms of their original sources.

## Citation

If you use this code, please cite it using the metadata in [CITATION.cff](CITATION.cff).
