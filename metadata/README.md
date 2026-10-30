# Metadata

Metadata for *Optimizing Spectrogram Front-Ends and Training Strategies for
Deep Learning-Based Acoustic Insect Classification* (Peter & Kunzelmann).
These files define exactly which recordings and species the models were
trained and evaluated on.

The audio itself is not redistributed. All recordings come from
**InsectSet459 version 1.0** (Faiß & Stowell; Zenodo,
doi:10.5281/zenodo.15675220). They are identified by their InsectSet459
file names, so the exact input can be reconstructed from that release.

## Files

| File | Rows | Content |
|---|---|---|
| `files_used.csv` | 24,950 | Every recording used, with its split (train / validation / test), species, licence, contributor and source links |
| `files_excluded.csv` | 1,347 | Every InsectSet459 recording not used, with the reason |
| `species_table.csv` | 451 | One row per species (class), with Catalogue of Life, GBIF and ITIS identifiers |
| `species_table_README.md` | | Data dictionary and sources for `species_table.csv` |
| `run_configurations.csv` | 112 | Every training run of the front-end study (40) and the training-strategy sweep (72), with the settings that differ from the baseline configuration |
| `eml.xml` | | Ecological Metadata Language (EML 2.2.0) record describing this folder: creators, licence, taxonomic coverage, methods, and the columns of each CSV file |

## `files_used.csv` and `files_excluded.csv`

| Column | Meaning |
|---|---|
| `file_name` | File name in InsectSet459 v1.0 |
| `species` | Class label (InsectSet459 species name, `Genus_epithet`) |
| `group` | Orthoptera or Cicadidae |
| `split` | `train`, `validation` or `test`, as in the official InsectSet459 split |
| `sample_rate_hz` | Sample rate of the InsectSet459 file, before resampling to 48 kHz; empty for files excluded by the sample-rate filter |
| `file_size_bytes` | Size of the file in the InsectSet459 v1.0 zip archive |
| `crc32` | CRC-32 checksum of the file in the InsectSet459 v1.0 zip archive (hexadecimal) |
| `license` | Licence of the individual recording, as given by InsectSet459 |
| `contributor` | Recordist / rights holder, as given by InsectSet459 |
| `observation_url` | Source observation (iNaturalist or xeno-canto); for the 12 BioAcoustica recordings, the BioAcoustica publication |
| `source_file_url` | Original audio file at the source; for BioAcoustica recordings, the original file name |
| `reason` | `files_excluded.csv` only; see below |

`crc32` and `file_size_bytes` identify the exact bytes distributed by
InsectSet459. They are the values stored in the archives `Train.zip`,
`Validation.zip` and `Test.zip`, so they can be checked without extracting
the audio, for example with `unzip -v Train.zip`. File names and formats are
those of InsectSet459, which converted some source recordings (for example,
`.m4a` files are distributed as `.mp3`).

A recording was excluded for one of two reasons:

- `sample_rate_below_44.1kHz` (1,284 files): its sample rate is below
  44.1 kHz, so the 0.4–22 kHz band analysed by the models is not fully
  represented;
- `species_below_per_split_minimum` (63 files): the file passed the
  sample-rate filter, but its species then had fewer than 3 training,
  2 validation or 1 test recordings, so the species was removed (8 species).

Recordings keep the licence given in the `license` column. Most are
CC BY-NC 4.0 or CC BY-SA 4.0; reuse of the audio must follow each
recording's licence and credit its contributor.

## `run_configurations.csv`

One row per training run. Together with the two published configurations
of the baseline (`configs/models/effnet_baseline.yml` and
`configs/transforms/transforms_baseline.yml` in the code repository), each
row fully specifies the run.

| Column | Meaning |
|---|---|
| `run` | Short description of the run. The modified baseline and the selected configuration are marked |
| `type` | `front_end_study` (stage 1) or `training_sweep` (stage 2) |
| all other columns | A setting, named by its path in the configuration files (for example `training.mixup_alpha` is `mixup_alpha` in the `training` section). A value replaces the baseline value; an empty cell means the baseline value applies |

Notes:

- `order.train`, `order.valid` and `order.test` list the transforms applied,
  in order, separated by `; `. Parameters of a transform not in these lists
  have no effect and are left empty.
- `transforms.RandomSpecAugment.*` and `transforms.Spectrogram.*` /
  `transforms.ResizeSpectrogram.*` are not in the baseline configuration; add
  them as a section when the run uses that transform. In
  `RandomSpecAugment`, `p` is the probability of applying SpecAugment to a
  batch (the *p* in the paper), and `specaugment_p` is the maximum
  proportion of time steps masked. The mask sizes scale with the input size.
- `training.mixup_alpha`, `training.gradient_clip_norm` and
  `training.weighted_loss_method` can also be set on the command line
  (`--mixup_alpha`, `--gradient_clip`, `--weighted_loss`), as in the code
  repository's README.
- All other settings, including the seed (42), the number of epochs (20)
  and the weight decay (10⁻⁴), are the same for every run.

## How the files were made

- `files_used.csv` and `files_excluded.csv`: derived from the list of
  recordings used in training, the InsectSet459 v1.0 annotation file and
  zip archives, and the metadata written by the preprocessing pipeline
  (original sample rates). Every recording matches the annotation file and
  the zip archives, and the splits agree with the official InsectSet459
  split.
- `species_table.csv`: derived from Catalogue of Life COL24.4, the GBIF
  Backbone Taxonomy and ITIS; see `species_table_README.md`.
- `run_configurations.csv`: derived from the effective configuration
  logged for each run (after command-line overrides). Each row, applied to
  the baseline, reproduces the logged configuration, and the marked rows
  equal the published baseline and best-model configurations.

## How to cite

If you use these files, please cite the paper and the dataset:

- Peter, T. J., Kunzelmann, J. *Optimizing Spectrogram Front-Ends and
  Training Strategies for Deep Learning-Based Acoustic Insect
  Classification.* [CITATION]
- Faiß, M., Stowell, D. (2025). *InsectSet459: A Large Dataset for
  Automatic Acoustic Identification of Insects (Orthoptera and Cicadidae)*,
  version 1.0. Zenodo. https://doi.org/10.5281/zenodo.15675220

This folder is part of release v1.0.0 of the code repository
(https://github.com/jaro-kunzelmann/insect-classification), which is
archived on Zenodo. The DOI of the archived release is given on the
repository's main page.

## Licence

The metadata in this folder is released under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The licences of the individual recordings are listed in the `license`
column and are not changed by this release.
