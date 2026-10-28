import os
from typing import Optional, Callable, Literal

from copy import deepcopy
import typing
import math
import warnings
import logging
import random
from glob import glob

import pandas as pd
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import (
    Spectrogram,
    Resample,
    MelScale,
    AmplitudeToDB,
    MelSpectrogram,
    SpecAugment,
)
from torchvision.transforms.v2 import MixUp
from torch_audiomentations import (
    Compose,
    OneOf,
    SomeOf,
    HighPassFilter,
    LowPassFilter,
    Gain,
    AddColoredNoise,
    ApplyImpulseResponse,
    AddBackgroundNoise,
)

# from torch_audiomentations.augmentations.random_crop import RandomCrop
from torch_audiomentations.utils.multichannel import is_multichannel
from torch_audiomentations.core.transforms_interface import (
    MultichannelAudioNotSupportedException,
)
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.core.transforms_interface import (
    BaseWaveformTransform,
)

from torchlibrosa.stft import LogmelFilterBank
from torchlibrosa.stft import Spectrogram as LibrosaSpectrogram
from torchlibrosa.augmentation import SpecAugmentation

import python_stretch
import librosa


class TimeStretchTorch(BaseWaveformTransform):
    """
    Time-stretch the signal without changing the pitch, using `python_stretch` (Signalsmith)
    or `librosa` under the hood.
    This transform acts as a wrapper and performs time-stretching on CPU using numpy.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        leave_length_unchanged: bool = True,
        method: Literal[
            "librosa_phase_vocoder", "signalsmith_stretch"
        ] = "signalsmith_stretch",
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param min_rate: Minimum time-stretch rate (<1 slows down, >1 speeds up)
        :param max_rate: Maximum time-stretch rate
        :param leave_length_unchanged: If True, the output is cropped/padded to match the input length.
                                       Essential for fixed-size batching.
        :param method: "signalsmith_stretch" (Higher quality, faster) or "librosa_phase_vocoder".
        :param mode: "per_example" (recommended), "per_channel", or "per_batch".
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided for TimeStretchTorch"
            )

        if min_rate < 0.1:
            raise ValueError("min_rate must be >= 0.1")
        if max_rate > 10:
            raise ValueError("max_rate must be <= 10")
        if min_rate > max_rate:
            raise ValueError("min_rate must not be greater than max_rate")
        if method not in ("librosa_phase_vocoder", "signalsmith_stretch"):
            raise ValueError(
                'method must be set to either "librosa_phase_vocoder" or "signalsmith_stretch"'
            )

        # Force leave_length_unchanged to True for compatibility
        if leave_length_unchanged is False:
            warnings.warn(
                "leave_length_unchanged=False is not currently not supported. "
                "Forcing to True."
            )

        self.leave_length_unchanged = True

        self._sample_rate = sample_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged
        self.method = method
        self._mode = mode

    def randomize_parameters(
        self,
        samples: Tensor = None,
        # following parameters are unused but kept for interface compatibility
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        Calculates random stretch rates based on the configured mode.
        """
        batch_size, num_channels, num_samples = samples.shape

        if self._mode == "per_example":
            self.transform_parameters["rates"] = [
                random.uniform(self.min_rate, self.max_rate)
                for _ in range(batch_size)
            ]
        elif self._mode == "per_channel":
            self.transform_parameters["rates"] = [
                [
                    random.uniform(self.min_rate, self.max_rate)
                    for _ in range(num_channels)
                ]
                for _ in range(batch_size)
            ]
        elif self._mode == "per_batch":
            self.transform_parameters["rates"] = [
                random.uniform(self.min_rate, self.max_rate)
            ]

    def _stretch_np(
        self,
        x: np.ndarray,
        rate: float,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Internal helper: Applies the stretch to a numpy array (Channels, Time) or (Time,).
        """
        original_shape = x.shape
        original_ndim = x.ndim

        # Ensure we have (Channels, Time) for processing
        if original_ndim == 1:
            x_channels = x[np.newaxis, :]  # add channel axis
        else:
            x_channels = x

        # Signalsmith Stretch Implementation
        if self.method == "signalsmith_stretch":
            # Signalsmith requires C-contiguous arrays
            if (
                original_ndim == 2
                and x_channels.shape[0] > 1
                and not x_channels.flags.c_contiguous
            ):
                x_channels = np.ascontiguousarray(x_channels)

            stretch = python_stretch.Signalsmith.Stretch()
            stretch.preset(x_channels.shape[0], sample_rate)
            stretch.setTimeFactor(rate)

            # Process returns a new numpy array (Channels, Time_out)
            y = stretch.process(x_channels)

        # Librosa Phase Vocoder Implementation
        else:
            if original_ndim == 1:
                y = librosa.effects.time_stretch(x, rate=rate)
            else:
                # Librosa's time_stretch expects (Frequency, Time), not (Channels, Time).
                # For multi-channel time-domain signals, we must loop over channels manually.
                channels = []
                for ch in range(x.shape[0]):
                    ch_y = librosa.effects.time_stretch(x[ch], rate=rate)
                    channels.append(ch_y)
                # Stack back to (Channels, Time)
                y = np.array(channels, dtype=x.dtype)

        # Restore original dimensionality if input was 1D
        if original_ndim == 1 and y.ndim > 1:
            y = y[0]

        return y

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        # 1. Validation and Setup
        device = samples.device  # to restore later
        dtype = samples.dtype
        batch_size, num_channels, original_length = samples.shape

        if sample_rate is not None and sample_rate != self._sample_rate:
            raise ValueError(
                f"sample_rate ({sample_rate}) must match the value passed into constructor ({self._sample_rate})"
            )
        sample_rate = self._sample_rate

        # 2. Move to CPU for processing
        # We detach to ensure no gradients are tracked for this numpy operation.
        # We explicitly cast to float() to ensure float32 dtype for librosa/signalsmith.
        samples_np = samples.detach().cpu().float().numpy()

        # List to hold processed numpy arrays before re-stacking
        processed_samples_list = []

        # 3. Apply Transform Loop
        for i in range(batch_size):
            # Extract current item (Channels, Time)
            current_item = samples_np[i]

            if self._mode == "per_channel":
                # Special handling for per-channel rates, every channel may have different rate
                stretched_channels = []
                for ch in range(num_channels):
                    rate = self.transform_parameters["rates"][i][ch]
                    y_ch = self._stretch_np(current_item[ch], rate, sample_rate)
                    stretched_channels.append(y_ch)

                # If lengths differ across channels ,
                # we must pad channels to match within this single item.
                max_ch_len = max(c.shape[-1] for c in stretched_channels)
                y_item = np.zeros(
                    (num_channels, max_ch_len), dtype=current_item.dtype
                )
                for ch_idx, ch_data in enumerate(stretched_channels):
                    y_item[ch_idx, : ch_data.shape[-1]] = ch_data
            else:
                # Standard per_example / per_batch
                if self._mode == "per_example":
                    rate = self.transform_parameters["rates"][i]
                elif self._mode == "per_batch":
                    rate = self.transform_parameters["rates"][0]
                else:
                    raise ValueError(f"Unknown mode: {self._mode}")

                y_item = self._stretch_np(current_item, rate, sample_rate)

            # 4. Handle Length Constraints (Crop/Pad)
            if self.leave_length_unchanged:
                # Force output to match original input length
                if y_item.shape[-1] > original_length:
                    # Crop
                    y_item = y_item[..., :original_length]
                elif y_item.shape[-1] < original_length:
                    # Pad
                    padding_len = original_length - y_item.shape[-1]
                    # np.pad expects ((before_axis0, after_axis0), (before_axis1, after_axis1)...)
                    # We only want to pad the last dimension (Time)
                    if y_item.ndim == 2:
                        pad_width = ((0, 0), (0, padding_len))
                    else:
                        pad_width = ((0, padding_len),)
                    y_item = np.pad(y_item, pad_width, mode="constant")

            processed_samples_list.append(y_item)

        # 5. Reassemble Batch
        if self.leave_length_unchanged:
            # All items are guaranteed to be 'original_length'
            samples_out_np = np.stack(processed_samples_list)
        else:
            # Variable lengths: Pad to the maximum length in this batch
            max_len = max(x.shape[-1] for x in processed_samples_list)
            samples_out_np = np.zeros(
                (batch_size, num_channels, max_len), dtype=samples_np.dtype
            )

            for i, x in enumerate(processed_samples_list):
                samples_out_np[i, ..., : x.shape[-1]] = x

        # 6. Convert back to Tensor and original device
        # Note: samples_out_np is float32, so we rely on 'dtype' to restore original precision if needed
        samples_out = torch.from_numpy(samples_out_np).to(
            device=device, dtype=dtype
        )

        return ObjectDict(
            samples=samples_out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class RandomSpecAugment(nn.Module):
    def __init__(
        self,
        n_time_masks: int,
        time_mask_param: int,
        n_freq_masks: int,
        freq_mask_param: int,
        iid_masks: bool = True,
        specaugment_p: float = 1.0,
        zero_masking: bool = False,
        p: float = 0.5,
    ):
        """
        Wraps torchaudio.transforms.SpecAugment and applies it with probability `p`.

        Args:
            All args are passed to the internal SpecAugment instance.
            p (float): Probability of applying SpecAugment at each forward pass.
        """
        super().__init__()
        self.p = p

        self.spec_augment = SpecAugment(
            n_time_masks=n_time_masks,
            time_mask_param=time_mask_param,
            n_freq_masks=n_freq_masks,
            freq_mask_param=freq_mask_param,
            iid_masks=iid_masks,
            p=specaugment_p,
            zero_masking=zero_masking,
        )

    def forward(self, x):
        if random.random() < self.p:
            return self.spec_augment(x)
        return x


class Pad(nn.Module):
    """Pad an audio signal to a fixed length."""

    def __init__(
        self,
        target_length: float,
        sample_rate: int,
        target_length_unit: str = "seconds",
        padding_type: str = "zeros",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.padding_type = padding_type
        if target_length_unit == "seconds":
            self.target_samples = int(self.sample_rate * target_length)
        elif target_length_unit == "samples":
            self.target_samples = int(target_length)
        else:
            raise ValueError(
                'target_length_unit must be "samples" or "seconds"'
            )

        if padding_type not in ["zeros", "repeat"]:
            raise ValueError('padding_type must be either "zeros" or "repeat"')

    def forward(self, waveform, sample_rate: Optional[int] = None):
        sample_rate = sample_rate or self.sample_rate
        padding_type = self.padding_type

        if sample_rate is None:
            raise RuntimeError("sample_rate is required")
        if waveform.shape[-1] < self.target_samples:
            # If the waveform is shorter than the target length, pad it
            if padding_type == "zeros":
                padding = torch.zeros(
                    waveform.shape[0],
                    waveform.shape[1],
                    self.target_samples - waveform.shape[-1],
                )
            elif padding_type == "repeat":
                repeats = (
                    math.ceil(self.target_samples / waveform.shape[-1]) - 1
                )
                padding = waveform.repeat(1, 1, repeats)
                padding = padding[: self.target_samples - waveform.shape[-1]]
            waveform = torch.cat((waveform, padding), dim=-1)
        return waveform


class RandomPad(nn.Module):
    """Pad an audio signal to a fixed length at a random center offset"""

    def __init__(
        self,
        target_length: float,
        sample_rate: int,
        target_length_unit: str = "seconds",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        if target_length_unit == "seconds":
            self.target_samples = int(self.sample_rate * target_length)
        elif target_length_unit == "samples":
            self.target_samples = int(target_length)
        else:
            raise ValueError(
                'target_length_unit must be "samples" or "seconds"'
            )

    def forward(self, waveform, sample_rate: Optional[int] = None):
        sample_rate = sample_rate or self.sample_rate

        if sample_rate is None:
            raise RuntimeError("sample_rate is required")
        if waveform.shape[-1] < self.target_samples:
            # If the waveform is shorter than the target length, pad it
            pad_total = self.target_samples - waveform.shape[-1]
            # TODO determinism (grab seed)
            pad_left = random.randint(0, pad_total)
            pad_right = pad_total - pad_left
            padding_left = torch.zeros(
                waveform.shape[0], waveform.shape[1], pad_left
            )
            padding_right = torch.zeros(
                waveform.shape[0], waveform.shape[1], pad_right
            )
            waveform = torch.cat(
                (padding_left, waveform, padding_right), dim=-1
            )
            return waveform
        else:
            return waveform


class PadCrop(Pad):
    """Pad or crop an audio signal to a fixed length."""

    def forward(self, waveform, sample_rate: Optional[int] = None):
        # Use the forward method from the Pad class to pad the waveform
        waveform = super().forward(waveform, sample_rate)
        # If the waveform is longer than the target length, crop it
        if waveform.shape[-1] > self.target_samples:
            waveform = waveform[..., : self.target_samples]
        return waveform


class CenterCrop(nn.Module):
    "Center crop an audio signal to a fixed length"

    def __init__(
        self,
        target_length: float,
        sample_rate: int,
        target_length_unit: str = "seconds",
    ):

        super().__init__()
        self.sample_rate = sample_rate
        if target_length_unit == "seconds":
            self.target_samples = int(self.sample_rate * target_length)
        elif target_length_unit == "samples":
            self.target_samples = int(target_length)
        else:
            raise ValueError(
                'target_length_unit must be "samples" or "seconds"'
            )

    def forward(
        self, waveform: torch.Tensor, sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        sample_rate = sample_rate or self.sample_rate

        if sample_rate is None:
            raise RuntimeError("sample_rate is required")

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # handle batch of 1

        num_samples = waveform.shape[-1]

        if num_samples == self.target_samples:
            pass
        elif num_samples > self.target_samples:
            start = (num_samples - self.target_samples) // 2
            end = start + self.target_samples
            waveform = waveform[..., start:end]
        else:
            raise UserWarning(
                "Waveform shorter than target length provided. Make sure waveform is padded before cropping."
            )

        # unsqueeze
        if waveform.dim() == 2:
            return waveform.unsqueeze(0)
        else:
            return waveform


class ResizeSpectrogram(nn.Module):
    """Resize the spectrogram to a predefined size."""

    supports_multichannel = True

    def __init__(
        self,
        size: typing.List,
        mode: str = "bilinear",
        align_corners: bool = False,
    ):
        """
        :param size: size to which spectrograms are to be resized.
        :param mode: the interpolation mode to use. Default is 'bilinear'.
        :param align_corners: if True, the input and output tensors are aligned by the corners of their cells,
                              else they are aligned by the center of their cells. Default is False.
        """
        super().__init__()
        self.size = tuple(
            size
        )  # evil hack to bypass yaml not understanding tuples
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, spectrograms):
        if len(spectrograms) == 0:
            warnings.warn(
                "An empty spectrograms tensor was passed to {}".format(
                    self.__class__.__name__
                )
            )
            return spectrograms

        if len(spectrograms.shape) != 4:

            raise RuntimeError(
                "expects input tensors to be four-dimensional, with"
                " dimension ordering like [batch_size, num_channels, height, width]."
            )

        if is_multichannel(spectrograms):
            if spectrograms.shape[1] > min(
                spectrograms.shape[2], spectrograms.shape[3]
            ):
                warnings.warn(
                    "Multichannel audio must have channels first, not channels last. In"
                    " other words, the shape must be (batch size, channels, height, width), not"
                    " (batch_size, height, width, channels)"
                )
            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException(
                    "{} only supports mono audio, not multichannel audio".format(
                        self.__class__.__name__
                    )
                )

        resized_spectrograms = F.interpolate(
            spectrograms,
            size=self.size,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return resized_spectrograms


# TODO replace with torchaudio implementation once it is released
# (https://github.com/iver56/torch-audiomentations/tree/main?tab=readme-ov-file#unreleased)
class RandomCrop(nn.Module):
    """Crop the audio to predefined length in max_length.
    Adapted from torch_audiomentations.augmentations.random_crop.RandomCrop
    to accommodate the case of the input having the target length and thus not needing to be cropped.
    """

    supports_multichannel = True

    def __init__(
        self,
        max_length: float,
        sample_rate: int,
        max_length_unit: str = "seconds",
    ):
        """
        :param max_length: length to which samples are to be cropped.
        :sample_rate: sampling rate of input samples.
        :max_length_unit: defines the unit of max_length.
            "seconds": Number of seconds
            "samples": Number of audio samples
        """
        super().__init__()
        self.sample_rate = sample_rate
        if max_length_unit == "seconds":
            self.num_samples = int(self.sample_rate * max_length)
        elif max_length_unit == "samples":
            self.num_samples = int(max_length)
        else:
            raise ValueError('max_length_unit must be "samples" or "seconds"')

    def forward(self, samples, sample_rate: typing.Optional[int] = None):

        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None:
            raise RuntimeError("sample_rate is required")

        if len(samples) == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(
                    self.__class__.__name__
                )
            )
            return samples

        if len(samples.shape) != 3:
            raise RuntimeError(
                "torch-audiomentations expects input tensors to be three-dimensional, with"
                " dimension ordering like [batch_size, num_channels, num_samples]. If your"
                " audio is mono, you can use a shape like [batch_size, 1, num_samples]."
            )

        if is_multichannel(samples):
            if samples.shape[1] > samples.shape[2]:
                warnings.warn(
                    "Multichannel audio must have channels first, not channels last. In"
                    " other words, the shape must be (batch size, channels, samples), not"
                    " (batch_size, samples, channels)"
                )
            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException(
                    "{} only supports mono audio, not multichannel audio".format(
                        self.__class__.__name__
                    )
                )

        if samples.shape[2] < self.num_samples:
            warnings.warn("audio length less than cropping length")
            return samples

        if samples.shape[2] == self.num_samples:
            return samples  # Return the samples as-is if they match the cropping length

        start_indices = torch.randint(
            0, samples.shape[2] - self.num_samples, (samples.shape[2],)
        )
        samples_cropped = torch.empty(
            (samples.shape[0], samples.shape[1], self.num_samples),
            device=samples.device,
        )
        for i, sample in enumerate(samples):

            samples_cropped[i] = sample.unsqueeze(0)[
                :, :, start_indices[i] : start_indices[i] + self.num_samples
            ]

        return samples_cropped


class NoiseInjection(nn.Module):
    def __init__(self, p=0.5, max_noise_level=0.5):
        """
        Noise injection augmentation to be applied on waveforms.

        Based and modified from
        https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet/blob/useIS459/src/custom/utils.py
        under MIT license.

        Parameters
        ----------
        always_apply: bool. Can be turned to True for debugging purposes.
        p: float between 0-1. Probability to apply the augmentation.
        max_noise_level: float. Maximum noise level of the injection.
        """
        super().__init__()
        self.noise_level = (0.0, max_noise_level)

    def forward(self, samples):
        noise_level = torch.distributions.Uniform(*self.noise_level).sample([1])
        n_samples = samples.shape[-1]
        noise = torch.randn(n_samples)
        augmented = samples + noise * noise_level
        return augmented


COLOR_DECAYS_TO_F = {
    "white": 0.0,
    "pink": 1.0,
    "brown": 2.0,
    "blue": -1.0,
    "violet": -2.0,
    "NoiseInjection": "",
}


class RandomColoredNoise(nn.Module):
    """Randomly apply white, pink, or brown noise using AddColoredNoise,
    based on f_decay values.
    This is to mimic noise additions done in the IS459 paper (InsectEffNet)
    Will only work on fixed sample rate datasets"""

    def __init__(
        self,
        sample_rate: float,
        p: float,
        min_snr_in_db: float,
        max_snr_in_db: float,
        max_noise_injection: float,
        colors: list[str],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.p = p
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.max_noise_injection = max_noise_injection
        self.colors = colors
        self.transforms = []

        # check if the selected color strings are valid
        if not set(self.colors).issubset(COLOR_DECAYS_TO_F.keys()):
            raise ValueError(
                f"colors must be one of {COLOR_DECAYS_TO_F.keys()}"
            )

        # instantiate transform objects as list items
        for color in colors:
            if color == "NoiseInjection":
                self.transforms.append(
                    NoiseInjection(
                        max_noise_level=self.max_noise_injection, p=1
                    )
                )
            else:
                f_decay = COLOR_DECAYS_TO_F[color]
                self.transforms.append(
                    AddColoredNoise(
                        sample_rate=self.sample_rate,
                        min_snr_in_db=self.min_snr_in_db,
                        max_snr_in_db=self.max_snr_in_db,
                        min_f_decay=f_decay,
                        max_f_decay=f_decay,
                        p=1.0,
                        output_type="tensor",
                    )
                )

    def forward(self, samples):
        # this will emulate a OneOf transform selection
        if random.random() > self.p:
            return samples  # skip transform with probability 1-p
        # choose color, apply and return
        return random.choice(self.transforms)(samples)


class WaveToLogMel(nn.Module):
    """
    Input:
      - [B, C, T] (C must be 1) or [B, T] waveform, float32, normalized
    Output (PANN-compatible):
      - [B, 1, time_steps, mel_bins]
    """

    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
    ):
        super().__init__()
        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pad_mode = pad_mode

        # code from PANN
        # Spectrogram extractor
        self.spectrogram_extractor = LibrosaSpectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, data_length)"""
        # Expect [B, C, T] (C==1) or [B, T]
        x = x.squeeze(1)
        x = self.spectrogram_extractor(
            x
        )  # (batch_size, 1, time_steps, freq_bins)
        # (batch_size, 1, time_steps, mel_bins) --> should work for effnet
        x = self.logmel_extractor(x)

        return x


class AudioDataset(Dataset):
    """
    Audio dataset. "A map-style dataset is one that implements the __getitem__() and __len__() protocols"
    """

    def __init__(self, df, data_path=None, transform=None):
        self.df = df
        self.data_path = data_path
        self.transform = transform

        # do some initial checks
        assert (
            "fpath_audio" in self.df.columns
        ), "expected file paths in column 'fpath_audio'"
        assert (
            self.df["fpath_audio"].apply(os.path.isfile).all()
        ), "Not all files found!"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fpath_audio = None
        try:
            fpath_audio = self.df.loc[idx, "fpath_audio"]

            # returns a float32 and SR in ints
            audio, sample_rate = torchaudio.load(fpath_audio)

            # adapt dimensions
            if audio.shape[0] > 1:
                # convert to mono
                audio = torch.mean(audio, dim=0, keepdim=True)

            if (
                len(audio.shape) == 2
            ):  # if audio is mono (1 channel)  ->  shape = (1, num_samples)
                audio = audio.unsqueeze(1)

            if self.transform:
                transformed = self.transform(audio, sample_rate=sample_rate)
                audio = transformed["samples"]

            label = self.df.loc[idx, "class"]
            label = torch.tensor(label).long()

            sample = {"audio": audio, "label": label}

            return sample

        except Exception as e:
            logging.warning(
                f"Warning: AudioDataset failed to load {fpath_audio} ({e})"
            )

    def get_file_df(self):
        return self.df


class FixedCropDataset(Dataset):
    def __init__(
        self,
        df_files: pd.DataFrame,
        crop_length_s: float = 5.0,
        overlap_s: float = 2.5,
        min_crop_duration_s: float = 0.5,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        # fpath, start_sample, num_samples, label
        self.entries = []
        self.df_files = df_files
        self.crop_length_s = crop_length_s

        # do some initial checks
        assert (
            "fpath_audio" in self.df_files.columns
        ), "expected file paths in column 'fpath_audio'"
        assert (
            self.df_files["fpath_audio"].apply(os.path.isfile).all()
        ), "Not all files found!"

        for file_idx, row in df_files.iterrows():
            sr = row["sample_rate"]
            total_samples = row["num_frames"]
            crop_samples = int(crop_length_s * sr)
            step = int((crop_length_s - overlap_s) * sr)
            fpath = row["fpath_audio"]
            label = row["class"]

            if total_samples < crop_samples:
                # pad-only case: include regardless of min_crop_duration_s
                # (minimum file durations are handled in dataset filtering)
                self.entries.append((file_idx, fpath, 0, crop_samples, label))
            else:
                # regular slicing
                for start in range(0, total_samples - crop_samples + 1, step):
                    self.entries.append(
                        (file_idx, fpath, start, crop_samples, label)
                    )

                # handle tail segment
                # ensures we do not include tiny, mostly silent segments at the end
                tail_start = total_samples - crop_samples
                # self.entries[-1][2] will get the start time of the previous (sliced) entry
                tail_duration_s = (
                    total_samples - (self.entries[-1][2] + crop_samples)
                ) / sr

                if tail_duration_s >= min_crop_duration_s:
                    self.entries.append(
                        (file_idx, fpath, tail_start, crop_samples, label)
                    )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        file_idx, fpath, start_sample, num_samples, label = self.entries[idx]

        # (torch.Tensor, int): Resulting Tensor and sample rate.
        audio, sr = torchaudio.load(
            fpath, frame_offset=start_sample, num_frames=num_samples
        )

        # adapt dimensions
        if audio.shape[0] > 1:
            # convert to mono
            audio = torch.mean(audio, dim=0, keepdim=True)

        if audio.dim() == 1:
            # [T] -> [B=1, C=1, T]
            audio = audio.unsqueeze(0).unsqueeze(0)

        elif audio.dim() == 2:
            # [C, T]  (e.g., stereo)
            if audio.size(0) > 1:
                # to mono
                audio = audio.mean(dim=0, keepdim=False)  # [T]
                audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            else:
                # already mono [1, T] -> add batch
                audio = audio.unsqueeze(0)  # [1, 1, T]

        elif audio.dim() == 3:
            # already [B, C, T]; keep as-is
            pass
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")

        # apply transformations
        if self.transform:
            transformed = self.transform(audio)
            audio = transformed["samples"]

        # check duration
        assert audio.size(-1) <= sr * self.crop_length_s, "Audio too long!"

        # add filename to sample to calculate per-file pooled metrics later
        sample = {"audio": audio, "label": label, "fpath_audio": fpath}

        return sample

    def get_file_df(self):
        # TODO also get start & crop_samples into the df
        return self.df_files.iloc[[file_idx for file_idx, *_ in self.entries]]


# new design to allow application of MixUp
def make_collate_fn(mixup_alpha=None, num_classes=None):
    mix_transform = None

    # MixUp will error if alpha is 0 or not float
    if mixup_alpha != 0 and mixup_alpha is not None:
        mix_transform = MixUp(alpha=float(mixup_alpha), num_classes=num_classes)
        logging.info(
            f"Creating collate_fn with MixUp transform (with alpha: {mixup_alpha})"
        )

    def collate_fn_new(batch):
        # Separate the audio and labels
        audio = [item["audio"].squeeze(1) for item in batch]  # List of [F, T]
        labels = [item["label"] for item in batch]  # List of ints
        fpaths = [item["fpath_audio"] for item in batch]  # List of strings

        audio = torch.stack(audio)  # Tensor [B, F, T]
        labels = torch.tensor(labels, dtype=torch.long)  # Tensor [B]

        # collate_fn_new will see enclosing scope, including mix_transform
        if mix_transform is not None:
            audio, labels = mix_transform(audio, labels)

        return {"audio": audio, "label": labels, "fpath_audio": fpaths}

    return collate_fn_new


# dictionary with transformations
TRANSFORMATIONS = {
    "AddColoredNoise": AddColoredNoise,
    "RandomColoredNoise": RandomColoredNoise,
    "Gain": Gain,
    "HighPassFilter": HighPassFilter,
    "LowPassFilter": LowPassFilter,
    "Pad": Pad,
    "PadCrop": PadCrop,
    "RandomCrop": RandomCrop,
    "Resample": Resample,
    "ResizeSpectrogram": ResizeSpectrogram,
    "Spectrogram": Spectrogram,
    "MelScale": MelScale,
    "MelSpectrogram": MelSpectrogram,
    "AmplitudeToDB": AmplitudeToDB,
    "SpecAugmentation": SpecAugmentation,
    "RandomSpecAugment": RandomSpecAugment,
    "ApplyImpulseResponse": ApplyImpulseResponse,
    "RandomPad": RandomPad,
    "CenterCrop": CenterCrop,
    "AddBackgroundNoise": AddBackgroundNoise,
    "LogmelFilterBank": LogmelFilterBank,
    "LibrosaSpectrogram": LibrosaSpectrogram,
    "WaveToLogMel": WaveToLogMel,
    "TimeStretch": TimeStretchTorch,
    # Add all other transform classes here
}


def resolve_ir_paths(config):
    # save path resolving of IR files
    if "ApplyImpulseResponse" in config.get("transforms").keys():
        fpaths = config["transforms"]["ApplyImpulseResponse"].get(
            "ir_paths", None
        )
        # sorted(): glob() returns filesystem enumeration order, and
        # ApplyImpulseResponse draws indices into this list with the seeded RNG. An
        # unsorted list therefore picks different IRs on different filesystems at the
        # same seed.
        res_fpaths = sorted(glob(fpaths, recursive=True))
        # TODO add this to documentation
        assert (
            len(res_fpaths) > 0
        ), f"No impulse response files found at {fpaths}"
        # add paths back into config
        config["transforms"]["ApplyImpulseResponse"]["ir_paths"] = res_fpaths

    return config


def resolve_env_sound_paths(config, mode, df):
    # df is needed to get environment sound paths
    # check if the keyword is there
    if "AddBackgroundNoise" in config.get("transforms").keys():
        # in case a path to csv holding the environment sounds is passed
        if "annotation_csv_path" in config["transforms"].get(
            "AddBackgroundNoise"
        ):
            # remove it from the config
            csv_fpath = config["transforms"]["AddBackgroundNoise"].pop(
                "annotation_csv_path"
            )
            df_envsounds = pd.read_csv(csv_fpath)
            # check if there are other classes than "envsound", log & drop if that's the case
            # this way, we can just provide the resampled annnotation csv path
            classes = df_envsounds["class_name"].unique()
            assert "envsound" in list(
                classes
            ), "No 'envsound' classes provided in transform config (AddBackgroundNoise: annotation_csv_path)"
            if len(classes) > 1:
                logging.info(
                    "There are other classes than 'envsound' in AddBackgroundNoise: annotation_csv_path. Will drop."
                )
                df_envsounds = df_envsounds[
                    df_envsounds["class_name"] == "envsound"
                ].copy()

        # no annotation_csv_path: a df must be passed in via the create_transforms function
        else:
            if df is None:
                raise ValueError(
                    "Must pass a dataframe if no 'annotation_csv_path' is specified under AddBackgroundNoise in the config"
                )
            # only keep environment sounds
            df_envsounds = df[df["class_name"] == "envsound"].copy()

        use_split = config["transforms"]["AddBackgroundNoise"].pop("use_split")
        # filter for use_split
        if use_split == True:
            # map subset names
            split_names = {
                "train": "train",
                "valid": "validation",
                "test": "test",
                "validation": "validation",
            }
            split_name = split_names[mode]
            # keep only files from the current split
            df_envsounds = df_envsounds[df_envsounds["subset"] == split_name]

        # add the paths into the config
        config["transforms"]["AddBackgroundNoise"]["background_paths"] = (
            df_envsounds["fpath_audio"].tolist()
        )

    return config


def apply_flat_overrides(config, overrides):
    """Recursively apply flat dot-style keys like 'Gain.min_gain_in_db'."""
    for k, v in overrides.items():
        keys = k.split(".")
        d = config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return config


def replace_spectrogram_type(transform_order, spec_type):
    def replace(t):
        if isinstance(t, str):
            return spec_type if t == "SpectrogramType" else t
        elif isinstance(t, dict):  # e.g., SomeOf or OneOf
            key = next(iter(t))
            val = t[key]
            if isinstance(val, dict) and "transforms" in val:
                val["transforms"] = [
                    replace(sub_t) for sub_t in val["transforms"]
                ]
            return {key: val}
        else:
            return t  # fallback, shouldn't happen

    return [replace(t) for t in transform_order]


def safe_tuple(list_or_int):
    if isinstance(list_or_int, list):
        return tuple(list_or_int)
    elif isinstance(list_or_int, int):
        return list_or_int
    else:
        raise ValueError("num_transforms must be of type list or int.")


def instantiate_transform(name, params):
    trsf_class = TRANSFORMATIONS[name]
    params_str = str(params)

    # truncate string (config will contain all paths to env sound files and IR files)
    if len(params_str) > 250:
        params_str = params_str[:250] + "..."
    logging.info(f"    {name}: {params_str}")
    return trsf_class(**params)


def create_transforms(
    config: dict,
    resizing_size: tuple,
    mode: str,
    sweep_overrides: dict = None,
    df: pd.DataFrame = None,
    p_override: float = None,  # can be used to override p values!= 1.0 globally
) -> Compose:
    """
    New function to allow for more intricate transform pipelines
    """
    if mode == "test":
        assert (
            sweep_overrides is None
        ), "Do not pass sweep_overrides to create_transforms in mode 'test'"
        assert (
            df is None
        ), "Do not pass a df to create_transforms in mode 'test'"
        assert (
            p_override is None
        ), "Do not pass p_override to create_transforms in mode 'test'"

    config = deepcopy(config)

    # resolve paths to impulse response files (provided as glob pattern in config)
    if "ApplyImpulseResponse" in config["order"]["train"]:
        config = resolve_ir_paths(config)

    # add paths to environment sound files into the config
    if "AddBackgroundNoise" in config["order"]["train"]:
        config = resolve_env_sound_paths(config, mode, df)

    if sweep_overrides:
        logging.info(
            "Creating transforms for mode %s, with sweep overrides %s",
            mode,
            sweep_overrides,
        )

    if p_override is not None:
        logging.info(
            "Overriding probabilities for augmentations with p != 1.0 to %s",
            p_override,
        )

    transforms_config = deepcopy(config.get("transforms", {}))
    mode_order = config.get("order", {}).get(mode, [])

    def parse_transform_spec(name_or_dict):
        # Basic transform name as string
        if isinstance(name_or_dict, str):
            params = transforms_config.get(name_or_dict, {})

            # Apply p_override if specified and current p != 1.0
            if p_override is not None and "p" in params.keys():
                original_p = params["p"]
                if original_p != 1.0:
                    logging.info(
                        "Overriding %s probability from %s to %s",
                        name_or_dict,
                        original_p,
                        p_override,
                    )
                    params["p"] = p_override

            # modify p values based on waveform_aug_scale.p from sweep
            if sweep_overrides is not None:
                if (
                    "waveform_aug_scale" in sweep_overrides
                    and "p" in params.keys()
                ):
                    new_p = params["p"] * sweep_overrides["waveform_aug_scale"]
                    assert (
                        0 <= new_p <= 1
                    ), f"p Must be between 0 and 1 ({name_or_dict}) ({params['p']=})"
                    logging.info(
                        "Overriding %s with new_p = %s", name_or_dict, new_p
                    )
                    params["p"] = new_p

                # override RandomSpecAugment.p from sweep
                if (
                    "spec_aug_p" in sweep_overrides
                    and name_or_dict == "RandomSpecAugment"
                ):
                    new_p = sweep_overrides["spec_aug_p"]

                    logging.info(
                        "Overriding RandomSpecAugment with new_p = %s", new_p
                    )

                    params["p"] = new_p

            return instantiate_transform(name_or_dict, params)

        # Dict = OneOf or SomeOf
        elif isinstance(name_or_dict, dict):
            if "SomeOf" in name_or_dict:
                logging.info("SomeOf(")
                cfg = name_or_dict["SomeOf"]
                transforms = [
                    parse_transform_spec(t) for t in cfg["transforms"]
                ]
                logging.info(")")

                p = cfg.get("p", 1.0)
                # Apply p_override if specified and current p != 1.0
                original_p = p
                if p_override is not None and p != 1.0:
                    p = p_override
                    logging.info(
                        "Overriding SomeOf p from %s to %s", original_p, p
                    )

                # modify SomeOf.p value based on waveform_aug_scale from sweep
                if (
                    sweep_overrides is not None
                    and "waveform_aug_scale" in sweep_overrides
                ):
                    p *= sweep_overrides["waveform_aug_scale"]
                    assert (
                        0 <= p <= 1
                    ), f"p Must be between 0 and 1 ({name_or_dict}) ({params['p']=})"
                    logging.info("Overriding %s with p = %s", name_or_dict, p)

                return SomeOf(
                    num_transforms=safe_tuple(cfg["num_transforms"]),
                    transforms=transforms,
                    p=p,
                    output_type="dict",
                )

            elif "OneOf" in name_or_dict:
                logging.info("OneOf(")
                cfg = name_or_dict["OneOf"]
                transforms = [
                    parse_transform_spec(t) for t in cfg["transforms"]
                ]
                logging.info(")")

                p = cfg.get("p", 1.0)

                # Apply p_override if specified and current p != 1.0
                original_p = p
                if p_override is not None and p != 1.0:
                    p = p_override
                    logging.info(
                        "Overriding OneOf p from %s to %s", original_p, p
                    )

                # modify SomeOf.p value based on waveform_aug_scale from sweep
                if (
                    sweep_overrides is not None
                    and "waveform_aug_scale" in sweep_overrides
                ):
                    p *= sweep_overrides["waveform_aug_scale"]
                    assert (
                        0 <= p <= 1
                    ), f"p Must be between 0 and 1 ({name_or_dict}) ({params['p']=})"
                    logging.info("Overriding %s with p = %s", name_or_dict, p)

                return OneOf(transforms=transforms, p=p, output_type="dict")

            else:
                raise ValueError(f"Unrecognized transform dict: {name_or_dict}")
        else:
            raise TypeError(f"Invalid transform format: {name_or_dict}")

    transforms = [parse_transform_spec(entry) for entry in mode_order]

    logging.info("Transformations for %s mode: %s", mode, transforms)

    return Compose(transforms=transforms, output_type="dict")


def verify_compose(transform_group: Compose, parent_p=1.0, prefix=""):
    """
    Recursively analyse a given torch_audiomentations.Compose object to verify
    it is in line with configs and assumptions.

    Returns:
        total_p (float): Expected probability of a transform being applied
    """
    total_p = 0.0
    if isinstance(transform_group, OneOf):
        group_p = getattr(transform_group, "p", 1.0)
        num_choices = len(transform_group.transforms)
        effective_p = parent_p * group_p
        selection_p = 1.0 / num_choices
        # print(f"OneOf {parent_p=} {group_p=} {num_choices=} {effective_p=}")

        for i, transform in enumerate(transform_group.transforms):
            # Uniform sampling assumed
            # print(f"{selection_p=}")
            total_p += verify_compose(
                transform,
                parent_p=effective_p * selection_p,
                prefix=f"{prefix}{i}-",
            )

    elif isinstance(transform_group, SomeOf):
        group_p = getattr(transform_group, "p", 1.0)

        # returns number of transforms that can be selected
        num_choices = len(transform_group.transforms)
        n_transforms = transform_group.num_transforms
        if isinstance(n_transforms, tuple):
            expected_num_transforms = sum(n_transforms) / 2
        else:
            expected_num_transforms = n_transforms

        selection_p = expected_num_transforms / num_choices

        effective_p = parent_p * group_p
        # print(f"Compose/SomeOf {parent_p=} {group_p=} {effective_p=}")
        for i, transform in enumerate(transform_group.transforms):
            total_p += verify_compose(
                transform,
                parent_p=effective_p * selection_p,
                prefix=f"{prefix}{i}-",
            )

    elif isinstance(transform_group, Compose):
        group_p = getattr(transform_group, "p", 1.0)
        effective_p = parent_p * group_p

        for i, transform in enumerate(transform_group.transforms):
            total_p += verify_compose(
                transform, parent_p=effective_p, prefix=f"{prefix}{i}-"
            )

    else:
        # Assume it's a leaf transform
        # print("Leaf")
        transform_name = type(transform_group).__name__
        transform_p = getattr(transform_group, "p", 1.0)
        p_mode = getattr(transform_group, "p_mode", "per_example")
        if p_mode == "per_example":
            final_p = parent_p * transform_p
        else:
            final_p = parent_p

        logging.info(f"{prefix}{transform_name}: Expected P = {final_p:.4f}")
        return final_p
    return round(total_p, 2)
