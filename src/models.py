import warnings
import logging
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

# PANN models
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

# Pre-trained image models
import timm

###########################################################
"""
PANN models
Code from Kong et al. 2019
https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/finetune_template.py
Adapted by Tobias Peter & Jaromir Kunzelmann
"""
###########################################################


def move_data_to_device(x, device):
    if "float" in str(x.dtype):
        x = torch.Tensor(x)
    elif "int" in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (
        x[0::2].transpose(0, -1) * mixup_lambda[0::2]
        + x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    ).transpose(0, -1)
    return out


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)  # np.atleast_1d(value)
    else:
        dict[key] = [value]


def forward(model, generator, return_input=False, return_target=False):
    """Forward data to a model.

    Args:
      model: object
      generator: object
      return_input: bool
      return_target: bool
    Returns:
      audio_name: (audios_num,)
      clipwise_output: (audios_num, num_classes)
      (ifexist) segmentwise_output: (audios_num, segments_num, num_classes)
      (ifexist) framewise_output: (audios_num, frames_num, num_classes)
      (optional) return_input: (audios_num, segment_samples)
      (optional) return_target: (audios_num, num_classes)
    """
    output_dict = {}
    device = next(model.parameters()).device

    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        batch_waveform = move_data_to_device(
            batch_data_dict["waveform"], device
        )

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)

        append_to_dict(output_dict, "audio_name", batch_data_dict["audio_name"])

        append_to_dict(
            output_dict,
            "clipwise_output",
            batch_output["clipwise_output"].data.cpu().numpy(),
        )

        if return_input:
            append_to_dict(output_dict, "waveform", batch_data_dict["waveform"])

        if return_target:
            if "target" in batch_data_dict.keys():
                append_to_dict(output_dict, "target", batch_data_dict["target"])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, num_classes)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, num_classes)
    """
    batch_size, time_steps, num_classes = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, num_classes)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, num_classes)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, num_classes)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, num_classes)"""

    return output


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class Cnn14(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        num_classes,
    ):

        super(Cnn14, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pad_mode = pad_mode

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
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

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        input = input.squeeze(1)
        x = self.spectrogram_extractor(
            input
        )  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {"pred": clipwise_output, "embedding": embedding}

        return output_dict


class Cnn10(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        num_classes,
    ):

        super(Cnn10, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pad_mode = pad_mode

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
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

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        input = input.squeeze(1)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {"pred": clipwise_output, "embedding": embedding}

        return output_dict


class Cnn6(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        num_classes,
    ):

        super(Cnn6, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pad_mode = pad_mode

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
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

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        input = input.squeeze(1)
        x = self.spectrogram_extractor(
            input
        )  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {"pred": clipwise_output, "embedding": embedding}

        return output_dict


class Transfer_Cnn14(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        num_classes,
        freeze_base,
    ):
        """Classifier for a new task using pretrained Cnn14 as a sub module."""
        super(Transfer_Cnn14, self).__init__()
        audioset_num_classes = 527

        self.base = Cnn14(
            sample_rate,
            window_size,
            hop_size,
            mel_bins,
            fmin,
            fmax,
            audioset_num_classes,
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, num_classes, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path, device="cpu"):
        checkpoint = torch.load(
            pretrained_checkpoint_path, map_location=torch.device(device)
        )
        self.base.load_state_dict(checkpoint["model"])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict["embedding"]

        # Refactor (apapting from multi-label to multi-class)
        logits = self.fc_transfer(embedding)
        log_softmax = torch.log_softmax(logits, dim=-1)
        # clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict["pred"] = logits

        return output_dict["pred"]
        # return output_dict


class Transfer_Cnn10(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        num_classes,
        freeze_base,
    ):
        """Classifier for a new task using pretrained Cnn14 as a sub module."""
        super(Transfer_Cnn10, self).__init__()
        audioset_num_classes = 527

        self.base = Cnn10(
            sample_rate,
            window_size,
            hop_size,
            mel_bins,
            fmin,
            fmax,
            audioset_num_classes,
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, num_classes, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path, device="cpu"):
        checkpoint = torch.load(
            pretrained_checkpoint_path, map_location=torch.device(device)
        )
        self.base.load_state_dict(checkpoint["model"])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict["embedding"]

        # Refactor (apapting from multi-label to multi-class)
        logits = self.fc_transfer(embedding)
        log_softmax = torch.log_softmax(logits, dim=-1)
        # clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict["pred"] = logits

        return output_dict["pred"]


class Transfer_Cnn6(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        num_classes,
        freeze_base,
    ):
        """Classifier for a new task using pretrained Cnn14 as a sub module."""
        super(Transfer_Cnn6, self).__init__()
        audioset_num_classes = 527

        self.base = Cnn6(
            sample_rate,
            window_size,
            hop_size,
            mel_bins,
            fmin,
            fmax,
            audioset_num_classes,
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, num_classes, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path, device="cpu"):
        checkpoint = torch.load(
            pretrained_checkpoint_path, map_location=torch.device(device)
        )
        self.base.load_state_dict(checkpoint["model"])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict["embedding"]

        # Refactor (apapting from multi-label to multi-class)
        logits = self.fc_transfer(embedding)
        log_softmax = torch.log_softmax(logits, dim=-1)
        # clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict["pred"] = logits

        return output_dict["pred"]
        # return output_dict


###########################################################
# Load models
###########################################################


def load_pann_model(config, config_trsf, freeze_base, num_classes, pretrain):
    """Load the PANN model with specified configurations."""

    model_name = config["model"]["model_name"]
    pann_config = config_trsf["pann"]

    # Dictionary to map model_type to the corresponding class
    model_classes = {
        "Cnn14": Transfer_Cnn14,
        "Cnn10": Transfer_Cnn10,
        "Cnn6": Transfer_Cnn6,
    }

    # Select the model class based on model_type
    if model_name not in model_classes:
        raise ValueError(f"Unsupported model_type: {model_name}")

    model_class = model_classes[model_name]
    model = model_class(
        sample_rate=pann_config["sample_rate"],
        window_size=pann_config["window_size"],
        hop_size=pann_config["hop_size"],
        mel_bins=pann_config["mel_bins"],
        fmin=pann_config["fmin"],
        fmax=pann_config["fmax"],
        num_classes=num_classes,
        freeze_base=freeze_base,
    )

    if pretrain == "default":
        pretrained_model = config["model"]["pann_weights"]
        model.load_from_pretrain(pretrained_checkpoint_path=pretrained_model)
        logging.info(
            "Loaded PANN %s model with default pre-trained weights from %s",
            model_name,
            pretrained_model,
        )
    elif pretrain == "custom":
        pretrained_model = config["model"]["custom_weights"]
        model.load_from_pretrain(pretrained_checkpoint_path=pretrained_model)
        logging.info(
            "Loaded PANN %s model with custom pre-trained weights from %s",
            model_name,
            pretrained_model,
        )
    else:
        logging.info(
            "Loaded PANN %s model without pre-trained weights.", model_name
        )

    # log trainable params
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {trainable_params:_}/{total_params:_}")
    return model


def load_timm_model(
    config, num_classes, pretrain, freeze_base, head_name="classifier"
):
    """Load the TIMM model with specified configurations."""

    model_name = config["model"]["model_name"]
    # drop_rate and drop_path_rate are 0 by default, so we can either omit them in the config
    # or explicitely set them to 0 in the config
    drop_rate = config["model"].get("drop_rate", None)
    drop_path_rate = config["model"].get("drop_path_rate", None)

    model = timm.create_model(
        model_name,
        pretrained=(pretrain == "default"),
        in_chans=1,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if pretrain == "custom":
        pretrained_model = config["model"]["custom_weights"]
        # load custom weights
        model_cp = torch.load(pretrained_model)
        num_classes_cp = model_cp["num_classes"]
        model_state_dict = model_cp["model_state_dict"]
        logging.info(f"Loaded model checkpoint has {num_classes_cp} classes")

        if num_classes != num_classes_cp:
            logging.warning(
                f"... but initialized model has {num_classes} classes. Stripping head."
            )
            # Strip classifier head from the pretrained checkpoint
            prefix = head_name + "."
            model_state_dict = {
                k: v
                for k, v in model_state_dict.items()
                if not k.startswith(prefix)
            }

        # load state dict into the model
        keys_msg = model.load_state_dict(model_state_dict, strict=False)
        logging.info(
            "Loaded model %s with custom pre-trained weights from: %s, with drop_rate: %s, and drop_path_rate %s",
            model_name,
            pretrained_model,
            drop_rate,
            drop_path_rate,
        )
        if keys_msg.missing_keys or keys_msg.unexpected_keys:
            logging.warning(
                "Missing or unexpected keys during state_dict load: %s. Missing keys are expected if model is being adapted. ",
                keys_msg,
            )

    elif pretrain == "default":
        logging.info(
            "Loaded model %s with default pre-trained weights: drop_rate %s, and drop_path_rate %s",
            model_name,
            drop_rate,
            drop_path_rate,
        )
    else:
        logging.info(
            "Loaded model %s without pre-trained weights: drop_rate %s, and drop_path_rate %s",
            model_name,
            drop_rate,
            drop_path_rate,
        )

    # this has to be done _after_ loading a checkpoint
    if freeze_base:
        # First, freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # Then unfreeze just the classification head
        for p in model.classifier.parameters():
            p.requires_grad = True
        logging.info("Base parameters are frozen (fine-tuning)")

    # log trainable params
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params_classifier = sum(
        p.numel() for p in model.classifier.parameters() if p.requires_grad
    )
    total_params_classifier = sum(
        p.numel() for p in model.classifier.parameters()
    )
    logging.info(f"Trainable params: {trainable_params:_}/{total_params:_}")
    logging.info(
        f"Trainable params (classifier head): {trainable_params_classifier:_}/{total_params_classifier:_}"
    )

    return model


def initialize_model(
    config, config_trsf, model_source, num_classes, freeze_base, pretrain
):
    """Initialize the model based on the provided configuration and source."""
    logging.info("Initializing model from source %s.", model_source)

    # ignore annoying pytorch warnings in log level INFO
    logger = logging.getLogger()
    with warnings.catch_warnings():
        if logger.getEffectiveLevel() <= logging.INFO:
            warnings.simplefilter("ignore", category=FutureWarning)

        # load model
        if model_source == "pann":
            model = load_pann_model(
                config, config_trsf, freeze_base, num_classes, pretrain
            )
        elif model_source == "timm":
            model = load_timm_model(config, num_classes, pretrain, freeze_base)
        else:
            raise ValueError(f"Unknown model source: {model_source}.")

    return model
