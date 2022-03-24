import numpy as np
import torch
import torch.nn as nn


def spec_conv1d(n_layer=3, n_channel=[64, 32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2]):
    """
    Construction of conv. layers. Note the current implementation always effectively turn to 1-D conv,
    inspired by https://arxiv.org/pdf/1704.04222.pdf.
    :param n_layer: number of conv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 ).
            The first channel is the number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :return: an object (nn.Sequential) constructed of specified conv. layers
    TODO:
        [x] directly use nn.Conv1d for implementation
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg

    # construct layers
    conv_layers = []
    for i in range(n_layer):
        in_channel, out_channel = n_channel[i:i + 2]
        conv_layers += [
            nn.Conv1d(in_channel, out_channel, filter_size[i], stride[i]),
            nn.BatchNorm1d(out_channel),
            nn.Tanh()
        ]

    return nn.Sequential(*conv_layers)


def spec_deconv1d(n_layer=3, n_channel=[64, 32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2]):
    """
    Construction of deconv. layers. Input the arguments in normal conv. order.
    E.g., n_channel = [1, 32, 16, 8] gives deconv. layers of [8, 16, 32, 1].
    :param n_layer: number of deconv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 )
            The first channel is the number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :return: an object (nn.Sequential) constructed of specified deconv. layers.
    TODO:
        [x] directly use nn.Conv1d for implementation
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg

    n_channel, filter_size, stride = n_channel[::-1], filter_size[::-1], stride[::-1]

    deconv_layers = []
    for i in range(n_layer - 1):
        in_channel, out_channel = n_channel[i:i + 2]
        deconv_layers += [
            nn.ConvTranspose1d(in_channel, out_channel, filter_size[i], stride[i]),
            nn.BatchNorm1d(out_channel),
            nn.Tanh()
        ]

    # Construct the output layer
    deconv_layers += [
        nn.ConvTranspose1d(n_channel[-2], n_channel[-1], filter_size[-1], stride[-1]),
        nn.Tanh()  # check the effect of with or without BatchNorm in this layer
    ]

    return nn.Sequential(*deconv_layers)


def fc(n_layer, n_channel, activation='tanh', batchNorm=True):
    """
    Construction of fc. layers.
    :param n_layer: number of fc. layers
    :param n_channel: in/output number of neurons for each layer ( len(n_channel) = n_layer + 1 )
    :param activation: allow either 'tanh' or None for now
    :param batchNorm: True|False, indicate apply batch normalization or not
    TODO:
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    assert activation in [None, 'tanh'], "Only implement 'tanh' for now"

    fc_layers = []
    for i in range(n_layer):
        layer = [nn.Linear(n_channel[i], n_channel[i + 1])]
        if batchNorm:
            layer.append(nn.BatchNorm1d(n_channel[i + 1]))
        if activation:
            layer.append(nn.Tanh())
        fc_layers += layer

    return nn.Sequential(*fc_layers)