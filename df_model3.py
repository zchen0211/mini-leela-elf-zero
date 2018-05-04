import os

import torch.nn as nn
import torch.distributed as dist

from model_base import Model


class Block(Model):
    def __init__(self, option_map, params):
        super().__init__(option_map, params)
        self.relu = nn.ReLU()
        self.conv_lower = self._conv_layer()
        self.conv_upper = self._conv_layer(relu=False)

    def _conv_layer(
            self,
            input_channel=None,
            output_channel=None,
            kernel=3,
            relu=True):
        if input_channel is None:
            input_channel = 224 # self.options.dim
        if output_channel is None:
            output_channel = 224 # self.options.dim

        layers = []
        layers.append(nn.Conv2d(
            input_channel,
            output_channel,
            kernel,
            padding=(kernel // 2),
        ))
        if True: # self.options.bn:
            layers.append(
                nn.BatchNorm2d(output_channel, track_running_stats=True))
        if relu:
            layers.append(self.relu)

        return nn.Sequential(*layers)

    def forward(self, s):
        s1 = self.conv_lower(s)
        s1 = self.conv_upper(s1)
        s1 = s1 + s
        s = self.relu(s1)
        return s


class GoResNet(Model):
    def __init__(self, option_map, params):
        super().__init__(option_map, params)
        self.blocks = []
        for _ in range(20): # self.options.num_block):
            self.blocks.append(Block(option_map, params))
        self.resnet = nn.Sequential(*self.blocks)

    def forward(self, s):
        return self.resnet(s)


class Model_PolicyValue(Model):
    def __init__(self, option_map, params):
        super().__init__(option_map, params)

        self.board_size = 19 # params["board_size"]
        self.num_future_actions = 362 # params["num_future_actions"]
        self.num_planes = 18 # params["num_planes"]

        # Network structure of AlphaGo Zero
        # https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html

        # Simple method. multiple conv layers.
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU(0.1) if self.options.leaky_relu else nn.ReLU()
        last_planes = 18 # self.num_planes

        self.init_conv = self._conv_layer(last_planes)

        self.pi_final_conv = self._conv_layer(224, 2, 1)
        self.value_final_conv = self._conv_layer(224, 1, 1)

        d = self.board_size ** 2

        # Plus 1 for pass.
        self.pi_linear = nn.Linear(d * 2, d + 1)
        self.value_linear1 = nn.Linear(d, 256)
        self.value_linear2 = nn.Linear(256, 1)

        # Softmax as the final layer
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.resnet = GoResNet(option_map, params)

    def _conv_layer(
            self,
            input_channel=None,
            output_channel=None,
            kernel=3,
            relu=True):
        if input_channel is None:
            input_channel = 224 # self.options.dim
        if output_channel is None:
            output_channel = 224 # self.options.dim

        layers = []
        layers.append(nn.Conv2d(
            input_channel,
            output_channel,
            kernel,
            padding=(kernel // 2)
        ))
        if True: # self.options.bn:
            layers.append(
                nn.BatchNorm2d(output_channel, track_running_stats=True))
        if relu:
            layers.append(self.relu)

        return nn.Sequential(*layers)

    def forward(self, x):
        s = self._var(x)
        # print(type(s), s.volatile)

        s = self.init_conv(s)
        # print("init conv", s)
        # print(s[0, 0, :, :])
        s = self.resnet(s)

        d = self.board_size ** 2

        pi = self.pi_final_conv(s)
        pi = self.pi_linear(pi.view(-1, d * 2))
        logpi = self.logsoftmax(pi)
        pi = logpi.exp()

        V = self.value_final_conv(s)
        V = self.relu(self.value_linear1(V.view(-1, d)))
        V = self.value_linear2(V)
        V = self.tanh(V)

        return dict(logpi=logpi, pi=pi, V=V)
