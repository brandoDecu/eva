# coding=utf-8
# Copyright 2018-2020 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch  # TODO: remove me
import pandas as pd
from torch import nn, Tensor

from src.models.catalog.frame_info import FrameInfo
from src.models.catalog.properties import ColorSpace
from src.udfs.filters.pytorch_abstract_filter import PytorchAbstractFilter


class CNNFilter(PytorchAbstractFilter):
    """
    Filter for removing frames based on presence
    or absence of certain objects using CNN.
    """

    def __init__(self):
        PytorchAbstractFilter.__init__(self)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.linear1 = nn.Linear(16 * 23 * 23, 64)
        self.linear2 = nn.Linear(64, 1)
        self.pool = nn.MaxPool2d(3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # TODO: don't hardcode this
        self.threshold = 0.6194
        self.load_state_dict(torch.load(
            "data/car_filter.pt", map_location=torch.device('cpu')))

    @property
    def name(self) -> str:
        return "object_filter"

    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)

    def _get_predictions(self, frames: Tensor) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: predictions of presence or absence of
            object of a specified class.
        """
        frames = self.relu(self.pool(self.conv1(frames)))
        frames = self.relu(self.pool(self.conv2(frames)))
        frames = self.relu(self.linear1(frames.reshape(frames.shape[0], -1)))
        frames = self.sigmoid(self.linear2(frames))
        return pd.DataFrame(frames.flatten() > self.threshold)