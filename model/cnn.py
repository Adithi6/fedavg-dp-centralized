import logging
import torch
import torch.nn as nn


def get_group_count(num_channels: int, preferred_groups: int = 4) -> int:
    """
    Return a valid GroupNorm group count.
    num_channels must be divisible by num_groups.
    """
    if num_channels % preferred_groups == 0:
        return preferred_groups

    for groups in [8, 4, 2, 1]:
        if num_channels % groups == 0:
            return groups

    return 1


class LeNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
    ):
        super().__init__()

        group1 = get_group_count(conv1_channels)
        group2 = get_group_count(conv2_channels)

        # GroupNorm is used instead of BatchNorm because it is more suitable
        # for DP-style training and small local client batches.
        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                conv1_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.GroupNorm(group1, conv1_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                conv1_channels,
                conv2_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.GroupNorm(group2, conv2_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flattened_dim = self._get_flattened_dim(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
        )

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logging.info(
            "LeNet with GroupNorm initialized | "
            f"conv1_channels={conv1_channels} | conv2_channels={conv2_channels} | "
            f"group1={group1} | group2={group2} | "
            f"flattened_dim={flattened_dim} | "
            f"total_params={total_params} | trainable_params={trainable_params}"
        )

    def _get_flattened_dim(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
    ) -> int:
        """
        Dynamically calculate flattened feature dimension.
        This avoids hardcoding dimensions like 32*7*7.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            out = self.features(dummy)
            return out.reshape(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x