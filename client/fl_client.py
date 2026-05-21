import logging
import os
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.cnn import LeNet
from utils.weights import apply_weight_arrays, weights_to_bytes


def build_model(
    model_name: str,
    device: str,
    input_channels: int,
    num_classes: int,
    input_height: int,
    input_width: int,
    conv1_channels: int,
    conv2_channels: int,
) -> nn.Module:
    """
    Build and return the selected model.
    Currently supports LeNet.
    """
    model_name = model_name.lower()

    if model_name == "lenet":
        return LeNet(
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
        ).to(device)

    raise ValueError(f"Unsupported model: {model_name}")


class FederatedClient:
    """
    Federated client for Method 1:

        FedAvg + DP

    DP mechanism:
        1. Train local model.
        2. Compute update delta = local model - global model.
        3. Clip update delta using clip_norm.
        4. Add Gaussian noise.
        5. Send noisy update for aggregation.

    For epsilon testing:
        If auto_noise=True:
            noise_std = base_noise / epsilon

        Lower epsilon -> higher noise -> stronger privacy -> lower accuracy.
        Higher epsilon -> lower noise -> weaker privacy -> higher accuracy.
    """

    def __init__(
        self,
        client_id: str,
        dataloader: DataLoader,
        device: str,
        weight_dtype: str,
        learning_rate: float,
        model_name: str,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
        n_clients_sim: int = 10,
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.weight_dtype = weight_dtype
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.n_clients_sim = n_clients_sim

        # ---------------------------------------------------------
        # DP Configuration
        # ---------------------------------------------------------
        self.dp_enabled = os.environ.get("DP_ENABLED", "1") == "1"

        # Clipping norm C
        self.dp_clip_norm = float(os.environ.get("DP_CLIP_NORM", "0.5"))

        # Target epsilon for privacy-utility experiments
        self.epsilon = float(os.environ.get("DP_EPSILON", "0.9"))
        self.delta = float(os.environ.get("DP_DELTA", "1e-5"))

        # If enabled, epsilon controls the actual Gaussian noise.
        # This makes changing epsilon meaningful for accuracy experiments.
        self.auto_noise = os.environ.get("DP_AUTO_NOISE", "1") == "1"
        self.base_noise = float(os.environ.get("DP_BASE_NOISE", "0.05"))

        if self.auto_noise:
            self.dp_noise_std = self.base_noise / max(self.epsilon, 1e-8)
        else:
            self.dp_noise_std = float(os.environ.get("DP_NOISE_STD", "0.01"))

        self.model = build_model(
            model_name=self.model_name,
            device=self.device,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer_name = os.environ.get("OPTIMIZER", "adam").lower()
        self.optimizer = self._build_optimizer()

        logging.info(
            f"[{self.client_id}] initialized | "
            f"approach=FedAvg+DP | "
            f"model={self.model_name} | optimizer={self.optimizer_name} | "
            f"learning_rate={self.learning_rate} | weight_dtype={self.weight_dtype}"
        )

        logging.info(
            f"[{self.client_id}] DP settings | "
            f"enabled={self.dp_enabled} | "
            f"clip_norm={self.dp_clip_norm} | "
            f"epsilon={self.epsilon} | "
            f"delta={self.delta} | "
            f"auto_noise={self.auto_noise} | "
            f"base_noise={self.base_noise} | "
            f"calculated_noise_std={self.dp_noise_std:.4f}"
        )

    def _build_optimizer(self):
        """
        Build optimizer.
        Keep the same optimizer for all 3 methods for fair comparison.
        """
        if self.optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )

        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _reset_optimizer(self):
        """
        Reset optimizer every FL round.
        """
        self.optimizer = self._build_optimizer()

    def _get_trainable_state(self):
        """
        Save trainable parameters before local training.
        """
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _apply_client_level_dp(self, initial_state):
        """
        Apply client-level update clipping and Gaussian noise.

        delta = local_model - global_model
        clipped_delta = delta * min(1, C / ||delta||_2)
        private_update = clipped_delta + Gaussian noise
        final_model = global_model + private_update
        """
        total_norm_sq = 0.0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            delta = param.data - initial_state[name]
            total_norm_sq += torch.sum(delta ** 2).item()

        total_norm = total_norm_sq ** 0.5

        clip_factor = min(
            1.0,
            self.dp_clip_norm / (total_norm + 1e-12),
        )

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                delta = param.data - initial_state[name]
                clipped_delta = delta * clip_factor

                if self.dp_enabled and self.dp_noise_std > 0:
                    noise = torch.normal(
                        mean=0.0,
                        std=self.dp_noise_std,
                        size=clipped_delta.shape,
                        device=clipped_delta.device,
                        dtype=clipped_delta.dtype,
                    )
                else:
                    noise = torch.zeros_like(clipped_delta)

                param.data.copy_(initial_state[name] + clipped_delta + noise)

        return total_norm, clip_factor

    def _local_accuracy(self):
        """
        Evaluate model on the client's local data.
        This is only local accuracy, not global accuracy.
        """
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x_v, y_v in self.dataloader:
                x_v = x_v.to(self.device)
                y_v = y_v.to(self.device)

                outputs = self.model(x_v)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == y_v).sum().item()
                total += y_v.size(0)

        return correct / total if total > 0 else 0.0

    def local_train(self, global_weight_arrays=None, epochs=1):
        """
        Perform local training and then apply client-level DP.
        """
        start_time = time.time()

        if global_weight_arrays is not None:
            apply_weight_arrays(self.model, global_weight_arrays)

        if epochs == 0:
            logging.info(f"[{self.client_id}] skipped local training because epochs=0")
            return

        initial_state = self._get_trainable_state()

        self._reset_optimizer()
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)
                loss = self.criterion(logits, y)

                loss.backward()

                # Training stability clipping.
                # This is separate from DP update clipping.
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

                if epoch == 0 and batch_idx == 0:
                    pred = torch.argmax(logits, dim=1)
                    logging.info(
                        f"[{self.client_id}] sample prediction check | "
                        f"pred={pred[0].item()} | actual={y[0].item()}"
                    )

        if self.dp_enabled:
            total_norm, clip_factor = self._apply_client_level_dp(initial_state)
        else:
            total_norm = 0.0
            clip_factor = 1.0

        local_acc = self._local_accuracy()

        exec_ms = (time.time() - start_time) * 1000
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

        logging.info(
            f"[{self.client_id}] local train completed | "
            f"loss={avg_loss:.4f} | "
            f"local_acc={local_acc * 100:.2f}% | "
            f"dp_enabled={self.dp_enabled} | "
            f"update_norm={total_norm:.4f} | "
            f"clip_factor={clip_factor:.4f} | "
            f"epsilon={self.epsilon} | "
            f"noise_std={self.dp_noise_std:.4f} | "
            f"time={exec_ms:.2f}ms"
        )

    def prepare_update(self) -> dict:
        """
        Prepare plain DP-protected model update payload.

        Method 1:
            FedAvg + DP
            No Dilithium
            No ZKP
        """
        start_time = time.time()

        update_bytes = weights_to_bytes(self.model, self.weight_dtype)

        payload = {
            "client_id": self.client_id,
            "update_bytes": update_bytes,

            # DP metadata
            "dp_enabled": self.dp_enabled,
            "dp_clip_norm": self.dp_clip_norm,
            "dp_noise_std": self.dp_noise_std,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "auto_noise": self.auto_noise,
            "base_noise": self.base_noise,
        }

        payload_size_kb = (
            len(self.client_id.encode("utf-8"))
            + len(update_bytes)
        ) / 1024.0

        prep_time = time.time() - start_time

        logging.info(
            f"[{self.client_id}] DP update prepared | "
            f"epsilon={self.epsilon} | "
            f"noise_std={self.dp_noise_std:.4f} | "
            f"payload_size={payload_size_kb:.2f} KB | "
            f"prep_time={prep_time:.4f}s"
        )

        return payload