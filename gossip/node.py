import logging
from torch.utils.data import DataLoader
import numpy as np

from client.fl_client import FederatedClient
from utils.weights import bytes_to_weight_arrays, apply_weight_arrays


class GossipNode:
    """
    GossipNode = FederatedClient + gossip inbox.

    Baseline version:
    - No Dilithium
    - No ZKP
    - Uses gossip communication
    - Supports decentralized local aggregation
    - Uses weighted FedAvg aggregation
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
        self.client = FederatedClient(
            client_id=client_id,
            dataloader=dataloader,
            device=device,
            weight_dtype=weight_dtype,
            learning_rate=learning_rate,
            model_name=model_name,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            n_clients_sim=n_clients_sim,
        )

        self.client_id = client_id

        self.own_submission: dict | None = None
        self.inbox: dict[str, dict] = {}

        # Number of local samples owned by this client
        self.num_samples = len(dataloader.dataset)

        logging.info(
            f"[{self.client_id}] gossip node initialized | "
            f"samples={self.num_samples} | "
            f"weight_dtype={weight_dtype} | learning_rate={learning_rate} | "
            f"model={model_name}"
        )

    def local_train(self, global_weight_arrays: list | None, epochs: int = 1):
        """
        Train local client model using received global/aggregated weights.
        """
        self.client.local_train(global_weight_arrays, epochs)

    def prepare_update(self) -> dict:
        """
        Create a normal model update without Dilithium signature or ZKP proof.
        Adds sample count for weighted FedAvg.
        """
        self.own_submission = self.client.prepare_update()

        # Add metadata needed for weighted FedAvg
        self.own_submission["num_samples"] = self.num_samples

        # Reset inbox for this round
        self.inbox.clear()

        logging.info(
            f"[{self.client_id}] plain update prepared and inbox reset | "
            f"num_samples={self.num_samples}"
        )

        return self.own_submission

    def receive_gossip(self, message: dict):
        """
        Receive a gossip message from another client.

        Duplicate updates and returned own updates are ignored.
        """
        if "client_id" not in message:
            logging.warning(f"[{self.client_id}] invalid gossip ignored: missing client_id")
            return

        if "update_bytes" not in message:
            logging.warning(f"[{self.client_id}] invalid gossip ignored: missing update_bytes")
            return

        origin_id = message["client_id"]

        if origin_id == self.client_id:
            logging.warning(
                f"[{self.client_id}] ignored returned own gossip from {origin_id}"
            )
            return

        if origin_id in self.inbox:
            logging.warning(
                f"[{self.client_id}] duplicate gossip ignored from {origin_id}"
            )
            return

        # If old updates do not have num_samples, assume equal weighting later
        if "num_samples" not in message:
            logging.warning(
                f"[{self.client_id}] received update from {origin_id} without num_samples"
            )

        self.inbox[origin_id] = message

        logging.info(
            f"[{self.client_id}] received gossip from {origin_id} | "
            f"inbox_size={len(self.inbox)}"
        )

    def get_all_submissions(self) -> list[dict]:
        """
        Return own update + all unique received gossip updates.
        """
        all_subs = []

        if self.own_submission is not None:
            all_subs.append(self.own_submission)

        all_subs.extend(self.inbox.values())

        return all_subs

    def clear_submissions(self):
        """
        Clear round-specific submissions.
        """
        self.own_submission = None
        self.inbox.clear()

        logging.info(f"[{self.client_id}] cleared round submissions")

    def aggregate_local_updates(self, submissions: list[dict], template_model):
        """
        Aggregate available updates using weighted FedAvg.

        Formula:
            w_global = sum_k (n_k / n_total) * w_k

        where:
            n_k = number of samples at client k
            w_k = model weights from client k
        """
        if not submissions:
            logging.warning(
                f"[{self.client_id}] no submissions available for aggregation"
            )
            return None

        logging.info(
            f"[{self.client_id}] aggregating {len(submissions)} submission(s)"
        )

        dtype_name = self.client.weight_dtype

        weight_sets = []
        sample_counts = []

        for sub in submissions:
            if "update_bytes" not in sub:
                logging.warning(
                    f"[{self.client_id}] skipped invalid submission: missing update_bytes"
                )
                continue

            arrays = bytes_to_weight_arrays(
                sub["update_bytes"],
                template_model,
                dtype_name=dtype_name,
            )

            weight_sets.append(arrays)

            # If num_samples is missing, fallback to equal weight of 1
            sample_counts.append(int(sub.get("num_samples", 1)))

        if not weight_sets:
            logging.warning(
                f"[{self.client_id}] no valid weight sets available after decoding"
            )
            return None

        total_samples = sum(sample_counts)

        if total_samples <= 0:
            logging.warning(
                f"[{self.client_id}] invalid total sample count; using equal average"
            )
            sample_counts = [1 for _ in weight_sets]
            total_samples = len(weight_sets)

        # Weighted FedAvg
        averaged = []

        for layer_idx in range(len(weight_sets[0])):
            weighted_layer = np.zeros_like(weight_sets[0][layer_idx])

            for client_idx, weights in enumerate(weight_sets):
                weight_factor = sample_counts[client_idx] / total_samples
                weighted_layer += weight_factor * weights[layer_idx]

            averaged.append(weighted_layer)

        apply_weight_arrays(self.client.model, averaged)

        logging.info(
            f"[{self.client_id}] local weighted aggregation completed | "
            f"updates={len(weight_sets)} | total_samples={total_samples} | "
            f"sample_counts={sample_counts}"
        )

        return averaged