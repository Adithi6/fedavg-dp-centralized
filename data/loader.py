import logging
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


def _label_distribution(labels: list[int]) -> dict[int, int]:
    """
    Count number of samples per class label.
    Example:
        {0: 120, 1: 340, 2: 45}
    """
    distribution = {}

    for label in labels:
        distribution[label] = distribution.get(label, 0) + 1

    return dict(sorted(distribution.items()))


def _partition_to_tensordataset(
    partition,
    normalize_mean: list[float],
    normalize_std: list[float],
) -> tuple[TensorDataset, list[int]]:
    """
    Convert one Flower/HuggingFace partition into a PyTorch TensorDataset.

    Returns:
        dataset: TensorDataset
        labels: list of labels for distribution logging
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(normalize_mean), tuple(normalize_std)),
    ])

    images = []
    labels = []

    for item in partition:
        image = transform(item["image"])
        label = int(item["label"])

        images.append(image)
        labels.append(label)

    if len(images) == 0:
        raise ValueError(
            "Empty client partition found. Try increasing min_partition_size "
            "or using a larger alpha value."
        )

    x_tensor = torch.stack(images)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)

    return dataset, labels


def make_client_loaders(
    n_clients: int,
    batch_size: int,
    alpha: float,
    dataset_name: str,
    partition_by: str,
    min_partition_size: int,
    self_balancing: bool,
    seed: int,
    test_batch_size: int,
    normalize_mean: list[float],
    normalize_std: list[float],
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create non-IID client DataLoaders using Flower DirichletPartitioner.

    Logs:
        - number of samples per client
        - unique labels per client
        - full label distribution per client
    """

    logging.info("=" * 80)
    logging.info("Creating non-IID client data partitions")
    logging.info("=" * 80)

    logging.info(
        f"Dataset={dataset_name} | "
        f"clients={n_clients} | "
        f"alpha={alpha} | "
        f"partition_by={partition_by} | "
        f"min_partition_size={min_partition_size} | "
        f"self_balancing={self_balancing} | "
        f"seed={seed}"
    )

    partitioner = DirichletPartitioner(
        num_partitions=n_clients,
        partition_by=partition_by,
        alpha=alpha,
        min_partition_size=min_partition_size,
        self_balancing=self_balancing,
        seed=seed,
    )

    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": partitioner},
    )

    client_loaders: List[DataLoader] = []

    all_client_label_distributions = {}

    for client_id in range(n_clients):
        partition = fds.load_partition(client_id, "train")

        dataset, labels = _partition_to_tensordataset(
            partition,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        client_loaders.append(loader)

        unique_labels = sorted(set(labels))
        label_dist = _label_distribution(labels)

        all_client_label_distributions[client_id] = label_dist

        logging.info("-" * 80)
        logging.info(
            f"Client {client_id} data distribution | "
            f"total_samples={len(labels)} | "
            f"unique_labels={unique_labels}"
        )

        logging.info(
            f"Client {client_id} label_distribution={label_dist}"
        )

    logging.info("=" * 80)
    logging.info("Client-wise label distribution summary")
    logging.info("=" * 80)

    for client_id, label_dist in all_client_label_distributions.items():
        logging.info(f"Client {client_id}: {label_dist}")

    logging.info("=" * 80)

    # ---------------- TEST DATA ----------------
    test_partition = fds.load_split("test")

    test_dataset, test_labels = _partition_to_tensordataset(
        test_partition,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_label_dist = _label_distribution(test_labels)

    logging.info(
        f"Test set created | "
        f"total_samples={len(test_labels)} | "
        f"label_distribution={test_label_dist}"
    )

    logging.info(
        f"Created {n_clients} client loaders using Dirichlet non-IID partitioning"
    )

    return client_loaders, test_loader