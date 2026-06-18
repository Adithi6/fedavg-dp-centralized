import logging
import yaml
import time
import random
import os

import numpy as np
import torch

from data.loader import make_client_loaders
from client.fl_client import FederatedClient
from utils.weights import (
    model_to_weight_arrays,
    apply_weight_arrays,
    bytes_to_weight_arrays,
)


# -------------------------------------------------------------------
# Config and Logging
# -------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """
    Load YAML configuration file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """
    Configure logging to both file and terminal.
    """
    log_file = config["logging"].get("log_file", "experiment.log")
    log_level = config["logging"].get("log_level", "INFO").upper()

    # Reset old handlers to avoid duplicate logs if rerun in same process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(
                log_file,
                mode="w",
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )


def set_reproducibility(seed: int):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"Random seed set to {seed}")


def _to_bool_string(value) -> str:
    """
    Convert YAML bool/string/int to environment-friendly 1/0 string.
    """
    value = str(value).lower()
    return "1" if value in ["true", "1", "yes"] else "0"


def apply_dp_config_to_environment(config: dict):
    """
    Bridge YAML DP config to FederatedClient via environment variables.

    If auto_noise = true:
        noise_std = base_noise / epsilon

    Lower epsilon -> higher noise -> stronger privacy -> lower accuracy.
    Higher epsilon -> lower noise -> weaker privacy -> higher accuracy.
    """
    dp = config.get("dp", {})
    training = config.get("training", {})

    dp_enabled = dp.get("enabled", True)
    dp_clip_norm = float(dp.get("clip_norm", 0.5))
    dp_epsilon = float(dp.get("epsilon", 1.0))
    dp_delta = float(dp.get("delta", 1e-5))

    dp_auto_noise = dp.get("auto_noise", True)
    dp_base_noise = float(dp.get("base_noise", 0.05))

    # Used only if auto_noise = false
    dp_noise_std = float(dp.get("noise_std", 0.01))

    optimizer = str(training.get("optimizer", "adam"))

    os.environ["DP_ENABLED"] = _to_bool_string(dp_enabled)
    os.environ["DP_CLIP_NORM"] = str(dp_clip_norm)
    os.environ["DP_EPSILON"] = str(dp_epsilon)
    os.environ["DP_DELTA"] = str(dp_delta)

    os.environ["DP_AUTO_NOISE"] = _to_bool_string(dp_auto_noise)
    os.environ["DP_BASE_NOISE"] = str(dp_base_noise)
    os.environ["DP_NOISE_STD"] = str(dp_noise_std)

    os.environ["OPTIMIZER"] = optimizer

    if os.environ["DP_AUTO_NOISE"] == "1":
        calculated_noise = dp_base_noise / max(dp_epsilon, 1e-8)
    else:
        calculated_noise = dp_noise_std

    logging.info(
        "DP config applied from YAML | "
        f"enabled={os.environ['DP_ENABLED']} | "
        f"clip_norm={dp_clip_norm} | "
        f"epsilon={dp_epsilon} | "
        f"delta={dp_delta} | "
        f"auto_noise={os.environ['DP_AUTO_NOISE']} | "
        f"base_noise={dp_base_noise} | "
        f"calculated_noise_std={calculated_noise:.4f} | "
        f"optimizer={optimizer}"
    )


def get_effective_noise_std(config: dict) -> float:
    """
    Return actual noise_std used by clients.
    """
    dp = config.get("dp", {})

    epsilon = float(dp.get("epsilon", 1.0))
    auto_noise = str(dp.get("auto_noise", True)).lower() in ["true", "1", "yes"]
    base_noise = float(dp.get("base_noise", 0.05))
    manual_noise = float(dp.get("noise_std", 0.01))

    if auto_noise:
        return base_noise / max(epsilon, 1e-8)

    return manual_noise


def log_experiment_summary(config: dict):
    """
    Log key experiment settings for Centralized FedAvg + Differential Privacy.
    """
    experiment = config["experiment"]
    model_cfg = config["model"]
    data = config["data"]
    dp = config.get("dp", {})
    training = config["training"]

    effective_noise = get_effective_noise_std(config)

    logging.info("-" * 80)
    logging.info("Experiment Configuration Summary")
    logging.info("-" * 80)
    logging.info("Approach: Centralized FedAvg + Differential Privacy")
    logging.info("Architecture: Central Server -> Clients -> Central Server Aggregation")

    logging.info(
        f"FL: clients={experiment['n_clients']} | "
        f"rounds={experiment['n_rounds']} | "
        f"local_epochs={experiment['local_epochs']}"
    )

    logging.info(
        f"Model: {model_cfg['name']} | "
        f"conv1={model_cfg['conv1_channels']} | "
        f"conv2={model_cfg['conv2_channels']} | "
        f"classes={model_cfg['num_classes']}"
    )

    logging.info(
        f"Training: optimizer={training.get('optimizer', 'adam')} | "
        f"learning_rate={training['learning_rate']}"
    )

    logging.info(
        f"Data: dataset={data['dataset_name']} | "
        f"batch_size={data['batch_size']} | "
        f"test_batch_size={data['test_batch_size']} | "
        f"alpha={data['alpha']} | "
        f"seed={data['seed']}"
    )

    logging.info(
        f"DP: enabled={dp.get('enabled', True)} | "
        f"clip_norm={dp.get('clip_norm', 0.5)} | "
        f"epsilon={dp.get('epsilon', 1.0)} | "
        f"delta={dp.get('delta', 1e-5)} | "
        f"auto_noise={dp.get('auto_noise', True)} | "
        f"base_noise={dp.get('base_noise', 0.05)} | "
        f"effective_noise_std={effective_noise:.4f}"
    )

    logging.info("-" * 80)


# -------------------------------------------------------------------
# Central Server Helpers
# -------------------------------------------------------------------

def build_global_model(model_config: dict, device: str):
    """
    Central server initializes the global model from scratch.
    """
    from model.cnn import LeNet

    model = LeNet(
        input_channels=int(model_config["input_channels"]),
        num_classes=int(model_config["num_classes"]),
        input_height=int(model_config["input_height"]),
        input_width=int(model_config["input_width"]),
        conv1_channels=int(model_config["conv1_channels"]),
        conv2_channels=int(model_config["conv2_channels"]),
    ).to(device)

    logging.info(
        "Central server: global model initialized | "
        f"model={model_config['name']} | device={device}"
    )

    return model


def server_broadcast_weights(clients: list, global_model) -> list:
    """
    Central server sends global model weights to all clients.
    """
    global_weights = model_to_weight_arrays(global_model)

    for client in clients:
        apply_weight_arrays(client.model, global_weights)

    logging.info(
        f"Central server: broadcasted global weights to {len(clients)} clients"
    )

    return global_weights


def server_fedavg_aggregate(
    submissions: list,
    global_model,
    weight_dtype: str,
) -> list:
    """
    Central server performs weighted FedAvg aggregation over all client updates.

    Formula:
        w_global = sum_k (n_k / n_total) * w_k

    where:
        n_k = number of samples at client k
        w_k = DP-protected model weights from client k
    """
    logging.info(
        "Central server aggregating client updates | "
        f"submissions={len(submissions)}"
    )

    weight_sets = []
    sample_counts = []

    for sub in submissions:
        arrays = bytes_to_weight_arrays(
            sub["update_bytes"],
            global_model,
            dtype_name=weight_dtype,
        )
        weight_sets.append(arrays)
        sample_counts.append(int(sub.get("num_samples", 1)))

    if not weight_sets:
        logging.warning("Central server: no valid weight sets for aggregation")
        return None

    total_samples = sum(sample_counts)

    if total_samples <= 0:
        logging.warning("Central server: invalid total sample count; using equal weights")
        sample_counts = [1] * len(weight_sets)
        total_samples = len(weight_sets)

    # Weighted FedAvg
    averaged = []

    for layer_idx in range(len(weight_sets[0])):
        weighted_layer = np.zeros_like(weight_sets[0][layer_idx])

        for client_idx, w in enumerate(weight_sets):
            weight_factor = sample_counts[client_idx] / total_samples
            weighted_layer += weight_factor * w[layer_idx]

        averaged.append(weighted_layer)

    apply_weight_arrays(global_model, averaged)

    logging.info(
        "Central server: FedAvg aggregation completed | "
        f"updates={len(weight_sets)} | "
        f"total_samples={total_samples} | "
        f"sample_counts={sample_counts}"
    )

    return averaged


def evaluate_model(model, test_loader, device: str) -> float:
    """
    Central server evaluates global model accuracy on test set.
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # ---------------- CONFIG ----------------
    config = load_config("config.yaml")
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("Centralized FedAvg + Differential Privacy")
    logging.info("=" * 80)

    data_config = config["data"]
    seed = int(data_config.get("seed", 42))

    set_reproducibility(seed)
    apply_dp_config_to_environment(config)
    log_experiment_summary(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # ---------------- EXTRACT CONFIG ----------------
    experiment_config = config["experiment"]
    training_config = config["training"]
    model_config = config["model"]
    weights_config = config["weights"]
    dp_config = config.get("dp", {})

    n_clients = int(experiment_config["n_clients"])
    n_rounds = int(experiment_config["n_rounds"])
    local_epochs = int(experiment_config["local_epochs"])
    learning_rate = float(training_config["learning_rate"])
    weight_dtype = weights_config["dtype"]

    effective_noise = get_effective_noise_std(config)

    # ---------------- DATA ----------------
    client_loaders, test_loader = make_client_loaders(
        n_clients=n_clients,
        batch_size=int(data_config["batch_size"]),
        alpha=float(data_config["alpha"]),
        dataset_name=data_config["dataset_name"],
        partition_by=data_config["partition_by"],
        min_partition_size=int(data_config["min_partition_size"]),
        self_balancing=bool(data_config["self_balancing"]),
        seed=seed,
        test_batch_size=int(data_config["test_batch_size"]),
        normalize_mean=data_config["normalize_mean"],
        normalize_std=data_config["normalize_std"],
    )

    # ---------------- STEP 1: Central Server Initializes Global Model ----------------
    logging.info("=" * 80)
    logging.info("STEP 1: Central server initializing global model")
    logging.info("=" * 80)

    global_model = build_global_model(model_config, device)

    # ---------------- STEP 2: Create Federated Clients ----------------
    logging.info("=" * 80)
    logging.info("STEP 2: Initializing federated clients")
    logging.info("=" * 80)

    clients = []
    client_sample_counts = []

    for i in range(n_clients):
        client = FederatedClient(
            client_id=f"client_{i}",
            dataloader=client_loaders[i],
            device=device,
            learning_rate=learning_rate,
            model_name=model_config["name"],
            weight_dtype=weight_dtype,
            input_channels=int(model_config["input_channels"]),
            num_classes=int(model_config["num_classes"]),
            input_height=int(model_config["input_height"]),
            input_width=int(model_config["input_width"]),
            conv1_channels=int(model_config["conv1_channels"]),
            conv2_channels=int(model_config["conv2_channels"]),
            n_clients_sim=int(dp_config.get("n_clients_sim", n_clients)),
        )
        clients.append(client)
        client_sample_counts.append(len(client_loaders[i].dataset))

    logging.info(f"Initialized {len(clients)} federated clients")

    # Initial broadcast + evaluate
    server_broadcast_weights(clients, global_model)

    initial_accuracy = evaluate_model(global_model, test_loader, device)
    logging.info(
        f"Central server: initial global test accuracy (before training): "
        f"{initial_accuracy * 100:.2f}%"
    )

    # ---------------- TRAINING ROUNDS ----------------
    experiment_start_time = time.time()

    round_accuracies = []
    round_times = []

    for round_id in range(1, n_rounds + 1):
        round_start_time = time.time()

        logging.info("=" * 80)
        logging.info(f"Round {round_id}/{n_rounds} | Centralized FedAvg + Differential Privacy")
        logging.info("=" * 80)

        # ---- STEP 2: Server sends global model weights to all clients ----
        logging.info(f"Round {round_id}: central server broadcasting global weights to all clients")
        server_broadcast_weights(clients, global_model)

        # ---- STEP 3 & 4: Each client trains locally ----
        logging.info(f"Round {round_id}: clients performing local training")

        for client in clients:
            client.local_train(global_weight_arrays=None, epochs=local_epochs)

        logging.info(f"Round {round_id}: local training completed for all clients")

        # ---- STEP 5 & 6: Each client computes DP-protected update and sends to server ----
        logging.info(f"Round {round_id}: clients preparing DP-protected updates for central server")

        submissions = []

        for client, n_samples in zip(clients, client_sample_counts):
            payload = client.prepare_update()
            payload["num_samples"] = n_samples
            submissions.append(payload)

        logging.info(
            f"Round {round_id}: received {len(submissions)}/{n_clients} "
            f"DP-protected updates at central server"
        )

        # ---- STEP 7 & 8: Server performs FedAvg aggregation ----
        logging.info(f"Round {round_id}: central server performing FedAvg aggregation")

        averaged_weights = server_fedavg_aggregate(
            submissions,
            global_model,
            weight_dtype=weight_dtype,
        )

        if averaged_weights is None:
            round_time = time.time() - round_start_time
            round_times.append(round_time)
            logging.warning(f"Round {round_id} skipped: aggregation returned None")
            logging.info(f"Round {round_id} summary | accuracy=N/A | round_time={round_time:.2f}s")
            continue

        # ---- STEP 9: Server evaluates accuracy ----
        accuracy = evaluate_model(global_model, test_loader, device)
        round_accuracies.append(accuracy)

        logging.info(
            f"Round {round_id}: central server global test accuracy: {accuracy * 100:.2f}%"
        )

        logging.info(
            f"Round {round_id} DP setting | "
            f"target_epsilon={dp_config.get('epsilon', 1.0)} | "
            f"delta={dp_config.get('delta', 1e-5)} | "
            f"clip_norm={dp_config.get('clip_norm', 0.5)} | "
            f"effective_noise_std={effective_noise:.4f}"
        )

        round_time = time.time() - round_start_time
        round_times.append(round_time)

        logging.info(
            f"Round {round_id} summary | "
            f"accuracy={accuracy * 100:.2f}% | "
            f"round_time={round_time:.2f}s"
        )

        logging.info(f"Round {round_id} completed")

    experiment_end_time = time.time()
    total_time = experiment_end_time - experiment_start_time

    # ---------------- FINAL SUMMARY ----------------
    logging.info("=" * 80)
    logging.info("Experiment Completed: Centralized FedAvg + Differential Privacy")
    logging.info("=" * 80)
    logging.info(f"Total training time: {total_time:.2f}s")

    if round_accuracies:
        final_accuracy = round_accuracies[-1]
        best_accuracy = max(round_accuracies)

        logging.info(f"Final global accuracy: {final_accuracy * 100:.2f}%")
        logging.info(f"Best global accuracy: {best_accuracy * 100:.2f}%")

        logging.info("Round-wise accuracy and time summary:")

        for idx, acc in enumerate(round_accuracies, start=1):
            time_value = round_times[idx - 1] if idx - 1 < len(round_times) else 0.0
            logging.info(
                f"Round {idx}: accuracy={acc * 100:.2f}% | "
                f"round_time={time_value:.2f}s"
            )
    else:
        logging.warning("No round accuracies recorded")

    if round_times:
        avg_round_time = sum(round_times) / len(round_times)
        logging.info(f"Average round time: {avg_round_time:.2f}s")

    logging.info(
        f"Final DP setting | "
        f"target_epsilon={dp_config.get('epsilon', 1.0)} | "
        f"clip_norm={dp_config.get('clip_norm', 0.5)} | "
        f"effective_noise_std={effective_noise:.4f} | "
        f"delta={dp_config.get('delta', 1e-5)}"
    )

    logging.info("=" * 80)


if __name__ == "__main__":
    main()