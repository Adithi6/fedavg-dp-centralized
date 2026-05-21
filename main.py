import logging
import yaml
import time
import random
import os

import torch

from data.loader import make_client_loaders
from gossip.node import GossipNode
from gossip.protocol import GossipProtocol
from utils.weights import model_to_weight_arrays


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
    Bridge YAML DP config to FederatedClient.

    This version supports epsilon-based automatic noise:

        if auto_noise = true:
            noise_std = base_noise / epsilon

    This makes changing epsilon meaningful:
        lower epsilon -> higher noise -> stronger privacy -> lower accuracy
        higher epsilon -> lower noise -> weaker privacy -> higher accuracy
    """
    dp = config.get("dp", {})
    training = config.get("training", {})

    dp_enabled = dp.get("enabled", True)
    dp_clip_norm = float(dp.get("clip_norm", 0.5))
    dp_epsilon = float(dp.get("epsilon", 0.9))
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

    epsilon = float(dp.get("epsilon", 0.9))
    auto_noise = str(dp.get("auto_noise", True)).lower() in ["true", "1", "yes"]
    base_noise = float(dp.get("base_noise", 0.05))
    manual_noise = float(dp.get("noise_std", 0.01))

    if auto_noise:
        return base_noise / max(epsilon, 1e-8)

    return manual_noise


def log_experiment_summary(config: dict):
    """
    Log key experiment settings.
    """
    experiment = config["experiment"]
    gossip = config["gossip"]
    model = config["model"]
    data = config["data"]
    dp = config.get("dp", {})
    training = config["training"]

    effective_noise = get_effective_noise_std(config)

    logging.info("-" * 80)
    logging.info("Experiment Configuration Summary")
    logging.info("-" * 80)

    logging.info("Approach: FedAvg + DP")

    logging.info(
        f"FL: clients={experiment['n_clients']} | "
        f"rounds={experiment['n_rounds']} | "
        f"local_epochs={experiment['local_epochs']}"
    )

    logging.info(
        f"Gossip: fanout={gossip['fanout']} | "
        f"max_hops={gossip['max_hops']}"
    )

    logging.info(
        f"Model: {model['name']} | "
        f"conv1={model['conv1_channels']} | "
        f"conv2={model['conv2_channels']} | "
        f"classes={model['num_classes']}"
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
        f"epsilon={dp.get('epsilon', 0.9)} | "
        f"delta={dp.get('delta', 1e-5)} | "
        f"auto_noise={dp.get('auto_noise', True)} | "
        f"base_noise={dp.get('base_noise', 0.05)} | "
        f"effective_noise_std={effective_noise:.4f}"
    )

    logging.info("-" * 80)


# -------------------------------------------------------------------
# FL Helpers
# -------------------------------------------------------------------

def choose_aggregator_node(nodes):
    """
    Select the node that received the maximum number of submissions.

    If multiple nodes have the same maximum count, select one randomly.
    """
    counts = []

    for node in nodes:
        submissions = node.get_all_submissions()
        count = len(submissions)
        counts.append((node, count))

        logging.info(
            f"[aggregator-selection] {node.client_id} submissions={count}"
        )

    max_count = max(count for _, count in counts)

    candidates = [
        node for node, count in counts
        if count == max_count
    ]

    aggregator = random.choice(candidates)

    logging.info(
        f"Selected aggregator: {aggregator.client_id} | "
        f"submissions={max_count}"
    )

    return aggregator


def sync_weights_to_all_nodes(nodes, weights):
    """
    Synchronize aggregated model weights to all clients.

    This uses local_train(..., epochs=0), because the client implementation
    applies global weights before checking epochs.
    """
    for node in nodes:
        node.local_train(weights, epochs=0)

    logging.info("Aggregated weights synced to all nodes")


def clear_round_state(nodes, gossip):
    """
    Clear round-specific node submissions and gossip state.
    """
    for node in nodes:
        node.clear_submissions()

    gossip.reset_round()


def evaluate_model(model, test_loader, device: str) -> float:
    """
    Evaluate global model on test data.
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

    data_config = config["data"]
    seed = int(data_config.get("seed", 42))

    set_reproducibility(seed)
    apply_dp_config_to_environment(config)
    log_experiment_summary(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # ---------------- BASIC CONFIG ----------------
    experiment_config = config["experiment"]
    gossip_config = config["gossip"]
    training_config = config["training"]
    model_config = config["model"]
    weights_config = config["weights"]
    dp_config = config.get("dp", {})

    n_clients = int(experiment_config["n_clients"])
    n_rounds = int(experiment_config["n_rounds"])
    local_epochs = int(experiment_config["local_epochs"])

    gossip_fanout = int(gossip_config["fanout"])
    gossip_max_hops = int(gossip_config["max_hops"])

    learning_rate = float(training_config["learning_rate"])
    dp_n_clients_sim = int(dp_config.get("n_clients_sim", n_clients))

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

    # ---------------- NODES ----------------
    nodes = []

    for i in range(n_clients):
        node = GossipNode(
            client_id=f"client_{i}",
            dataloader=client_loaders[i],
            device=device,
            learning_rate=learning_rate,
            model_name=model_config["name"],
            weight_dtype=weights_config["dtype"],
            input_channels=int(model_config["input_channels"]),
            num_classes=int(model_config["num_classes"]),
            input_height=int(model_config["input_height"]),
            input_width=int(model_config["input_width"]),
            conv1_channels=int(model_config["conv1_channels"]),
            conv2_channels=int(model_config["conv2_channels"]),
            n_clients_sim=dp_n_clients_sim,
        )

        nodes.append(node)

    logging.info(f"Created {len(nodes)} gossip FL nodes")

    # ---------------- GOSSIP ----------------
    gossip = GossipProtocol(
        fanout=gossip_fanout,
        max_hops=gossip_max_hops,
        seed=seed,
    )

    # ---------------- INITIAL MODEL SYNC ----------------
    initializer = random.choice(nodes)
    init_weights = model_to_weight_arrays(initializer.client.model)

    sync_weights_to_all_nodes(nodes, init_weights)

    logging.info(
        f"Initial global model taken from {initializer.client_id} "
        f"and synced to all nodes"
    )

    initial_accuracy = evaluate_model(initializer.client.model, test_loader, device)

    logging.info(
        f"Initial global test accuracy before training: {initial_accuracy * 100:.2f}%"
    )

    # ---------------- TRAINING ----------------
    experiment_start_time = time.time()

    round_accuracies = []
    round_times = []

    for round_id in range(1, n_rounds + 1):
        round_start_time = time.time()

        logging.info("=" * 80)
        logging.info(f"Round {round_id}/{n_rounds}")
        logging.info("=" * 80)

        clear_round_state(nodes, gossip)

        # 1. Local training from latest synced model
        logging.info(f"Round {round_id}: local training started")

        for node in nodes:
            node.local_train(None, epochs=local_epochs)

        logging.info(f"Round {round_id}: local training completed")

        # 2. Prepare DP-updated model updates
        logging.info(f"Round {round_id}: preparing client updates")

        for node in nodes:
            node.prepare_update()

        logging.info(f"Round {round_id}: client updates prepared")

        # 3. Gossip propagation
        logging.info(f"Round {round_id}: gossip propagation started")

        gossip.run_round(nodes)
        gossip.print_gossip_summary()

        logging.info(f"Round {round_id}: gossip propagation completed")

        # 4. Select aggregator
        aggregator = choose_aggregator_node(nodes)
        submissions = aggregator.get_all_submissions()

        if len(submissions) == n_clients:
            logging.info(
                f"[{aggregator.client_id}] complete aggregation possible | "
                f"{len(submissions)}/{n_clients} submissions"
            )
        else:
            logging.warning(
                f"[{aggregator.client_id}] incomplete aggregation | "
                f"{len(submissions)}/{n_clients} submissions available"
            )

        if len(submissions) == 0:
            round_time = time.time() - round_start_time
            round_times.append(round_time)

            logging.warning(
                f"Round {round_id} skipped because no submissions were available"
            )
            logging.info(
                f"Round {round_id} summary | accuracy=N/A | round_time={round_time:.2f}s"
            )
            continue

        # 5. Aggregate
        averaged_weights = aggregator.aggregate_local_updates(
            submissions,
            aggregator.client.model,
        )

        if averaged_weights is None:
            round_time = time.time() - round_start_time
            round_times.append(round_time)

            logging.warning(
                f"Round {round_id} skipped because aggregation returned None"
            )
            logging.info(
                f"Round {round_id} summary | accuracy=N/A | round_time={round_time:.2f}s"
            )
            continue

        # 6. Sync aggregated global model to all clients
        weights = model_to_weight_arrays(aggregator.client.model)
        sync_weights_to_all_nodes(nodes, weights)

        # 7. Evaluate global model
        accuracy = evaluate_model(aggregator.client.model, test_loader, device)
        round_accuracies.append(accuracy)

        logging.info(
            f"Round {round_id} global test accuracy: {accuracy * 100:.2f}%"
        )

        # 8. Log DP setting instead of misleading huge calculated epsilon
        logging.info(
            f"Round {round_id} DP setting | "
            f"target_epsilon={dp_config.get('epsilon', 0.9)} | "
            f"delta={dp_config.get('delta', 1e-5)} | "
            f"clip_norm={dp_config.get('clip_norm', 0.5)} | "
            f"effective_noise_std={effective_noise:.4f}"
        )

        # 9. Round time summary
        round_time = time.time() - round_start_time
        round_times.append(round_time)

        logging.info(
            f"Round {round_id} summary | "
            f"accuracy={accuracy * 100:.2f}% | "
            f"round_time={round_time:.2f}s"
        )

        logging.info(f"Round {round_id} completed")

        clear_round_state(nodes, gossip)

    experiment_end_time = time.time()
    total_time = experiment_end_time - experiment_start_time

    # ---------------- FINAL SUMMARY ----------------
    logging.info("=" * 80)
    logging.info("Experiment Completed: FedAvg + DP")
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
        f"target_epsilon={dp_config.get('epsilon', 0.9)} | "
        f"clip_norm={dp_config.get('clip_norm', 0.5)} | "
        f"effective_noise_std={effective_noise:.4f} | "
        f"delta={dp_config.get('delta', 1e-5)}"
    )

    logging.info("=" * 80)


if __name__ == "__main__":
    main()