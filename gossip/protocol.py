import logging
import random
import time
from typing import List, Dict, Any


class GossipProtocol:
    """
    Gossip protocol for decentralized FL communication.

    Each client update is spread using randomized peer forwarding.

    Parameters:
        fanout: Number of peers selected by each node for forwarding.
        max_hops: Maximum forwarding depth allowed for each update.

    Features:
        - Prevents repeated forwarding by the same node for the same origin.
        - Avoids forwarding back to the same node unnecessarily.
        - Tracks gossip communication logs.
        - Supports duplicate filtering at receiver side through GossipNode.
    """

    def __init__(self, fanout: int, max_hops: int, seed: int | None = None):
        if fanout <= 0:
            raise ValueError("fanout must be greater than 0")

        if max_hops <= 0:
            raise ValueError("max_hops must be greater than 0")

        self.fanout = fanout
        self.max_hops = max_hops
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        # Tracks forwarding state:
        # (origin_client_id, forwarder_client_id)
        self._seen_forward: set[tuple[str, str]] = set()

        # Tracks who already received each origin update
        self._received_by_origin: dict[str, set[str]] = {}

        self.gossip_timings: list[dict] = []

    def reset_round(self):
        """
        Reset gossip state at the beginning of each FL round.
        """
        self._seen_forward.clear()
        self._received_by_origin.clear()
        self.gossip_timings.clear()

        logging.info("Gossip round state reset")

    def spread(
        self,
        origin_node,
        all_nodes,
        message: dict,
        hop: int = 0,
    ):
        """
        Recursively spread one client's update through the network.

        Args:
            origin_node: The node currently forwarding the message.
            all_nodes: List of all gossip nodes.
            message: Update message prepared by the original client.
            hop: Current hop depth.
        """
        if "client_id" not in message:
            raise ValueError("Invalid gossip message: missing client_id")

        if "update_bytes" not in message:
            raise ValueError("Invalid gossip message: missing update_bytes")

        origin_client_id = message["client_id"]
        forwarder_id = origin_node.client_id

        state_id = (origin_client_id, forwarder_id)

        if state_id in self._seen_forward:
            logging.info(
                f"[gossip] update from {origin_client_id} already forwarded by "
                f"{forwarder_id}; skipping"
            )
            return

        if hop >= self.max_hops:
            logging.info(
                f"[gossip] max hops reached | origin={origin_client_id} | "
                f"forwarder={forwarder_id} | hop={hop}"
            )
            return

        self._seen_forward.add(state_id)

        if origin_client_id not in self._received_by_origin:
            self._received_by_origin[origin_client_id] = set()

        # Choose peers except:
        # 1. current forwarder itself
        # 2. original client
        # 3. nodes that already received this origin update
        peers = [
            node for node in all_nodes
            if node.client_id != forwarder_id
            and node.client_id != origin_client_id
            and node.client_id not in self._received_by_origin[origin_client_id]
        ]

        if not peers:
            logging.info(
                f"[gossip] no new peers available | "
                f"origin={origin_client_id} | forwarder={forwarder_id}"
            )
            return

        targets = random.sample(peers, min(self.fanout, len(peers)))

        for target in targets:
            start_time = time.perf_counter()

            target.receive_gossip(message)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self._received_by_origin[origin_client_id].add(target.client_id)

            self.gossip_timings.append({
                "origin": origin_client_id,
                "from": forwarder_id,
                "to": target.client_id,
                "hop": hop + 1,
                "accepted": True,
                "time_ms": elapsed_ms,
            })

            logging.info(
                f"[gossip] {forwarder_id} -> {target.client_id} | "
                f"origin={origin_client_id} | hop={hop + 1} | "
                f"time={elapsed_ms:.4f} ms | [FORWARDED]"
            )

            self.spread(
                origin_node=target,
                all_nodes=all_nodes,
                message=message,
                hop=hop + 1,
            )

    def run_round(self, nodes):
        """
        Run one full gossip communication round.

        Each node must first call prepare_update().
        Then every client's update is spread through the gossip network.
        """
        self.reset_round()

        for node in nodes:
            if node.own_submission is None:
                raise RuntimeError(
                    f"{node.client_id} has no submission. "
                    f"Call prepare_update() before run_round()."
                )

        for node in nodes:
            logging.info(f"[gossip] spreading update from {node.client_id}")

            self.spread(
                origin_node=node,
                all_nodes=nodes,
                message=node.own_submission,
                hop=0,
            )

    def print_gossip_summary(self):
        """
        Print gossip communication summary for the current round.
        """
        if not self.gossip_timings:
            logging.info("No gossip records available for this round")
            return

        logging.info("-" * 90)
        logging.info(
            f"Gossip log | fanout={self.fanout} | max_hops={self.max_hops}"
        )
        logging.info("-" * 90)
        logging.info(
            f"{'Origin':<12} {'From':<12} {'To':<12} "
            f"{'Hop':<5} {'Accepted':<10} {'Time(ms)':<10}"
        )
        logging.info("-" * 90)

        for record in self.gossip_timings:
            logging.info(
                f"{record['origin']:<12} "
                f"{record['from']:<12} "
                f"{record['to']:<12} "
                f"{record['hop']:<5} "
                f"{str(record['accepted']):<10} "
                f"{record['time_ms']:<10.4f}"
            )

        logging.info("-" * 90)
        logging.info(f"Total gossip transmissions: {len(self.gossip_timings)}")

        coverage = {}

        for record in self.gossip_timings:
            origin = record["origin"]
            receiver = record["to"]

            if origin not in coverage:
                coverage[origin] = set()

            coverage[origin].add(receiver)

        for origin, receivers in coverage.items():
            logging.info(
                f"Coverage for {origin}: "
                f"{len(receivers)} receiver(s) -> {sorted(receivers)}"
            )