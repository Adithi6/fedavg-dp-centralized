# Secure Decentralized Federated Learning using Gossip Protocol

This project implements a secure decentralized Federated Learning framework using FedAvg, Differential Privacy (DP), and Gossip-based model propagation.

Unlike traditional Federated Learning systems, this implementation does not rely on a central server. Clients communicate directly with peer nodes and exchange model updates through a gossip protocol, improving scalability and fault tolerance while preserving data privacy.

---

## Features

- Decentralized Federated Learning (No Central Server)
- FedAvg-based Model Aggregation
- Differential Privacy with Gradient Clipping and Noise Addition
- Gossip-based Peer-to-Peer Model Propagation
- Non-IID Data Distribution using Dirichlet Partitioning
- LeNet-style CNN for MNIST Classification
- Configurable Privacy and Communication Parameters
- YAML-based Experiment Configuration

---

## System Architecture

1. Initialize participating clients.
2. Distribute non-IID MNIST data across clients.
3. Each client trains locally on private data.
4. Differential Privacy is applied to model updates.
5. Updates are propagated using the Gossip Protocol.
6. Clients aggregate received updates using FedAvg.
7. Training continues for multiple communication rounds.

---

## Experimental Setup

| Parameter | Value |
|------------|---------|
| Dataset | MNIST |
| Model | LeNet-style CNN |
| Number of Clients | 10 |
| Communication Rounds | 150 |
| Local Epochs | 5 |
| Data Distribution | Non-IID Dirichlet (α = 0.5) |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| DP Epsilon (ε) | 1.5 |
| DP Delta (δ) | 1e-5 |
| Clip Norm | 0.5 |
| Gossip Fanout | 2 |
| Gossip Max Hops | 3 |

---

## Repository Structure

```text
fedavg_dp_baseline/
│
├── client/
├── data/
├── gossip/
├── model/
├── utils/
├── config.yaml
├── main.py
└── experiment.log
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd fedavg_dp_baseline
```

Install dependencies:

```bash
pip install torch torchvision pyyaml numpy pandas flwr-datasets
```

---

## Configuration

The experiment settings are defined in `config.yaml`.

Example:

```yaml
experiment:
  n_clients: 10
  n_rounds: 150
  local_epochs: 5

gossip:
  fanout: 2
  max_hops: 3

dp:
  enabled: true
  clip_norm: 0.5
  epsilon: 1.5
  delta: 1e-5
  auto_noise: true
  base_noise: 0.05

training:
  optimizer: adam
  learning_rate: 0.001
```

---

## Running the Project

Execute:

```bash
python main.py
```

Training logs and evaluation results will be generated during execution.

---

## Running on Google Colab

```python
from google.colab import files

uploaded = files.upload()

!unzip fedavg_dp_baseline.zip
%cd fedavg_dp_baseline

!pip install torch torchvision pyyaml numpy pandas
!pip install flwr-datasets

!python main.py
```

---

## Technologies Used

- Python
- PyTorch
- Federated Learning
- Differential Privacy
- Gossip Protocol
- MNIST Dataset
- YAML Configuration
- Google Colab

---

## Research Motivation

Federated Learning protects user data by keeping it on local devices. However, decentralized communication introduces challenges related to privacy, scalability, and reliable update propagation.

This project investigates a decentralized Federated Learning framework that combines:

- FedAvg for collaborative learning
- Differential Privacy for protecting client updates
- Gossip Protocol for server-free communication

The goal is to study the impact of privacy-preserving mechanisms in decentralized Federated Learning environments.

---

## Author

**Adithi**  
B.Tech, Computer Science and Engineering  
NMAM Institute of Technology (NMAMIT)  
Research Intern, NITK Surathkal

---
