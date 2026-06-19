# Centralized Federated Learning with Differential Privacy

## Overview

This project implements a centralized Federated Learning (FL) framework with Differential Privacy (DP) for privacy-preserving model training.

The system follows the Federated Averaging (FedAvg) algorithm, where multiple clients train locally on their private datasets and send model updates to a central server. Differential Privacy is applied to client updates through gradient clipping and Gaussian noise addition before aggregation, reducing the risk of information leakage from individual training samples.

The implementation is evaluated on the MNIST dataset using a LeNet-style Convolutional Neural Network (CNN) under a non-IID data distribution.

---

## Features

* Centralized Federated Learning architecture
* Federated Averaging (FedAvg) aggregation
* Differential Privacy (DP)

  * Gradient clipping
  * Gaussian noise injection
  * Configurable privacy budget (ε)
* Non-IID client data partitioning using Dirichlet distribution
* LeNet-style CNN model
* Accuracy and privacy monitoring
* Configurable experimental setup through YAML configuration

---

## System Configuration

### Dataset

* MNIST
* Non-IID Dirichlet partitioning (α = 0.5)

### Model

* LeNet-style CNN
* Group Normalization

### Federated Learning

* Number of Clients: 10
* Communication Rounds: 150
* Local Epochs: 5
* Aggregation Method: FedAvg

### Differential Privacy

* Epsilon (ε): 1.0
* Delta (δ): 1e-5
* Clip Norm: 0.5
* Automatic Noise Calibration Enabled

---

## Project Structure

```text
client/
    fl_client.py

data/
    loader.py

model/
    cnn.py

utils/
    weights.py

config.yaml
main.py
README.md
```

## Installation

```bash
pip install torch torchvision datasets pyyaml numpy pandas
```

## Running the Project

```bash
python main.py
```

Training progress, privacy statistics, and global model accuracy will be displayed in the console and written to the log file.

---

## Workflow

1. Server initializes the global model.
2. Global model weights are distributed to all clients.
3. Clients perform local training.
4. Differential Privacy is applied using clipping and Gaussian noise.
5. Clients send privacy-protected model updates to the server.
6. Server performs FedAvg aggregation.
7. Global model is updated and redistributed for the next round.

---

## Experimental Setup

| Parameter     | Value |
| ------------- | ----- |
| Dataset       | MNIST |
| Clients       | 10    |
| Rounds        | 150   |
| Local Epochs  | 5     |
| Alpha         | 0.5   |
| Epsilon       | 1.0   |
| Delta         | 1e-5  |
| Clip Norm     | 0.5   |
| Optimizer     | Adam  |
| Learning Rate | 0.001 |

---

## Research Context

This repository represents Approach 1 of a privacy-preserving Federated Learning study:

**Approach 1:** FedAvg + Differential Privacy

Additional approaches extend this baseline with:

* Post-Quantum Cryptography (Dilithium)
* Zero-Knowledge Proof (ZKP) based update validation

---

## Author

Adithi

B.Tech Computer Science and Engineering

NMAM Institute of Technology (NMAMIT)

Research Internship – NITK Surathkal
