# Federated Learning with Robust Aggregation

This project implements a basic Federated Learning (FL) simulation using socket programming and PyTorch.
It includes a defense mechanism using Trimmed Mean Aggregation to counter model poisoning attacks.

## Files

- `model.py`: Defines the neural network architecture.
- `train_eval.py`: Contains functions for training and evaluation.
- `server.py`: Runs the FL server which coordinates training and aggregates models.
- `client.py`: Runs an FL client that trains on local data and communicates with the server.
- `README.md`: This file.

## Requirements

- Python 3.8+
- PyTorch
- tqdm
- NumPy

## Running the Simulation

1. Start the server:
   ```bash
   python server.py
   ```

2. Start clients (in separate terminals or VMs):
   ```bash
   python client.py
   ```

## Notes

- Non-IID MNIST is used for training across clients.
- Clients communicate using sockets over specified ports.
- Aggregation strategy can be changed in `server.py`.
