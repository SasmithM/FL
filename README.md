Federated Learning System

Overview
This project implements a Federated Learning (FL) system, a distributed machine learning approach that enables multiple clients to collaboratively train a shared model while keeping their data localized. Unlike traditional centralized machine learning, federated learning preserves data privacy by allowing devices or clients to train models on their own data without sharing the raw data with a central server.

In this implementation, a server coordinates training across multiple clients, aggregating model updates while maintaining data privacy. This approach is particularly valuable for applications requiring data privacy, such as healthcare, finance, and mobile device analytics.

Project Setup :

Architecture

FL-Server: Coordinates training rounds and aggregates model updates

FL-Client1 & FL-Client2: Local training nodes with private datasets

System Requirements
  Ubuntu 20.04/22.04
  Python 3.8+
  Minimum 2GB RAM per VM
  Bridged Network b/w VMs

