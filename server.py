# server.py
import socket
import pickle
import torch
import time
import select
import matplotlib.pyplot as plt
from model import SimpleCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configuration
NUM_CLIENTS = 2
MODEL_PORT = 9998
UPDATE_PORT = 9999
ROUNDS = 5

# Timeout settings
MODEL_TIMEOUT = 60  # 60 seconds to collect all clients
UPDATE_TIMEOUT = 300  # 5 minutes to receive updates

def average_weights(w_list):
    avg_weights = {}
    for key in w_list[0].keys():
        stacked = torch.stack([w[key] for w in w_list])
        avg_weights[key] = torch.mean(stacked, dim=0)
    return avg_weights

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def main():
    print("[Server] Starting federated learning server...")
    global_model = SimpleCNN()
    acc_history = []

    # Prepare test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create main listening sockets
    model_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    model_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    model_socket.bind(('0.0.0.0', MODEL_PORT))
    model_socket.listen(10)  # Allow backlog of 10 connections
    model_socket.setblocking(False)  # Non-blocking mode
    
    update_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    update_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    update_socket.bind(('0.0.0.0', UPDATE_PORT))
    update_socket.listen(10)
    update_socket.setblocking(False)

    print(f"[Server] Listening on ports {MODEL_PORT} (models) and {UPDATE_PORT} (updates)")

    # Use select for efficient I/O multiplexing
    inputs = [model_socket, update_socket]
    
    for round_num in range(ROUNDS):
        print(f"\n[Server] === Round {round_num + 1}/{ROUNDS} ===")
        start_time = time.time()
        client_weights = []
        
        # --- PHASE 1: Send global model to clients ---
        print(f"[Server] Distributing model to {NUM_CLIENTS} clients...")
        model_connections = []
        model_start = time.time()
        
        while len(model_connections) < NUM_CLIENTS:
            # Check timeout
            if time.time() - model_start > MODEL_TIMEOUT:
                print(f"[Server] Model distribution timeout ({MODEL_TIMEOUT}s)")
                break
                
            # Check for incoming connections
            readable, _, _ = select.select(inputs, [], [], 1.0)
            for s in readable:
                if s is model_socket:
                    try:
                        conn, addr = s.accept()
                        print(f"[Server] Model connection from {addr}")
                        conn.setblocking(True)
                        model_connections.append(conn)
                    except BlockingIOError:
                        continue
                elif s is update_socket:
                    # We're not handling updates in this phase
                    try:
                        conn, addr = s.accept()
                        print(f"[Server] Unexpected update connection from {addr}")
                        conn.close()
                    except:
                        pass
        
        # Send model to connected clients
        global_weights = global_model.state_dict()
        serialized = pickle.dumps(global_weights)
        for conn in model_connections:
            try:
                conn.sendall(serialized)
                print(f"[Server] Model sent to {conn.getpeername()}")
                conn.close()
            except Exception as e:
                print(f"[Server] Error sending model: {e}")
        
        model_time = time.time() - model_start
        print(f"[Server] Model distribution completed in {model_time:.1f}s")
        print(f"[Server] Waiting {MODEL_TIMEOUT//2}s for clients to start training...")
        time.sleep(MODEL_TIMEOUT//2)  # Give clients time to start training
        
        # --- PHASE 2: Receive client updates ---
        print(f"\n[Server] Collecting updates from clients...")
        update_connections = []
        update_start = time.time()
        received_updates = 0
        
        while received_updates < NUM_CLIENTS:
            # Check timeout
            if time.time() - update_start > UPDATE_TIMEOUT:
                print(f"[Server] Update collection timeout ({UPDATE_TIMEOUT}s)")
                break
                
            # Check for incoming connections
            readable, _, _ = select.select(inputs, [], [], 1.0)
            for s in readable:
                if s is update_socket:
                    try:
                        conn, addr = s.accept()
                        print(f"[Server] Update connection from {addr}")
                        conn.setblocking(True)
                        update_connections.append(conn)
                    except BlockingIOError:
                        continue
                elif s is model_socket:
                    # We're not handling models in this phase
                    try:
                        conn, addr = s.accept()
                        print(f"[Server] Unexpected model connection from {addr}")
                        conn.close()
                    except:
                        pass
            
            # Process connections that have data
            for conn in update_connections[:]:
                try:
                    # Check if connection has data
                    ready = select.select([conn], [], [], 0.1)
                    if ready[0]:
                        data = b""
                        while True:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            data += chunk
                            
                        if data:
                            weights = pickle.loads(data)
                            client_weights.append(weights)
                            received_updates += 1
                            print(f"[Server] Received update from {conn.getpeername()} ({received_updates}/{NUM_CLIENTS})")
                        conn.close()
                        update_connections.remove(conn)
                except Exception as e:
                    print(f"[Server] Error receiving update: {e}")
                    conn.close()
                    update_connections.remove(conn)
        
        update_time = time.time() - update_start
        print(f"[Server] Update collection completed in {update_time:.1f}s")
        
        # --- WEIGHT AGGREGATION ---
        if client_weights:
            print(f"[Server] Aggregating {len(client_weights)} updates...")
            global_model.load_state_dict(average_weights(client_weights))
            
            # Test global model
            accuracy = test(global_model, test_loader)
            acc_history.append(accuracy)
            print(f"[Server] Round {round_num+1} accuracy: {accuracy:.2f}%")
        else:
            print("[Server] No updates received this round!")
            acc_history.append(0)  # Placeholder for failed round

        round_time = time.time() - start_time
        print(f"[Server] Round completed in {round_time:.1f} seconds")
        print(f"[Server] Waiting 5s before next round...")
        time.sleep(5)  # Brief pause between rounds

    # Finalize training
    print("\n[Server] Federated learning complete!")
    model_socket.close()
    update_socket.close()
    
    # Save results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, ROUNDS+1), acc_history, 'o-', linewidth=2)
    plt.title('Federated Learning Performance', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, ROUNDS+1))
    plt.ylim(0, 100)
    plt.savefig("fl_results.png", dpi=300, bbox_inches='tight')
    print("[Server] Results plot saved to fl_results.png")

if __name__ == "__main__":
    main()
