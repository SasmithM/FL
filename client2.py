# client.py
import socket
import pickle
import torch
import time
from model import SimpleCNN
from train_eval import train, get_client_loader

# Configuration
SERVER_IP = '192.168.1.161'  # Server IP address
MODEL_PORT = 9998            # Port for receiving models
UPDATE_PORT = 9999           # Port for sending updates
CLIENT_ID = 2                # Set to 1 for client1, 2 for client2

# Connection settings
CONNECT_TIMEOUT = 60         # Increased timeout to 60 seconds
RECV_TIMEOUT = 120

def connect_with_retry(ip, port, max_attempts=5, delay=5):
    """Connect to server with retry mechanism"""
    for attempt in range(1, max_attempts+1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CONNECT_TIMEOUT)
            print(f"  Connecting to {ip}:{port} (attempt {attempt}/{max_attempts})")
            sock.connect((ip, port))
            return sock
        except socket.timeout:
            print(f"  Connection attempt {attempt} timed out")
        except Exception as e:
            print(f"  Connection error: {e}")
        time.sleep(delay)  # Wait before retrying
    return None

def main():
    print(f"\n{'='*50}")
    print(f"[Client {CLIENT_ID}] Starting Federated Learning Client")
    print(f"{'='*50}")
    
    # Initialize model and data
    model = SimpleCNN()
    data_loader = get_client_loader(CLIENT_ID, batch_size=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for round_num in range(1, 6):
        round_start = time.time()
        print(f"\n[Client {CLIENT_ID}] === Round {round_num}/5 ===")
        
        # --- RECEIVE GLOBAL MODEL ---
        print(f"[Client {CLIENT_ID}] Requesting global model...")
        sock = connect_with_retry(SERVER_IP, MODEL_PORT, max_attempts=10, delay=3)
        model_data = b""
        
        if sock:
            try:
                sock.settimeout(RECV_TIMEOUT)
                print("  Receiving model data...")
                start_time = time.time()
                
                while True:
                    try:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        model_data += chunk
                        # Break if we've been receiving for too long
                        if time.time() - start_time > 120:
                            break
                    except socket.timeout:
                        break
                    except Exception:
                        break
                
                if model_data:
                    model.load_state_dict(pickle.loads(model_data))
                    print(f"  Model received ({len(model_data)/1024:.1f} KB)")
                else:
                    print("  Received empty model data")
            except Exception as e:
                print(f"  Model receive error: {e}")
            finally:
                sock.close()
        else:
            print("  Model connection failed, skipping round")
            continue

        # --- LOCAL TRAINING ---
        print(f"[Client {CLIENT_ID}] Starting local training...")
        train_start = time.time()
        updated_weights = train(model, data_loader, optimizer)
        train_time = time.time() - train_start
        print(f"  Training completed in {train_time:.1f} seconds")

        # --- SEND UPDATES TO SERVER ---
        print(f"[Client {CLIENT_ID}] Sending updates to server...")
        sock = connect_with_retry(SERVER_IP, UPDATE_PORT, max_attempts=10, delay=3)
        
        if sock:
            try:
                sock.settimeout(RECV_TIMEOUT)
                serialized = pickle.dumps(updated_weights)
                print(f"  Sending update ({len(serialized)/1024:.1f} KB)")
                sock.sendall(serialized)
                print("  Update sent successfully")
            except Exception as e:
                print(f"  Update send error: {e}")
            finally:
                sock.close()
        else:
            print("  Update connection failed")

        round_time = time.time() - round_start
        print(f"[Client {CLIENT_ID}] Round completed in {round_time:.1f} seconds")
        
        # Short pause between rounds
        time.sleep(1)

    print(f"\n[Client {CLIENT_ID}] All rounds completed")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
