# train_eval.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def train(model, loader, optimizer, epochs=1):
    model.train()
    total_loss, total_samples = 0.0, 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / total_samples
        print(f"  Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    return model.state_dict()

def get_client_loader(client_id, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Split data between clients
    if client_id == 1:
        indices = list(range(0, 30000))
    elif client_id == 2:
        indices = list(range(30000, 60000))
    else:
        raise ValueError("Invalid client ID")
    
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
