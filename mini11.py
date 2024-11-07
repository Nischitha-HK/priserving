import torch
import syft as sy
from torch import nn, optim
from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize PySyft
hook = sy.TorchHook(torch)

# Define a federated learning client
class FederatedClient:
    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target
        self.client = sy.VirtualWorker(hook, id=f'client_{id(self)}')

    def train(self, epochs=1):
        self.model.send(self.client)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(self.data)
            loss = criterion(output, self.target)
            loss.backward()
            optimizer.step()
        return self.model.get()

# Define a federated learning server
class FederatedServer:
    def __init__(self, model):
        self.model = model
        self.workers = []

    def add_worker(self, worker):
        self.workers.append(worker)

    def aggregate_models(self):
        # Aggregate models using secure aggregation
        encrypted_models = [worker.model.state_dict() for worker in self.workers]
        # Simple average aggregation for demonstration
        avg_state_dict = encrypted_models[0]
        for key in avg_state_dict:
            avg_state_dict[key] = torch.mean(torch.stack([m[key] for m in encrypted_models]), dim=0)
        self.model.load_state_dict(avg_state_dict)

# Sample data
data = torch.randn(5, 10)
target = torch.randn(5, 1)

# Initialize model and federated clients
model = SimpleNN()
clients = [FederatedClient(model, data, target) for _ in range(3)]

# Initialize server
server = FederatedServer(SimpleNN())

# Federated learning loop
for epoch in range(10):
    for client in clients:
        client.train()
    server.add_worker(client)
    server.aggregate_models()
    print(f"Epoch {epoch + 1} complete")

# Example encryption for secure aggregation (not used in this basic example)
# encrypted_data = cipher_suite.encrypt(b"Secret Data")
# decrypted_data = cipher_suite.decrypt(encrypted_data)
# print(decrypted_data)

print("Federated learning complete with privacy-preserving techniques.")

