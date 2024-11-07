import torch
from torch import nn, optim
import matplotlib.pyplot as plt

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a federated learning client
class FederatedClient:
    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target

    def train(self, epochs=1):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(self.data)
            loss = criterion(output, self.target)
            loss.backward()
            optimizer.step()
        return self.model

# Define a federated learning server
class FederatedServer:
    def __init__(self, model):
        self.model = model

    def aggregate_models(self, models):
        # Aggregate models using simple averaging
        avg_state_dict = models[0].state_dict()
        for key in avg_state_dict.keys():
            avg_state_dict[key] = torch.mean(torch.stack([m.state_dict()[key] for m in models]), dim=0)
        self.model.load_state_dict(avg_state_dict)

    def evaluate(self, data, target):
        with torch.no_grad():
            output = self.model(data)
            loss = nn.MSELoss()(output, target)
        return loss.item()

# Simulate clients and server
def simulate_federated_learning(num_clients, num_rounds, data, target, iid=True):
    global_model = SimpleNN()
    clients = []

    if iid:
        # IID: Distribute data evenly
        data_splits = torch.chunk(data, num_clients)
        target_splits = torch.chunk(target, num_clients)
    else:
        # Non-IID: Assign different data distribution
        data_splits = [data[i::num_clients] for i in range(num_clients)]
        target_splits = [target[i::num_clients] for i in range(num_clients)]

    for i in range(num_clients):
        clients.append(FederatedClient(SimpleNN(), data_splits[i], target_splits[i]))

    server = FederatedServer(global_model)
    accuracies = []

    for round_num in range(num_rounds):
        models = []
        for client in clients:
            trained_model = client.train()
            models.append(trained_model)

        server.aggregate_models(models)
        accuracy = 1 / (1 + server.evaluate(data, target))  # Example metric
        accuracies.append(accuracy)
        print(f"Round {round_num+1}, Accuracy: {accuracy}")

    return accuracies

# Generate dummy data
torch.manual_seed(0)
data = torch.randn(100, 10)
target = torch.randn(100, 1)

# Simulate federated learning
num_clients = 5
num_rounds = 10
iid_accuracies = simulate_federated_learning(num_clients, num_rounds, data, target, iid=True)
non_iid_accuracies = simulate_federated_learning(num_clients, num_rounds, data, target, iid=False)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(iid_accuracies, label='IID Data (Plaintext FL)')
plt.plot(non_iid_accuracies, label='Non-IID Data (Plaintext FL)')
plt.xlabel('Rounds')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy over Rounds (IID vs Non-IID Data)')
plt.legend()
plt.grid(True)
plt.savefig('federated_learning_accuracy_comparison.png')  # Save the plot as a file
