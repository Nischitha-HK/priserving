import numpy as np
import matplotlib.pyplot as plt

# Simulated data
num_clients = np.arange(10, 101, 10)  # Number of clients ranging from 10 to 100
grads_per_client = 1000  # Assume each client handles 1000 Grads
total_grads = grads_per_client * num_clients

# Assuming communication costs and verification costs scale linearly with the number of clients
base_comm_cost_per_client = 5.0  # in kB, base communication cost per client
verification_cost_constant = 14.45  # constant verification cost in kB for AggS

total_comm_costs = base_comm_cost_per_client * num_clients
verification_costs = np.full(num_clients.shape, verification_cost_constant)

# Plotting the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Total communication costs and verification costs with number of clients
ax1.plot(num_clients, total_comm_costs, label='Total Communication Cost (kB)', marker='o', color='blue')
ax1.plot(num_clients, verification_costs, label='Verification Cost (kB)', marker='x', color='orange')
ax1.set_xlabel('Number of Clients')
ax1.set_ylabel('Cost (kB)')
ax1.set_title('Total Communication vs Verification Cost')
ax1.legend()
ax1.grid(True)

# Verification cost with number of clients and Grads
ax2.plot(num_clients, verification_costs, label='Verification Cost (kB)', marker='x', color='orange')
ax2.set_xlabel('Number of Clients')
ax2.set_ylabel('Verification Cost (kB)')
ax2.set_title('Verification Cost vs Number of Clients')
ax2.grid(True)
ax2.legend()

# Save the plot as a file
plt.tight_layout()
plt.savefig('verification_communication_costs.png')
plt.show()
