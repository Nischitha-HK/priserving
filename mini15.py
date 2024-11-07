import numpy as np
import matplotlib.pyplot as plt

# Simulated data for clients' communication costs and verification costs
num_clients = 10
client_ids = np.arange(1, num_clients + 1)
communication_costs = np.random.uniform(200, 500, num_clients)  # Total communication costs in kB
verification_costs = np.random.uniform(0.1, 0.3, num_clients) * communication_costs  # Verification costs as a fraction of total

# Calculate non-verification costs
non_verification_costs = communication_costs - verification_costs

# Plotting the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot total communication costs vs. verification costs
ax1.bar(client_ids, communication_costs, label='Total Communication Cost', alpha=0.7, color='blue')
ax1.bar(client_ids, verification_costs, label='Verification Cost', alpha=0.7, color='orange')
ax1.set_xlabel('Client ID')
ax1.set_ylabel('Communication Cost (kB)')
ax1.set_title('Client Communication Costs')
ax1.legend()
ax1.grid(True)

# Plot total Grads (representative of computation tasks) and verification costs
grads_count = np.random.uniform(50000, 100000, num_clients)  # Simulated number of Grads
ax2.bar(client_ids, grads_count, label='Number of Grads', alpha=0.7, color='green')
ax2.plot(client_ids, verification_costs, label='Verification Cost (kB)', color='red', marker='o')
ax2.set_xlabel('Client ID')
ax2.set_ylabel('Number of Grads')
ax2.set_title('Verification Costs vs. Number of Grads')
ax2.legend(loc='upper left')
ax2.grid(True)

# Save the plot as a file
plt.tight_layout()
plt.savefig('client_communication_verification_costs.png')
