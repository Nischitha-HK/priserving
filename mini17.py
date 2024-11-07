import numpy as np
import matplotlib.pyplot as plt

# Simulated data
num_clients = np.arange(10, 101, 10)  # Number of clients ranging from 10 to 100
grads_per_client = 1000  # Assume each client handles 1000 Grads

# Simulated computational costs (in arbitrary units)
comp_cost_verify_net = num_clients * 5  # Assume VerifyNet scales linearly with number of clients
comp_cost_verifl = num_clients * 6      # VERIFL has slightly more overhead
comp_cost_non_interactive_vfl = num_clients * 3  # Non-interactive VFL has lower overhead

# Simulated communication costs (in kB)
comm_cost_verify_net = num_clients * 15  # VerifyNet and VERIFL have higher communication costs
comm_cost_verifl = num_clients * 16
comm_cost_non_interactive_vfl = num_clients * 10  # Non-interactive VFL has lower communication cost

# Plotting the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot computational costs
ax1.plot(num_clients, comp_cost_verify_net, label='VerifyNet', marker='o', color='blue')
ax1.plot(num_clients, comp_cost_verifl, label='VERIFL', marker='x', color='orange')
ax1.plot(num_clients, comp_cost_non_interactive_vfl, label='Non-interactive VFL', marker='^', color='green')
ax1.set_xlabel('Number of Clients')
ax1.set_ylabel('Computational Cost (arbitrary units)')
ax1.set_title('Computational Cost Comparison')
ax1.legend()
ax1.grid(True)

# Plot communication costs
ax2.plot(num_clients, comm_cost_verify_net, label='VerifyNet', marker='o', color='blue')
ax2.plot(num_clients, comm_cost_verifl, label='VERIFL', marker='x', color='orange')
ax2.plot(num_clients, comm_cost_non_interactive_vfl, label='Non-interactive VFL', marker='^', color='green')
ax2.set_xlabel('Number of Clients')
ax2.set_ylabel('Communication Cost (kB)')
ax2.set_title('Communication Cost Comparison')
ax2.legend()
ax2.grid(True)

# Save the plot as a file
plt.tight_layout()
plt.savefig('protocol_comparison_costs.png')
plt.show()
