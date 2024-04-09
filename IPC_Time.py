import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('figure_gif.csv')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot IPC vs Time for both cores
ax.plot(df['Time_03'], df['IPC_03'], label='P (big) core', color='purple')
ax.plot(df['Time_19'], df['IPC_19'], label='E (small) core', color='green')

# Set labels, title, and legend
ax.set_xlabel('Time (in secs)', fontsize=14, labelpad=20)
ax.set_ylabel('Instructions per Cycle (IPC)', fontsize=14, labelpad=20)  # Increase labelpad value to adjust distance
ax.set_title('IPC vs Time', fontsize=16)
ax.legend()
ax.grid(True)

# Set font size for ticks
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Save plot as PNG
plt.savefig('ipc_vs_time_final.png')
# Set the x-axis limits to start from 0
plt.xlim(left=0)
plt.show()
