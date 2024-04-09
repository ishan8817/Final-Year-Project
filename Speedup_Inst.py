import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the dataset
dataset = pd.read_csv("Merged_IPC_6.csv")

# Plot Cumulative Instructions vs Speed_up
plt.figure(figsize=(14, 8))  # Larger figure size
plt.plot(dataset['Cumulative Instructions'], dataset['Speed_up'], color='skyblue')
plt.xlabel('Cumulative Instructions', fontsize=16)
plt.ylabel('Speedup Factor', fontsize=16)

# Format x-axis ticks as scientific notation with 10^11
def format_ticks(x, _):
    if int(x/1e11) == 0:
        return f'{int(x)}'
    else:
        return f'{int(x/1e11)}x10^11'

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

# Set the x-axis limits to start from 0
plt.xlim(left=0)

plt.title('Cumulative Instructions vs Speedup Factor', fontsize=16)
plt.grid(True)

# Set font size for all labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot as an image file
plt.savefig('cumulative_instructions_vs_speedup_3.png', dpi=300)  # Adjust dpi for higher resolution

# Display the plot
plt.show()
