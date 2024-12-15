import matplotlib.pyplot as plt

# Data with detected information (True/False for color mapping)
data = [
    (0, 0, True),
    (0.1, 0.00140, False),
    (0.15, 0.00289, True),
    (0.2, 0.00487, True),
    (0.3, 0.011771, True),
    (0.4, 0.01941, True),
    (0.5, 0.02889, True),
    (0.6, 0.04493, True),
    (0.7, 0.06113, True),
    (0.8, 0.07728, True),
    (0.9, 0.09669, True),
]

# Unzip the data into separate lists for x (adversarial ratio), y (MMD value), and detected (True/False)
x_values, y_values, detected = zip(*data)

# Plot the graph with color coding based on detected (True/False)
plt.figure(figsize=(8, 5))

# Use a list comprehension to assign colors based on detection (True -> Green, False -> Red)
colors = ['g' if d else 'r' for d in detected]

# Plot the data
plt.scatter(x_values, y_values, c=colors, label='MMD Value', edgecolors='k')

# Set titles and labels
plt.title('MMD Value vs Adversarial Ratio', fontsize=14)
plt.xlabel('Adversarial Examples Ratio', fontsize=12)
plt.ylabel('MMD Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Custom legend
import matplotlib.lines as mlines
green_patch = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='g', markersize=10, label='Right Detected')
red_patch = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='r', markersize=10, label='Wrong Detected')

plt.legend(handles=[green_patch, red_patch], fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()
