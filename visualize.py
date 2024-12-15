import matplotlib.pyplot as plt

# data = [
#     (0, 0, True),
#     (0.1, 0.00140, False),
#     (0.15, 0.00289, True),
#     (0.2, 0.00487, True),
#     (0.3, 0.011771, True),
#     (0.4, 0.01941, True),
#     (0.5, 0.02889, True),
#     (0.6, 0.04493, True),
#     (0.7, 0.06113, True),
#     (0.8, 0.07728, True),
#     (0.9, 0.09669, True),
# ]

# x_values, y_values, detected = zip(*data)

# plt.figure(figsize=(8, 5))

# colors = ['g' if d else 'r' for d in detected]

# plt.scatter(x_values, y_values, c=colors, label='MMD Value', edgecolors='k')

# plt.title('MMD Value vs Adversarial Ratio', fontsize=14)
# plt.xlabel('Adversarial Examples Ratio', fontsize=12)
# plt.ylabel('MMD Value', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)

# import matplotlib.lines as mlines
# green_patch = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='g', markersize=10, label='Right Detected')
# red_patch = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='r', markersize=10, label='Wrong Detected')

# plt.legend(handles=[green_patch, red_patch], fontsize=10)

# # Show the plot
# plt.tight_layout()
# plt.show()

samples_pgd = [1300, 1000, 800, 500, 200, 100] 
correct_pgd = [9, 9, 8, 7, 5, 2] 

samples_fgsm = [1300, 1000, 800, 500, 200, 100]  
correct_fgsm = [8, 8, 8, 7, 4, 2]  

plt.figure(figsize=(8, 5))

plt.plot(samples_pgd, correct_pgd, marker='o', linestyle='-', color='g', label='PGD')

plt.plot(samples_fgsm, correct_fgsm, marker='o', linestyle='-', color='r', label='FGSM')

plt.title('Comparison of PGD and FGSM Correct Detections', fontsize=14)
plt.xlabel('Number of Samples', fontsize=12)
plt.ylabel('Correct Detections', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()