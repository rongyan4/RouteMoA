import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['PLCC', 'SRCC', 'Top-3 AR']
independent = [0.443, 0.526, 0.606]
reference_based = [0.647, 0.644, 0.776]

x = np.arange(len(metrics))  # the label locations

# # Set x-axis position
# group_width = 0.8  # Total width per group
# inner_gap = 0.4    # Bar spacing within group (increase this value to separate the two bars)
# x = np.arange(len(metrics)) * (group_width + inner_gap)  # Interval per group

width = 0.3  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 5))


# Create bars
rects1 = ax.bar(x - width/2-0.03, independent, width, label='Independent Answering', color='#FFC9B6', edgecolor='black', linewidth=0.7)
rects2 = ax.bar(x + width/2+0.03, reference_based, width, label='Reference-based Summarization', color='#C2D7FA', edgecolor='black', linewidth=0.7)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(bottom=0.3, top=0.85)
ax.set_ylabel('Score', fontsize=16)
# ax.set_title('Evaluation Metrics Comparison', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=16)
ax.legend(fontsize=15)

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=13)

autolabel(rects1)
autolabel(rects2)

# Adjust layout
plt.tight_layout()
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save as vector graphics (SVG and PDF)
plt.savefig('srcc.svg', format='svg', bbox_inches='tight')
plt.savefig('srcc.pdf', format='pdf', bbox_inches='tight')

# plt.show()