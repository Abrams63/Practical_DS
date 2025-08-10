import matplotlib.pyplot as plt
import numpy as np
import collections


def plot_distribution(data):
    """
    Plots the distribution of data using a bar chart.

    Parameters:
    data (array-like): An array of categorical data items.
    """

    counts = collections.Counter(data)
    categories = list(counts.keys())
    frequencies = [counts[cat] for cat in categories]
    
    fig, ax = plt.subplots()
    
    ax.bar(categories, frequencies, color=['blue', 'green', 'red'])
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of Categorical Data')
    
    return fig

# Example data
data = np.random.choice(['A', 'B', 'C'], size=100)
plot_distribution(data)
