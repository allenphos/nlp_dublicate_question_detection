import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

class TopWordsCoverage:
    """
    Analyze cumulative coverage of top words in cleaned texts.
    """
    @staticmethod
    def plot_coverage(cleaned_texts, thresholds=[0.9, 0.95, 0.99], plot_limit=10000):
        # Combine all cleaned texts into a single string and split into words
        all_words = " ".join(cleaned_texts).split()
        # Count occurrences of each word
        word_counts = Counter(all_words)
        # Sort counts in descending order
        sorted_counts = sorted(word_counts.values(), reverse=True)
        # Calculate the total number of words
        total = sum(sorted_counts)
        # Compute cumulative sum (fraction of total) for coverage curve
        cum_sum = np.cumsum(sorted_counts) / total

        # Create a figure for the plot
        plt.figure(figsize=(10, 5))
        # Plot cumulative coverage curve for top words (up to plot_limit)
        plt.plot(cum_sum[:plot_limit])
        # Draw vertical lines where each threshold (e.g., 90%, 95%, 99%) is reached
        for thresh in thresholds:
            idx = np.argmax(cum_sum >= thresh)
            plt.axvline(x=idx, linestyle='--', label=f"{int(thresh*100)}% at {idx} words")
        # Add plot title and labels
        plt.title("Cumulative Word Coverage by Frequency")
        plt.xlabel("Number of Most Frequent Words")
        plt.ylabel("Fraction of All Tokens")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Print out how many words are needed to cover each threshold
        for thresh in thresholds:
            idx = np.argmax(cum_sum >= thresh)
            print(f"{int(thresh*100)}% of the corpus is covered by {idx} words.")
