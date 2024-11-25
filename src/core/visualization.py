# import numpy as np
#
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
#
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# from loguru import logger
#
#
# class Visualization:
#
#     @staticmethod
#     def visualize(distances, figsize=(10, 5), titles=None):
#         ncols = len(distances)
#         fig, ax = plt.subplots(ncols=ncols, figsize=figsize)
#
#         for i in range(ncols):
#             axes = ax[i] if ncols > 1 else ax
#             distance = distances[i]
#             axes.imshow(distance)
#             axes.set_xticks(np.arange(distance.shape[0]))
#             axes.set_yticks(np.arange(distance.shape[1]))
#             axes.set_xticklabels(np.arange(distance.shape[0]))
#             axes.set_yticklabels(np.arange(distance.shape[1]))
#             for j in range(distance.shape[0]):
#                 for k in range(distance.shape[1]):
#                     text = axes.text(k, j, str(round(distance[j, k].item(), 3)),
#                                      ha="center", va="center", color="w")
#             title = titles[i] if titles and len(titles) > i else "Text Distance"
#             axes.set_title(title, fontsize="x-large")
#
#         fig.tight_layout()
#         plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualization:
    def visualize(self, similarity_matrices: list, titles: list) -> None:
        """
        Visualize the cosine similarity matrices using heatmaps.

        Args:
            similarity_matrices (list): A list of similarity matrices (e.g., CLS and MEAN).
            titles (list): Titles for each similarity matrix.
        """
        num_matrices = len(similarity_matrices)
        fig, axes = plt.subplots(1, num_matrices, figsize=(num_matrices * 6, 6))
        if num_matrices == 1:
            axes = [axes]

        for i, (matrix, ax) in enumerate(zip(similarity_matrices, axes)):
            sns.heatmap(matrix, annot=False, cmap="viridis", ax=ax, square=True)
            ax.set_title(titles[i])
            ax.set_xlabel("Key Phrases")
            ax.set_ylabel("Key Phrases")
        plt.tight_layout()
        plt.show()

