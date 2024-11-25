# todo:
#  1) what treatment options were discussed;
#  2) which ones received most attention

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from loguru import logger

import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.preprocess_data import Preprocess
from src.core.consts import BERT_MODEL_Bio_ClinicalL, BERT_MODE_Bio


class TreatmentVars:

    async def data_keys(self):
        """
        Fetches and returns the key phrases (treatment options) from preprocessed data.
        """
        data: pd.DataFrame = await Preprocess().fully_preprocessed_data()
        key_phrases: list = data['Key Phrases'].values.tolist()
        return key_phrases

    async def treatment_frequency(self):
        """
        Computes and returns the frequency of treatment options discussed in the dataset.
        """
        data_keys = await self.data_keys()
        data_keys_split = [item.split() for item in data_keys]
        key_phrases = [word for sublist in data_keys_split for word in sublist]
        frequency = Counter(key_phrases)
        sorted_frequency = frequency.most_common()

        logger.info(f"Most common treatment options: {sorted_frequency}")
        return sorted_frequency

    async def bert_similarity(self):
        """
        Calculates and returns the BERT embeddings for the treatment options.
        """
        tokenizer = BertTokenizer.from_pretrained(BERT_MODE_Bio)
        model = BertModel.from_pretrained(BERT_MODE_Bio)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        data_keys = await self.data_keys()
        encodings = tokenizer(data_keys, padding=True, return_tensors='pt')

        logger.info(f'Encodings: {encodings.keys()}')
        logger.info([f'{tokens} ==> {tokenizer.convert_ids_to_tokens(tokens)}' for tokens in encodings['input_ids']])
        with torch.no_grad():
            embeddings = model(**encodings)[0]

        logger.info(f'Embedding shape: {embeddings.shape}')
        return embeddings

    async def compute_similarity_and_rank(self):
        """
        Computes cosine similarity between treatment options based on BERT embeddings,
        and ranks them by both frequency and semantic similarity.
        """
        embeddings = await self.bert_similarity()

        cls_embeddings = embeddings[:, 0, :]
        normalized_cls = F.normalize(cls_embeddings, p=2, dim=1)

        cls_dist = normalized_cls.matmul(normalized_cls.T)
        cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
        cls_dist = cls_dist.cpu().numpy()

        cls_similarity = cosine_similarity(cls_dist)

        treatment_freq = await self.treatment_frequency()
        treatments, frequencies = zip(*treatment_freq)

        top_n = 10
        top_treatments = treatments[:top_n]
        related_treatments = self.get_related_treatments(cls_similarity, top_treatments)

        logger.info(f"Top related treatments based on BERT similarity: {related_treatments}")
        self.visualize_top_treatments(top_treatments, frequencies[:top_n], related_treatments)

    def get_related_treatments(self, similarity_matrix, top_treatments):
        """
        Given the cosine similarity matrix and the list of top treatments,
        return the most related treatments.
        """
        num_treatments = len(top_treatments)
        related_treatments = {}
        for idx in range(num_treatments):
            similarity_scores = similarity_matrix[idx]
            related_indices = similarity_scores.argsort()[-4:-1][::-1]
            valid_related_indices = [i for i in related_indices if i < num_treatments]
            related_treatments[top_treatments[idx]] = [top_treatments[i] for i in valid_related_indices]

        return related_treatments

    def visualize_top_treatments(self, treatments, frequencies, related_treatments):
        """
        Visualizes the top treatments and their most related treatments.
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(treatments), y=list(frequencies), palette='Blues_d')
        plt.xticks(rotation=45)
        plt.title('Top 10 Most Discussed Treatments')
        plt.xlabel('Treatment Option')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        for treatment, related in related_treatments.items():
            print(f"Treatment: {treatment}")
            print(f"Related Treatments: {', '.join(related)}\n")

    async def combine_visualization(self) -> None:
        """
        Combines the treatment frequency analysis and the BERT-based similarity visualization.
        """
        await self.compute_similarity_and_rank()
