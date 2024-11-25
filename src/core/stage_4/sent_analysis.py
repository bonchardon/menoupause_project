# todo answer questions:
#  1) factors were driving positive sentiment in the conversations;
#  2) the main causes of negative sentiment;
#  3) notable patterns in sentiment across different topics

from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd

from src.core.preprocess_data import Preprocess


class SentimentAnalysis:

    async def data_keys(self):
        """
        Fetches and returns the key phrases (treatment options) from preprocessed data.
        """
        data: pd.DataFrame = await Preprocess().fully_preprocessed_data()
        key_phrases: DataFrame = data[['Sentiment', 'Key Phrases']]
        return key_phrases

    async def possitive_sentiment(self):
        data = await self.data_keys()
        positive_data = data[data['Sentiment'] == 'Negative']

        vectorizer = TfidfVectorizer(max_features=50)
        X = vectorizer.fit_transform(positive_data['Key Phrases'])
        terms = vectorizer.get_feature_names_out()

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
            ' '.join(positive_data['Key Phrases']))

        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Keywords Driving Negative Sentiment")
        plt.show()


