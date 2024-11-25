# todo:
#  1) main topics discussed in the online community;
#  2) key themes emerging from the conversation (NER).

import gensim
from gensim import corpora

from nltk.tokenize import word_tokenize
import nltk

from loguru import logger

import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from src.core.preprocess_data import Preprocess
from src.core.consts import DIRECTORY, FILE_PLACE, TOPIC, WEIGHT, TERM

nltk.download('punkt')


class GeneralDiscussionAnalysis:

    async def lda_topics(self) -> DataFrame:
        try:
            df: DataFrame = await Preprocess.fully_preprocessed_data()
            tokenized_docs = [word_tokenize(str(doc)) for doc in df['Key Phrases']]
            dictionary = corpora.Dictionary(tokenized_docs)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
            lda_model = gensim.models.LdaMulticore(corpus, num_topics=3, id2word=dictionary, passes=10)

            topic_terms = []
            for idx, topic in enumerate(lda_model.print_topics(num_words=10), start=1):
                topic_string = topic[1]
                topic_data = topic_string.split(' + ')
                for term in topic_data:
                    weight, word = term.split('*')
                    topic_terms.append([idx, word.strip('"'), float(weight)])
            lda_df: DataFrame = pd.DataFrame(topic_terms, columns=[TOPIC, TERM, WEIGHT])
            return lda_df

        except Exception as e:
            logger.error(f'Error in LDA topics: {e!r}')
