from re import sub
from csv import DictReader

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

import numpy as np
import pandas as pd
from pandas import DataFrame

from loguru import logger

from src.core.consts import DATA_SOURCE, KEY_PHRASES

np.random.seed(42)


class Preprocess:

    @staticmethod
    def lemmas(text_data: str) -> list[str]:
        try:
            word_lemmas: WordNetLemmatizer = WordNetLemmatizer()
            words: list[str] = word_tokenize(text_data)
            return [word_lemmas.lemmatize(word) for word in words]
        except ValueError as e:
            logger.error(f'Error in lemmatizing: {e!r}')

    @staticmethod
    def get_rid_of_punctuation_and_nan(df: pd.DataFrame) -> list[str]:
        try:
            # key_phrases_data = []
            # with open(DATA_SOURCE, 'r') as file:
            #     csv_reader = DictReader(file)
            #     for row in csv_reader:
            #         text_data: str = row[KEY_PHRASES]
            #         text_data: str = str(text_data)
            #         text_data: str = text_data.replace('nan', '').strip()
            #         cleaned_text: str = sub(r'[^\w\s]', ' ', text_data)
            #         sentences: list[str] = cleaned_text.splitlines()
            #         sentences: list[str] = [sentence.strip() for sentence in sentences if sentence.strip() != '']
            #         key_phrases_data.extend(sentences)
            # return key_phrases_data
            df_cleaned = df[df[KEY_PHRASES].notna() & (df[KEY_PHRASES].str.strip() != '')]
            df_cleaned[KEY_PHRASES] = df_cleaned[KEY_PHRASES].apply(
                lambda text: sub(r'[^\w\s]', ' ', str(text)).strip()
            )
            return df_cleaned[KEY_PHRASES].tolist()
        except ValueError as e:
            logger.error(f'Error in removing punctuation: {e!r}')

    @staticmethod
    async def stop_words_removal(word_list: list[str]) -> list[str]:
        stop_words: set[str] = set(stopwords.words('english'))
        return [word for word in word_list if word not in stop_words]

    @classmethod
    async def fully_preprocessed_data(cls) -> DataFrame:
        try:
            df: DataFrame = pd.read_csv(DATA_SOURCE)
            cleaned_texts = cls.get_rid_of_punctuation_and_nan(df)
            # if len(cleaned_texts) != len(df):
            #     logger.error(f"Mismatch in data length: {len(cleaned_texts)} vs {len(df)}")
            #     return df
            lemmatized_texts = [cls.lemmas(text) for text in cleaned_texts]
            stopwords_removed = [await cls.stop_words_removal(text) for text in lemmatized_texts]
            df_cleaned = df[df[KEY_PHRASES].notna() & (df[KEY_PHRASES].str.strip() != '')].copy()
            df_cleaned[KEY_PHRASES] = [' '.join(text) for text in stopwords_removed]
            return df_cleaned
        except (ValueError, IndexError) as e:
            logger.error(f'Error in preprocessing data: {e!r}')
