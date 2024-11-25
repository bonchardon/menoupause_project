# todo:
#  1) what treatment options were discussed;
#  2) which ones received most attention.

# sequence classification (for task 4 + analysis) is for sentiment analysis!
# BUT here I need to use NER identification algos
# todo: apply f1, confusion matrix tp evaluate the outcomes and make conclusions upon that

from loguru import logger
from accelerate import Accelerator

from pandas.core.interchange.dataframe_protocol import DataFrame

import numpy as np

import tensorflow as tf

from transformers import (AutoTokenizer, AutoModel, BertForTokenClassification, BertTokenizer,
                          TFBertForTokenClassification, BatchEncoding,
                          TFTrainingArguments, Trainer, TrainingArguments, DataCollatorForTokenClassification)

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.core.preprocess_data import Preprocess
from src.core.consts import BERT_MODE_Bio


class TreatmentOptions:

    async def X_y_data(self):
        try:
            data: DataFrame = await Preprocess().fully_preprocessed_data()
            df: DataFrame = data[['Sentiment', 'Key Phrases']]

            X: list = list(df['Key Phrases'])
            y: list = df['Sentiment']

            sentiment_map = {
                'Neutral': 0,
                'Positive': 1,
                'Negative': -1
            }
            y = y.map(sentiment_map)
            y = y.to_numpy()
            return X, y
        except Exception as e:
            logger.error(f'There is an error. Here you can find more info: {e!r}')

    async def bio_ner(self):
        try:
            # todo: upload a pre-trained model
            # todo: fine-tune a model
            # todo: ensure there are no issues with the model itself (such as over-fitting, under-fitting),
            #  since training data isn't that massive

            X, y = await self.X_y_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            accelerator = Accelerator()

            tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
            model = TFBertForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', from_pt=True)

            # train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(model)

            train_encoding: BatchEncoding = tokenizer(X_train, truncation=True, padding=True, return_tensors='tf', max_length=512)
            test_encoding: BatchEncoding = tokenizer(X_test, truncation=True, padding=True, return_tensors='tf', max_length=512)

            logger.info(f'Train Encoding: {train_encoding['input_ids'].shape}')
            logger.info(f'Test Encoding: {test_encoding['input_ids'].shape}')

            train_labels = []
            test_labels = []

            for idx, _ in enumerate(X_train):
                input_ids = train_encoding['input_ids'][idx]
                attention_mask = train_encoding['attention_mask'][idx]
                label = y_train[idx]
                label_seq = [label if mask == 1 else -100 for mask in attention_mask]
                train_labels.append(label_seq)

            for idx, _ in enumerate(X_test):
                input_ids = test_encoding['input_ids'][idx]
                attention_mask = test_encoding['attention_mask'][idx]
                label = y_test[idx]
                label_seq = [label if mask == 1 else -100 for mask in attention_mask]
                test_labels.append(label_seq)

            # Convert labels to numpy arrays
            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)

            # Step 4: Create TensorFlow datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encoding), y_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encoding), y_test))

            logger.info(f'Train dataset: {train_dataset}')
            logger.info(f'Test dataset: {test_dataset}')

            train_dataset = train_dataset.batch(4)
            test_dataset = test_dataset.batch(8)

            training_args = TFTrainingArguments(
                output_dir='./results',
                num_train_epochs=2,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                learning_rate=3e-5,
                use_cpu=True,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                # data_collator=data_collator,
                # accelerator=accelerator
            )
            trainer.train()
            trainer.evaluate(dict(test_dataset))
            output = trainer.predict(test_dataset)

            logger.info(f'Predictions shape: {output.predictions.shape}')
            logger.info(f'Label IDs shape: {output.label_ids.shape}')

            # preds = np.argmax(output.predictions, axis=-1)

            # preds_flat = preds.flatten()
            # y_test_flat = y_test.flatten()
            #
            # cm = confusion_matrix(y_test_flat, preds_flat)
            # logger.info(f"Confusion Matrix:\n{cm}")
            # return cm

        except Exception as e:
            logger.error(f'There is an issue. More details provided: {e!r}')

    # async def bio_ner(self):
    #     try:
    #         # Await the X_y_data coroutine to get X_train, y_train, X_test, and y_test
    #         X, y = await self.X_y_data()
    #
    #         # Split the data into training and testing sets
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    #         # Load the tokenizer and model for NER
    #         tokenizer = AutoTokenizer.from_pretrained(BERT_MODE_Bio_ClinicalL)
    #         model = AutoModel.from_pretrained(BERT_MODE_Bio_ClinicalL)
    #
    #         # Tokenize the data
    #         train_encoding = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    #         test_encoding = tokenizer(X_test, truncation=True, padding=True, max_length=512)
    #
    #         # Ensure tokenized data is valid
    #         assert len(train_encoding['input_ids']) == len(y_train), "Mismatch between X and y lengths"
    #         assert len(test_encoding['input_ids']) == len(y_test), "Mismatch between X and y lengths"
    #
    #         train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encoding), y_train))
    #         test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encoding), y_test))
    #         training_args = TFTrainingArguments(
    #             output_dir='./results',
    #             num_train_epochs=2,
    #             per_device_train_batch_size=4,
    #             per_device_eval_batch_size=8,
    #             warmup_steps=500,
    #             weight_decay=0.01,
    #             logging_dir='./logs',
    #             logging_steps=10
    #         )
    #         with training_args.strategy.scope():
    #             model = AutoModel.from_pretrained(BERT_MODE_Bio_ClinicalL)
    #         trainer = Trainer(
    #             model=model,
    #             args=training_args,
    #             train_dataset=train_dataset,
    #             eval_dataset=test_dataset
    #         )
    #         trainer.train()
    #         trainer.evaluate(test_dataset)
    #
    #         # TODO: Count percentage of those treatment options that received the most attention
    #         output = trainer.predict(test_dataset)
    #         cm = confusion_matrix(y_test, output)
    #         return cm
    #
    #     except Exception as e:
    #         logger.error(f'There is an issue. More details provided: {e!r}')

