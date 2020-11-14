import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K
from bert_features import BertFeaturizer
from bert_layer import BertLayer
import joblib
import pandas as pd


class BERT(object):
    def __init__(self, bert_path, max_seq_length=64, bert_tune_layers=10):
        self.max_seq_length = max_seq_length
        self.bert_tune_layers = bert_tune_layers
        self.bert_path = bert_path
        self.session = tf.Session()
        self.featurizer = BertFeaturizer(self.bert_path, self.session, max_seq_length=max_seq_length)
        self.model = self.__build_model()
        self.__initialize_vars(self.session)

    def __initialize_vars(self, session):
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        K.set_session(session)

    def __build_model(self):
        in_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]

        bert_output = BertLayer(n_fine_tune_layers=self.bert_tune_layers)(bert_inputs)
        dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
        pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

        model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        return model

    def fit(self, texts, labels, validation_ratio=0.1):
        val_texts = texts[:int(validation_ratio*len(texts))]
        train_texts = texts[int(validation_ratio * len(texts)):]
        val_labels = labels[:int(validation_ratio * len(texts))]
        train_labels = labels[int(validation_ratio * len(texts)):]

        train_input_ids, train_input_masks, train_segment_ids, train_labels = self.featurizer.convert_text_to_features(train_texts,
                                                                                                         train_labels)
        val_input_ids, val_input_masks, val_segment_ids, val_labels = self.featurizer.convert_text_to_features(val_texts,
                                                                                                   val_labels)

        self.model.fit([train_input_ids, train_input_masks, train_segment_ids],
                       train_labels,
                       validation_data=([val_input_ids, val_input_masks, val_segment_ids], val_labels),
                       epochs=2,
                       batch_size=32)




if __name__ == "__main__":
    sess = tf.Session()
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    my_bert = BERT(bert_path)

    dataset = pd.read_csv("spam_dataset.csv")
    texts, labels = dataset["text"].tolist(), dataset["label"].tolist()

    my_bert.fit(texts, labels)




