import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class BertFeaturizer(object):
    def __init__(self, bert_path, session, max_seq_length=64):
        self.bert_path = bert_path
        self.max_seq_length = max_seq_length
        self.session = session
        self.tokenizer = self.__create_tokenizer_from_hub_module(bert_path)

    def __create_tokenizer_from_hub_module(self, bert_path):
        """Get the vocab file and casing info from the Hub module."""
        bert_module = hub.Module(bert_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.session.run(
            [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
        )

        return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def __convert_single_example(self, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        if isinstance(example, PaddingInputExample):
            input_ids = [0] * self.max_seq_length
            input_mask = [0] * self.max_seq_length
            segment_ids = [0] * self.max_seq_length
            label = 0
            return input_ids, input_mask, segment_ids, label

        tokens_a = self.tokenizer.tokenize(example.text_a)
        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[0 : (self.max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        return input_ids, input_mask, segment_ids, example.label

    def __convert_examples_to_features(self, examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""

        input_ids, input_masks, segment_ids, labels = [], [], [], []
        for example in tqdm(examples, desc="Converting examples to features"):
            input_id, input_mask, segment_id, label = self.__convert_single_example(example)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)
        return (
            np.array(input_ids),
            np.array(input_masks),
            np.array(segment_ids),
            np.array(labels).reshape(-1, 1),
        )

    def convert_text_to_examples(self, texts, labels):
        """Create InputExamples"""
        InputExamples = []
        if labels:
            for text, label in zip(texts, labels):
                InputExamples.append(
                    InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
                )
        else:
            for text in texts:
                InputExamples.append(
                    InputExample(guid=None, text_a=" ".join(text), text_b=None, label=0)
                )
        return InputExamples

    def convert_text_to_features(self, texts, labels):
        examples = self.convert_text_to_examples(texts, labels)
        input_ids, input_masks, segment_ids, labels = self.__convert_examples_to_features(examples)
        return input_ids, input_masks, segment_ids, labels
