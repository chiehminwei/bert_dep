# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import itertools
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers.python.layers import initializers
from collections import defaultdict
from dep_parser import Parser


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, words, heads, rels):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      words: white space seperated string, 1st column of CoNLL
      heads: int [], 6th column of CoNLL
      rels: white space seperated string, 7th column of CoNLL
    """
    self.guid = guid
    self.words = words
    self.heads = heads
    self.rels = rels


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, head_label_ids, rel_label_ids, token_start_mask):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.head_label_ids = head_label_ids
    self.rel_label_ids = rel_label_ids
    self.token_start_mask = token_start_mask


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_examples(self, mode, data_dir):
    """Gets a collection of `InputExample`s."""
    raise NotImplementedError()

  def get_labels(self, data_dir):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class CoNLLProcessor(DataProcessor):

  def get_examples(self, mode, data_dir):
    """Gets a collection of `InputExample`s for the data set."""
    assert mode in ['train', 'dev', 'test'], "mode must be 'train', 'dev', or 'test'"
    lines = self._read_tsv(os.path.join(data_dir, 'en_ewt-ud-' + mode + '.conllu'))    
    example_count = 0
    examples = []
    words = []
    tags_1 = []
    tags_2 = []
    heads = []
    rels = []
    for line in lines:
      if len(line) == 10:
        word = line[1]
        tag_1 = line[3]
        tag_2 = line[4]
        try: # check input is well formed
          head = int(line[6])
        except:
          tf.logging.info("Input is mal-formed! %s" % (line))
          continue
        head = line[6]
        rel = line[7]

        words.append(word)
        tags_1.append(tag_1)
        tags_2.append(tag_2)
        heads.append(head)
        rels.append(rel)

      else:
        if line and line[0].startswith('#'):
          if words:
            guid = "%s-%d" % (mode, example_count)
            words = tokenization.convert_to_unicode(' '.join(words))
            heads = tokenization.convert_to_unicode(' '.join(heads))
            rels = tokenization.convert_to_unicode(' '.join(rels))
            examples.append(
              InputExample(guid=guid, words=words, heads=heads, rels=rels))
          words = []
          tags_1 = []
          tags_2 = []
          heads = []
          rels = []
        else:
          example_count += 1
    # Accounts for last sentence in file
    if words:
      example_count -= 1
      guid = "%s-%d" % (mode, example_count)
      words = tokenization.convert_to_unicode(' '.join(words))
      heads = tokenization.convert_to_unicode(' '.join(heads))
      rels = tokenization.convert_to_unicode(' '.join(rels))
      examples.append(
        InputExample(guid=guid, words=words, heads=heads, rels=rels))

    return examples

  def get_labels(self, data_dir):
    """See base class."""
    train_lines = self._read_tsv(os.path.join(data_dir, "en_ewt-ud-train.conllu"))
    dev_lines = self._read_tsv(os.path.join(data_dir, "en_ewt-ud-dev.conllu"))
    test_lines = self._read_tsv(os.path.join(data_dir, "en_ewt-ud-test.conllu"))
    lines = train_lines + dev_lines + test_lines
    head_counts = defaultdict(int)
    rel_counts = defaultdict(int)
    for line in lines:
      if len(line) == 10:
        try:
          head = int(line[6])
        except:
          continue
        head = tokenization.convert_to_unicode(line[6])
        rel = tokenization.convert_to_unicode(line[7])
        head_counts[head] += 1
        rel_counts[rel] += 1

    total_head_labels = len(head_counts)
    total_rel_labels = len(rel_counts)
    tf.logging.info("There are {} rel labels and {} head labels in total.".format(total_rel_labels, total_head_labels))
    
    head_labels = sorted(head_counts, key=head_counts.get, reverse=True)
    rel_labels = sorted(rel_counts, key=rel_counts.get, reverse=True)
    return head_labels, rel_labels


def convert_single_example(ex_index, example, head_label_list, rel_label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  
  # -1 is for UNK and [SEP] and [CLS] (root)
  rel_label_map = defaultdict(lambda: -1)
  for (i, label) in enumerate(rel_label_list):    
    rel_label_map[label] = i

  # -1 is for index longer than max_seq_length
  # 0 is for [CLS] (root)
  head_label_map = defaultdict(lambda: -1)
  for (i, label) in enumerate(head_label_list):
    real_label = int(label)
    head_label_map[label] = real_label
    if real_label >= max_seq_length-1:
        head_label_map[label] = -1
    
  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  segment_ids = [0] * max_seq_length
  
  orig_tokens = example.words.split()
  rel_labels = example.rels.split()
  head_labels = example.heads.split()
  
  # Token map will be an int -> int mapping between the `orig_tokens` index and the `bert_tokens` index.
  token_map = {}
  bert_tokens = []
  token_start_idxs = [0]
  token_start_mask = [0] * max_seq_length
  head_label_ids = []
  rel_label_ids = []
  head_label_ids = [] # Similar to rel_label_ids, 0 is reserved for UNK, [CLS], and [SEP]

  bert_tokens.append("[CLS]")
  rel_label_ids.append(-1)
  head_label_ids.append(0)
  for orig_token, head_label, rel_label in zip(orig_tokens, head_labels, rel_labels):
    sub_tokens = tokenizer.tokenize(orig_token)
    rel_label_ids.extend([rel_label_map[rel_label]] + [-1] * (len(sub_tokens)-1))
    token_start_idxs.append(len(bert_tokens))
    bert_tokens.extend(sub_tokens)

  if len(bert_tokens) >= max_seq_length:
    tf.logging.info("*** Skipping sentence of length: %d tokens***" % (len(bert_tokens)))
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in bert_tokens]))
    
    bert_tokens = bert_tokens[:max_seq_length-1]
    head_label_ids = head_label_ids[:max_seq_length-1]
    rel_label_ids = rel_label_ids[:max_seq_length-1]
    return

  for start_idx in token_start_idxs:
    if start_idx >= max_seq_length: break
    token_start_mask[start_idx] = 1
  
  orig_index = 0
  for i, idx in enumerate(token_start_idxs):
    token_map[i] = idx

  for orig_token, head_label, rel_label in zip(orig_tokens, head_labels, rel_labels):
    sub_tokens = tokenizer.tokenize(orig_token)
    head_label_ids.extend([token_map[head_label_map[head_label]]] + [len(head_label_ids)-1] * (len(sub_tokens)-1))


  bert_tokens.append("[SEP]")
  head_label_ids.append(-1)
  rel_label_ids.append(-1)

  input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)  
  
  # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
  input_mask = [1] * len(input_ids)

  assert len(input_ids) == len(rel_label_ids) == len(head_label_ids), 'len input_ids: {}, len rel_label_ids: {}, len head_label_ids: {}'.format(len(input_ids), len(rel_label_ids), len(head_label_ids))

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    rel_label_ids.append(-1)
    head_label_ids.append(-1)
    input_mask.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(head_label_ids) == max_seq_length
  assert len(rel_label_ids) == max_seq_length
  assert len(token_start_mask) == max_seq_length
  
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in bert_tokens]))
    tf.logging.info("token_start_mask: %s" % " ".join([str(x) for x in token_start_mask]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("rel_labels: %s" % (" ".join([str(x) for x in rel_label_ids])))
    tf.logging.info("head_labels: %s" % ( " ".join([str(x) for x in head_label_ids])))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      head_label_ids=head_label_ids,
      rel_label_ids=rel_label_ids,
      token_start_mask=token_start_mask)
  return feature


def file_based_convert_examples_to_features(
    examples, head_label_list, rel_label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, head_label_list, rel_label_list,
                                     max_seq_length, tokenizer)
    if not feature: continue

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["head_label_ids"] = create_int_feature(feature.head_label_ids)
    features["rel_label_ids"] = create_int_feature(feature.rel_label_ids)
    features["token_start_mask"] = create_int_feature(feature.token_start_mask)
   
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "head_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "rel_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "token_start_mask": tf.FixedLenFeature([seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn



def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 head_labels_one_hot, rel_labels_one_hot, num_head_labels, num_rel_labels, 
                 use_one_hot_embeddings, token_start_mask, mlp_droupout_rate, arc_mlp_size, label_mlp_size):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
  embedding = model.get_sequence_output()
  batch_size, max_seq_length, embedding_size = modeling.get_shape_list(embedding, expected_rank=3)
  masked_embedding = tf.boolean_mask(embedding, token_start_mask)
  # lengths = tf.reduce_sum(input_mask, reduction_indices=1)  # [batch_size] vector, sequence lengths of current batch
  mask = tf.to_float(token_start_mask)
  
  parser = Parser(initializers, is_training, mlp_droupout_rate, token_start_mask, arc_mlp_size, label_mlp_size)
  output = parser.compute(embedding, head_labels_one_hot, rel_labels_one_hot, num_head_labels, num_rel_labels, mask)
  return output


def model_fn_builder(bert_config, num_rel_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, max_seq_length, mlp_droupout_rate, arc_mlp_size, label_mlp_size):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    head_label_ids = features["head_label_ids"]
    rel_label_ids = features["rel_label_ids"]
    token_start_mask = features["token_start_mask"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    num_head_labels = max_seq_length
    head_labels_one_hot = tf.one_hot(head_label_ids, num_head_labels)
    rel_labels_one_hot = tf.one_hot(rel_label_ids, num_rel_labels)

    output = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, head_labels_one_hot,
        rel_labels_one_hot, num_head_labels, num_rel_labels, use_one_hot_embeddings, token_start_mask,
        mlp_droupout_rate, arc_mlp_size, label_mlp_size)

    total_loss = output['loss']

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(output):
        arc_accuracy = output['arc_accuracy']
        rel_accuracy = output['rel_accuracy']
        
        return {
            "arc_accuracy": arc_accuracy,
            "rel_accuracy": rel_accuracy
        }

      eval_metrics = (metric_fn, [output])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)

    else:
      arc_probabilities, rel_probabilities = output['probabilities'] 
      arc_predictions, rel_predictions = output['predictions']

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
         mode=mode,
         predictions={"arc_probabilities": arc_probabilities, "rel_probabilities": rel_probabilities, 
                      "arc_predictions": arc_predictions, "rel_predictions": rel_predictions, 
                      "token_start_mask": token_start_mask, "input_mask": input_mask},
         scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn