# encoding=utf-8

"""
bert-blstm-crf layer
@Author:Macan
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, seq_length, labels, lengths, is_training):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training

    def add_blstm_crf_layer(self, crf_only):
        """
        blstm-crf网络
        :return: 
        """
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            #blstm
            lstm_output = self.blstm_layer(self.embedded_chars)
            #project
            logits = self.project_bilstm_layer(lstm_output)
        #crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return ((loss, logits, pred_ids))

    def add_blstm_crf_layer_not_really_working(self, crf_only):
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        #blstm
        lstm_output = self.blstm_layer(self.embedded_chars)
        #project
        logits = self.project_bilstm_layer(lstm_output)
        loss = tf.losses.softmax_cross_entropy(self.labels, logits, self.lengths, label_smoothing=0.9)
        pred_ids = tf.math.argmax(logits, -1)

        return ((loss, logits, pred_ids))

    def _which_cell(self):
        """
        RNN 类型
        :return: 
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LayerNormBasicLSTMCell(self.hidden_unit, dropout_keep_prob=self.dropout_rate)
            #cell_tmp = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        # 是否需要进行dropout
        if self.dropout_rate is not None:
            cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=self.dropout_rate)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._which_cell()
        cell_bw = self._which_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw
    def blstm_layer(self, embedding_chars):
        """
                
        :return: 
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars, shape=[-1, self.embedding_dims]) #[batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.lengths)
            return tf.reduce_mean(-log_likelihood), trans

class MLP_and_softmax(object):
    def __init__(self, embedded_chars, hidden_size, num_layers, hidden_dropout_prob,
                 initializers, num_labels, seq_length, labels, length_mask, is_training):
        """
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: MLP的隐含单元个数
        :param num_layers: MLP的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.length_mask = length_mask
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training
        if not is_training:
            self.hidden_dropout_prob = 0.0

    def compute(self):
        prev_output = self.embedded_chars
        with tf.variable_scope('MLP_and_softmax'):          
          for layer_idx in range(self.num_layers):
            with tf.variable_scope("MLP_layer_%d" % layer_idx):
                layer_input = prev_output
                layer_output =  tf.layers.dense(
                  layer_input,
                  self.hidden_size,
                  tf.nn.relu,
                  kernel_initializer=self.initializers.xavier_initializer())
                layer_output = dropout(layer_output, self.hidden_dropout_prob)
                prev_output = layer_output

          logits = tf.layers.dense(prev_output, self.num_labels, kernel_initializer=self.initializers.xavier_initializer())
          #logits_after_mask = logits * self.input_mask
          loss = tf.losses.softmax_cross_entropy(self.labels, logits, self.length_mask, label_smoothing=0.9)

          #cross_entropy = self.labels * 

          #loss = tf.contrib.seq2seq.sequence_loss(logits,
          #                                   self.labels,
          #                                   self.length_mask,
          #                                   average_across_timesteps=False, average_across_batch=False)
          pred_ids = tf.math.argmax(logits, -1)
        return loss, logits, pred_ids       



def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output