import modeling
import numpy as np
import tensorflow as tf
import linalg

from tensorflow.contrib import rnn
from tensorflow.contrib import crf




class Parser(object):

	def __init__(self, is_training, num_head_labels, num_rel_labels, mlp_droput_rate, token_start_mask, arc_mlp_size, label_mlp_size, batch_size):
		self.is_training = is_training
		self.mlp_droput_rate = mlp_droput_rate
		self.arc_mlp_size = arc_mlp_size
		self.label_mlp_size = label_mlp_size
		self.token_start_mask = token_start_mask
		self.num_head_labels = num_head_labels
		self.num_rel_labels = num_rel_labels		
		self.batch_size = batch_size


	def __call__(self, inputs, gold_heads, gold_labels, head_label_ids_for_indexing, rel_label_ids_for_indexing):
		
		inputs = tf.layers.dropout(inputs, self.mlp_droput_rate, training=self.is_training)	
		with tf.variable_scope('arc_h', reuse=tf.AUTO_REUSE):
			arc_h = self.MLP(inputs, self.arc_mlp_size)

		with tf.variable_scope('arc_d', reuse=tf.AUTO_REUSE):
			arc_d = self.MLP(inputs, self.arc_mlp_size)

		with tf.variable_scope('lab_h', reuse=tf.AUTO_REUSE):
			lab_h = self.MLP(inputs, self.label_mlp_size)

		with tf.variable_scope('lab_d', reuse=tf.AUTO_REUSE):
			lab_d = self.MLP(inputs, self.label_mlp_size)

		with tf.variable_scope('s_arc', reuse=tf.AUTO_REUSE):
			s_arc = self.biaffine(arc_d, arc_h, 
							n_in=self.arc_mlp_size,
							bias_x=True,
							bias_y=False)

		with tf.variable_scope('s_lab', reuse=tf.AUTO_REUSE):
			lab_attn = self.biaffine(lab_d, lab_h, 
								n_in=self.label_mlp_size,
								n_out=self.num_rel_labels,
								bias_x=True,
								bias_y=True)

			s_lab = tf.transpose(lab_attn, perm=[0, 2, 3, 1])

		output = {}
		
		loss = self.get_loss(s_arc, s_lab, gold_heads, gold_labels, head_label_ids_for_indexing)
		output['loss'] = loss
		
		if not self.is_training:
			pred_heads, pred_labels = self.decode(s_arc, s_lab)
			#pred_heads = self.decode(s_arc, s_lab)
			#arc_accuracy = self.get_accuracy(pred_heads, None, gold_heads, gold_labels)
			arc_accuracy, rel_accuracy = self.get_accuracy(pred_heads, pred_labels, gold_heads, gold_labels)
			output['arc_accuracy'] = arc_accuracy
			output['rel_accuracy'] = rel_accuracy
			output['arc_predictions'] = pred_heads
			output['rel_predictions'] = pred_labels
		return output
		

	def get_loss(self, s_arc, s_lab, gold_heads, gold_labels, head_label_ids_for_indexing):
		s_lab = self.select_indices(s_lab, head_label_ids_for_indexing)		
		gold_heads = tf.one_hot(gold_heads, self.num_head_labels)
		gold_labels = tf.one_hot(gold_labels, self.num_rel_labels)
		# arc_loss = tf.losses.softmax_cross_entropy(gold_heads, s_arc, weights=self.token_start_mask, label_smoothing=0.9)  
		lab_loss = tf.losses.softmax_cross_entropy(gold_labels, s_lab, weights=self.token_start_mask, label_smoothing=0.9)
		#loss = arc_loss
		# loss = arc_loss + lab_loss
		loss = lab_loss
		return loss

	def decode(self, s_arc, s_lab):
		pred_heads = tf.argmax(s_arc, -1)
		s_lab = self.select_indices(s_lab, pred_heads)
		pred_labels = tf.argmax(s_lab, -1)
		#return pred_heads
		return pred_heads, pred_labels

	def get_accuracy(self, pred_heads, pred_labels, gold_heads, gold_labels):
		arc_accuracy = tf.metrics.accuracy(gold_heads, pred_heads, self.token_start_mask)
		rel_accuracy = tf.metrics.accuracy(gold_labels, pred_labels, self.token_start_mask)
		return arc_accuracy, rel_accuracy


	def MLP(self, inputs, mlp_size):
		mlp = tf.layers.dense(
					inputs,
					mlp_size,
					modeling.gelu,
					kernel_initializer=tf.orthogonal_initializer())
		mlp = tf.layers.dropout(mlp, self.mlp_droput_rate, training=self.is_training)			
		return mlp

	def biaffine(self, x, y, n_in, n_out=1, bias_x=True, bias_y=True):
		self.n_in = n_in
		self.n_out = n_out
		self.bias_x = bias_x
		self.bias_y = bias_y
		batch_size, max_seq_length, embedding_size = modeling.get_shape_list(x, expected_rank=3)
		self.weight = tf.get_variable("biaffine_weight", 
									shape=[self.batch_size, n_out, n_in + bias_x, n_in + bias_y],
									dtype=tf.float32)

		if self.bias_x:
			x = tf.concat([x, tf.ones(tf.stack([batch_size, max_seq_length, 1]))], 2)
		if self.bias_y:
			y = tf.concat([y, tf.ones(tf.stack([batch_size, max_seq_length, 1]))], 2)
		# [batch_size, 1, seq_len, d]
		x = tf.expand_dims(x, 1)
		x = tf.broadcast_to(x, [batch_size, n_out, max_seq_length, n_in + bias_x])
		# [batch_size, 1, seq_len, d]
		y = tf.expand_dims(y, 1)
		y = tf.broadcast_to(y, [batch_size, n_out, max_seq_length, n_in + bias_y])
		# [batch_size, n_out, seq_len, d_1] @ [batch_size, n_out, d_1, d_2] @ [batch_size, n_out, d_2, seq_len]
		# => [batch_size, n_out, seq_len, d_2] @ [batch_size, 1, d_2, seq_len]
		# => [batch_size, n_out, seq_len, seq_len]
		s = x @ self.weight @ tf.transpose(y, perm=[0, 1, 3, 2])
		# remove dim 1 if n_out == 1
		if n_out == 1:
			s = tf.squeeze(s, 1)

		return s

	def select_indices(self, inputs, indices):
		# inputs = [batch_size, seq_len, seq_len, n_out]
		# indices = [batch_size, seq_len]
		# Construct nd_indices
		indices = tf.cast(indices, dtype=tf.int32)
		batch_size, seq_len = modeling.get_shape_list(indices, expected_rank=2)

		batches = tf.broadcast_to(tf.reshape(tf.range(batch_size),[batch_size,1]),[batch_size, seq_len])
		seqs = tf.broadcast_to(tf.range(seq_len), [batch_size, seq_len])

		nd_indices = tf.stack([batches, seqs, indices], axis=2)
		result = tf.gather_nd(inputs, nd_indices)
		return result


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