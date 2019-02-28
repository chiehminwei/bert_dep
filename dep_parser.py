import modeling
import numpy as np
import tensorflow as tf
import linalg





class Parser(object):

	def __init__(self, is_training, num_head_labels, num_rel_labels, mlp_droput_rate, token_start_mask, arc_mlp_size, label_mlp_size):
		self.is_training = is_training
		self.mlp_droput_rate = mlp_droput_rate
		self.arc_mlp_size = arc_mlp_size
		self.label_mlp_size = label_mlp_size
		self.token_start_mask = token_start_mask
		self.num_head_labels = num_head_labels
		self.num_rel_labels = num_rel_labels		


	def __call__(self, inputs, gold_heads, gold_labels):
		
		inputs = tf.layers.dropout(inputs, self.mlp_droput_rate, training=self.is_training)	
		arc_h = self.MLP(inputs, self.arc_mlp_size)
		arc_d = self.MLP(inputs, self.arc_mlp_size)
		lab_h = self.MLP(inputs, self.label_mlp_size)
		lab_d = self.MLP(inputs, self.label_mlp_size)

		s_arc = self.biaffine(arc_d, arc_h, 
						n_in=self.arc_mlp_size,
						bias_x=True,
						bias_y=False)

		lab_attn = self.biaffine(lab_d, lab_h, 
							n_in=self.label_mlp_size,
							n_out=self.num_rel_labels,
							bias_x=True,
							bias_y=True)

		s_lab = tf.transpose(lab_attn, perm=[0, 2, 3, 1])

		output = {}
		
		loss = self.get_loss(s_arc, s_lab, gold_heads, gold_labels)
		output['loss'] = loss
		
		if not self.is_training:
			pred_heads, pred_labels = self.decode(s_arc, s_lab)
			arc_accuracy, rel_accuracy = self.get_accuracy(pred_heads, pred_labels, gold_heads, gold_labels)
			output['arc_accuracy'] = arc_accuracy
			output['rel_accuracy'] = rel_accuracy
			output['arc_predictions'] = pred_heads
			output['rel_predictions'] = pred_labels
		return output
		

	def get_loss(self, s_arc, s_lab, gold_heads, gold_labels):
		s_lab = self.select_indices(s_lab, gold_heads)		
		gold_heads = tf.one_hot(gold_heads, self.num_head_labels)
		gold_labels = tf.one_hot(gold_labels, self.num_rel_labels)
		arc_loss = tf.losses.softmax_cross_entropy(gold_heads, s_arc, weight=self.token_start_mask, label_smoothing=0.9)  
		lab_loss = tf.losses.softmax_cross_entropy(gold_labels, s_lab, weight=self.token_start_mask, label_smoothing=0.9)

		loss = arc_loss + lab_loss
		return loss

	def decode(self, s_arc, s_lab):
		pred_heads = tf.argmax(s_arc, -1)
		s_lab = self.select_indices(s_lab, pred_heads)
		pred_labels = tf.argmax(s_lab, -1)

		return pred_heads, pred_labels

	def get_accuracy(pred_heads, pred_labels, gold_heads, gold_labels):
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
									[batch_size, n_out, n_in + bias_x, n_in + bias_y], 
									reuse=True,
									dtype=tf.float32,
									initializer=tf.zeros_initializer)

		if self.bias_x:
			x = tf.concat([x, tf.ones(tf.stack([batch_size, max_seq_length, 1]))], 2)
		if self.bias_y:
			y = tf.concat([y, tf.ones(tf.stack([batch_size, max_seq_length, 1]))], 2)
		# [batch_size, 1, seq_len, d]
		x = tf.expand_dims(x, 1)
		# [batch_size, 1, seq_len, d]
		y = tf.expand_dims(y, 1)
		# [batch_size, 1, seq_len, d_1] @ [batch_size, n_out, d_1, d_2] @ [batch_size, 1, d_2, seq_len]
		# => [batch_size, n_out, seq_len, d_2] @ [batch_size, 1, d_2, seq_len]
		# => [batch_size, n_out, seq_len, seq_len]
		s = x @ self.weight @ tf.transpose(y, perm=[0, 1, 3, 2])
		# remove dim 1 if n_out == 1
		s = tf.squeeze(s, 1)

		return s

	def select_indices(inputs, indices):
		nd_indices = tf.stack([tf.range(tf.shape(inputs)[0]), indices], axis=1)
		result = tf.gather_nd(inputs, nd_indices)
		return result