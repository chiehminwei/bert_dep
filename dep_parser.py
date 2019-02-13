import modeling
import numpy as np
import tensorflow as tf
import linalg

class Parser(object):

	def __init__(self, initializers, is_training, mlp_droput_rate, arc_mlp_size=500, label_mlp_size=100):
		self.initializers = initializers
		self.is_training = is_training
		self.mlp_droput_rate = mlp_droput_rate
		self.arc_mlp_size = arc_mlp_size
		self.label_mlp_size = label_mlp_size


	def compute(self, inputs, head_labels_one_hot, rel_labels_one_hot, num_head_labels, num_rel_labels, token_start_mask):
		with tf.variable_scope('MLP'):
			dep_mlp, head_mlp = self.MLP(inputs, self.arc_mlp_size, self.label_mlp_size)
			dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.arc_mlp_size], dep_mlp[:,:,self.arc_mlp_size:]
			head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.arc_mlp_size], head_mlp[:,:,self.arc_mlp_size:]
		
		with tf.variable_scope('Arcs',):
			arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
			arc_output = self.arc_output(arc_logits, head_labels_one_hot, token_start_mask)
			if self.is_training:
				predictions = head_labels_one_hot
			else:
				predictions = arc_output['predictions']				

		with tf.variable_scope('Rels'):
			rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_rel_labels, predictions)
			rel_output = self.rel_output(rel_logits, rel_labels_one_hot, token_start_mask)
			rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
		
		output = {}
		output['probabilities'] = tf.tuple([arc_output['probabilities'],
											rel_output['probabilities']])
		output['predictions'] = tf.stack([arc_output['predictions'],
										 rel_output['predictions']])
		output['arc_accuracy'] = arc_output['accuracy']
		output['rel_accuracy'] = rel_output['accuracy']
		output['loss'] = arc_output['loss'] + rel_output['loss'] 
		
		return output


	def get_arc_mask(self, logits, token_start_mask):
		mask = tf.ones_like(logits, dtype=tf.float32)
		mask_horizontal = tf.expand_dims(token_start_mask, axis=-1)
		mask_vertical = tf.expand_dims(token_start_mask, axis=1)
		mask = mask * mask_horizontal * mask_vertical
		return mask


	def arc_output(self, logits, labels_one_hot, token_start_mask):
		mask = self.get_arc_mask(logits, token_start_mask)

		batch_size, max_seq_length, embedding_size = modeling.get_shape_list(logits, expected_rank=3)
		lengths = tf.reduce_sum(mask, -1)
		lengths_mask = tf.sequence_mask(lengths, embedding_size)
		lengths_indices = tf.where(lengths_mask)
		
		dense_logsoftmax = dense_masked_logsoftmax(logits, mask, lengths_indices)
		loss = cross_entropy(dense_logsoftmax, labels_one_hot)
		
		dense = to_dense(logits, mask, lengths_indices)
		predictions = tf.math.argmax(dense, -1) # (batch_size, bucket_size) 

		targets_for_accuracy = tf.math.argmax(labels_one_hot, -1)
		accuracy = tf.metrics.accuracy(targets_for_accuracy, predictions, weights=token_start_mask)
		
		output = {
			'probabilities': dense_logsoftmax,
			'predictions': predictions,
			'accuracy': accuracy,
			'loss': loss
		}
		
		return output


	def MLP(self, inputs, arc_mlp_size, label_mlp_size):
		inputs = tf.layers.dropout(inputs, self.mlp_droput_rate, training=self.is_training)	    
		dep_mlp = tf.layers.dense(
					inputs,
					arc_mlp_size + label_mlp_size,
					modeling.gelu,
					kernel_initializer=self.initializers.xavier_initializer())

		head_mlp = tf.layers.dense(
					inputs,
					arc_mlp_size + label_mlp_size,
					modeling.gelu,
					kernel_initializer=self.initializers.xavier_initializer())
		
		return dep_mlp, head_mlp


	def bilinear_classifier(self, inputs1, inputs2, add_bias1=True, add_bias2=False):
		inputs1 = tf.layers.dropout(inputs1, self.mlp_droput_rate, training=self.is_training)	    
		inputs2 = tf.layers.dropout(inputs2, self.mlp_droput_rate, training=self.is_training)	    
		
		# bilin = (batch_size, bucket_size, n_classes, bucket_size)
		bilin = linalg.bilinear(inputs1, inputs2, 1,
								add_bias1=add_bias1,
								add_bias2=add_bias2,
								initializer=tf.zeros_initializer,
								moving_params=None)
		# output = (batch_size, bucket_size, bucket_size)
		output = tf.squeeze(bilin)
		return output


	def conditional_bilinear_classifier(self, inputs1, inputs2, n_classes, probs, add_bias1=True, add_bias2=True):
		input_shape = tf.shape(inputs1)
		batch_size = input_shape[0]
		bucket_size = input_shape[1]
		input_size = inputs1.get_shape().as_list()[-1]
		input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size+1]
		output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])

		if self.is_training:
			probs = tf.stop_gradient(probs) # (batch_size, bucket_size, bucket_size)
		
		inputs1 = tf.layers.dropout(inputs1, self.mlp_droput_rate, training=self.is_training)	 # (batch_size, bucket_size, arc_mlp_size) 
		inputs2 = tf.layers.dropout(inputs2, self.mlp_droput_rate, training=self.is_training)	 # (batch_size, bucket_size, label_mlp_size)
		
		inputs1 = tf.concat([inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))], 2)
		inputs1.set_shape(input_shape_to_set)
		inputs2 = tf.concat([inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))], 2)
		inputs2.set_shape(input_shape_to_set)
		
		bilin = linalg.bilinear(inputs1, inputs2,   # bilin = (batch_size, bucket_size, n_classes, bucket_size)
						 n_classes,
						 add_bias1=add_bias1,
						 add_bias2=add_bias2,
						 initializer=tf.zeros_initializer,
						 moving_params=None)
		weighted_bilin = tf.linalg.matmul(bilin, tf.expand_dims(probs, 3)) 
		# (batch_size, bucket_size, n_classes, bucket_size) * (batch_size, bucket_size, bucket_size, 1) =>
		weighted_bilin = tf.squeeze(weighted_bilin) # (batch_size, bucket_size, n_classes)
		
		return weighted_bilin, bilin

		
	def conditional_probabilities(self, logits4D, transpose=True):
		if transpose:									 # logits4D = (batch_size, bucket_size, n_classes, bucket_size)
		  logits4D = tf.transpose(logits4D, [0,1,3,2])   # logits4D = (batch_size, bucket_size, bucket_size, n_classes)
		original_shape = tf.shape(logits4D)
		n_classes = original_shape[3]
		
		logits2D = tf.reshape(logits4D, tf.stack([-1, n_classes]))
		probabilities2D = tf.nn.softmax(logits2D)
		return tf.reshape(probabilities2D, original_shape)


	def rel_output(self, logits, targets, token_start_mask):
		mask_horizontal = tf.expand_dims(token_start_mask, axis=-1)
		probabilities = tf.nn.softmax(logits)
		predictions = tf.math.argmax(logits, -1)
		targets_for_accuracy = tf.math.argmax(targets, -1)
		accuracy = tf.metrics.accuracy(targets_for_accuracy, predictions, weights=token_start_mask)


def masked_logsoftmax(logits, mask):
	"""
	Masked softmax over dim 1
	:param logits: (N, L)
	:param mask: (N, L)
	:return: probabilities (N, L)
	"""
	indices = tf.where(mask)
	values = tf.gather_nd(logits, indices)
	denseShape = tf.cast(tf.shape(logits), tf.int64)
	sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
	result = tf.scatter_nd(sparseResult.indices, tf.log(sparseResult.values), sparseResult.dense_shape)
	result.set_shape(logits.shape)
	result = tf.cast(result, tf.float32)
	return result


def dense_masked_logsoftmax(logits, mask, lengths_indices):
	indices = tf.where(mask)
	values = tf.gather_nd(logits, indices)
	denseShape = tf.cast(tf.shape(logits), tf.int64)
	sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
	result = tf.scatter_nd(lengths_indices, tf.log(sparseResult.values), sparseResult.dense_shape)
	result.set_shape(logits.shape)
	result = tf.cast(result, tf.float32)
	return result


def to_dense(logits, mask, lengths_indices):
	indices = tf.where(mask)
	values = tf.gather_nd(logits, indices)
	denseShape = tf.cast(tf.shape(logits), tf.int64)
	sparseResult = tf.SparseTensor(indices, values, denseShape)
	result = tf.scatter_nd(lengths_indices, sparseResult.values, sparseResult.dense_shape)
	result.set_shape(logits.shape)
	result = tf.cast(result, tf.float32)
	return result


def cross_entropy(logsoftmax, labels_one_hot, label_smoothing=0.9):
	loss = -logsoftmax * labels_one_hot * label_smoothing
	loss = tf.reduce_sum(loss)
	return loss