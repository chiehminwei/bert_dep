from modeling import gelu
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
			# token_start_mask = (batch_size, bucket_size, bucket_size)
			arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
			arc_output = self.arc_output(arc_logits, head_labels_one_hot, token_start_mask)
			if self.is_training:
				predictions = head_labels_one_hot
			else:
				predictions = arc_output['predictions']				

		with tf.variable_scope('Rels'):
			rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_rel_labels, predictions)
			rel_output = self.output(rel_logits, rel_labels_one_hot, token_start_mask)
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


	def MLP(self, inputs, arc_mlp_size, label_mlp_size):

		inputs = tf.layers.dropout(inputs, self.mlp_droput_rate, training=self.is_training)	    
		dep_mlp = tf.layers.dense(
					inputs,
					arc_mlp_size + label_mlp_size,
					gelu,
					kernel_initializer=self.initializers.xavier_initializer())

		head_mlp = tf.layers.dense(
					inputs,
					arc_mlp_size + label_mlp_size,
					gelu,
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
		# probs = (batch_size, bucket_size, bucket_size)
		if self.is_training:
			probs = tf.stop_gradient(probs)
		
		# inputs = (batch_size, bucket_size, arc_mlp_size/ label_mlp_size)
		inputs1 = tf.layers.dropout(inputs1, self.mlp_droput_rate, training=self.is_training)	    
		inputs2 = tf.layers.dropout(inputs2, self.mlp_droput_rate, training=self.is_training)	 
		
		inputs1 = tf.concat([inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))], 2)
		inputs1.set_shape(input_shape_to_set)
		inputs2 = tf.concat([inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))], 2)
		inputs2.set_shape(input_shape_to_set)
		
		# bilin = (batch_size, bucket_size, n_classes, bucket_size)
		bilin = linalg.bilinear(inputs1, inputs2,
						 n_classes,
						 add_bias1=add_bias1,
						 add_bias2=add_bias2,
						 initializer=tf.zeros_initializer,
						 moving_params=None)
		weighted_bilin = tf.linalg.matmul(bilin, tf.expand_dims(probs, 3)) #IDK this line
		# expanded_probs = (batch_size, bucket_size, bucket_size, 1)
		# weighted_bilin = (batch_size, bucket_size, n_classes, 1) lmao idk
		
		return weighted_bilin, bilin

	def conditional_probabilities(self, logits4D, transpose=True):
		# logits4D = (batch_size, bucket_size, n_classes, bucket_size)
		if transpose:
		  logits4D = tf.transpose(logits4D, [0,1,3,2])
		# logits4D = (batch_size, bucket_size, bucket_size, n_classes)
		original_shape = tf.shape(logits4D)
		n_classes = original_shape[3]
		
		logits2D = tf.reshape(logits4D, tf.stack([-1, n_classes]))
		probabilities2D = tf.nn.softmax(logits2D)
		return tf.reshape(probabilities2D, original_shape)

	def arc_output(self, logits, targets, token_start_mask):
		
		mask = tf.ones_like(logits, dtype=tf.float32)
		mask_horizontal = tf.expand_dims(token_start_mask, axis=-1)
		mask_vertical = tf.expand_dims(token_start_mask, axis=1)
		mask = mask * mask_horizontal * mask_vertical

		masked_logits = logits * mask
		
		# probabilities = (batch_size, bucket_size, bucket_size)
		probabilities = tf.sparse.softmax(masked_logits) 

		# predictions = (batch_size, bucket_size) 
		predictions = tf.math.argmax(masked_logits, -1)
		loss = tf.losses.sparse_softmax_cross_entropy(targets, masked_logits, mask_horizontal)
		accuracy = tf.metrics.accuracy(targets, predictions, weights=mask_horizontal)

		output = {
			'probabilities': probabilities,
			'predictions': predictions,
			'accuracy': accuracy,
			'loss': loss
		}
		
		return output