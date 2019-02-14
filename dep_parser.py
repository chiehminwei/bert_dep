import modeling
import numpy as np
import tensorflow as tf
import linalg

class Parser(object):

	def __init__(self, initializers, is_training, mlp_droput_rate, token_start_mask, arc_mlp_size=500, label_mlp_size=100):
		self.initializers = initializers
		self.is_training = is_training
		self.mlp_droput_rate = mlp_droput_rate
		self.arc_mlp_size = arc_mlp_size
		self.label_mlp_size = label_mlp_size
		self.token_start_mask = token_start_mask

	def compute2(self, inputs, head_labels_one_hot, rel_labels_one_hot, num_head_labels, num_rel_labels, token_start_mask):
		with tf.variable_scope('FML'):
			batch_size, max_seq_length, embedding_size = modeling.get_shape_list(inputs, expected_rank=3)
			inputs = tf.layers.dropout(inputs, self.mlp_droput_rate, training=self.is_training)	 
			mlp = tf.layers.dense(
					inputs,
					arc_mlp_size,
					modeling.gelu,
					kernel_initializer=self.initializers.xavier_initializer()) 
			logits = tf.layers.dense(
					mlp,
					embedding_size,
					modeling.gelu,
					kernel_initializer=self.initializers.xavier_initializer())
			predictions = tf.math.argmax(logits, -1) # (batch_size, bucket_size) 
			loss = tf.losses.softmax_cross_entropy(head_labels_one_hot, logits, weights=token_start_mask, label_smoothing=0.9)    
			
			output = {
				'predictions': predictions,
				'loss': loss
			}

			if not self.is_training:
				probabilities = tf.nn.softmax(logits)
				output['probabilities'] = probabilities

				targets_for_accuracy = tf.math.argmax(labels_one_hot, -1)
				accuracy = tf.metrics.accuracy(targets_for_accuracy, predictions, self.token_start_mask)
				output['accuracy'] = accuracy

			return output


	def compute(self, inputs, head_labels_one_hot, rel_labels_one_hot, num_head_labels, num_rel_labels, token_start_mask):
		with tf.variable_scope('MLP'):
			dep_mlp, head_mlp = self.MLP(inputs, self.arc_mlp_size, self.label_mlp_size)
			dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.arc_mlp_size], dep_mlp[:,:,self.arc_mlp_size:]
			head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.arc_mlp_size], head_mlp[:,:,self.arc_mlp_size:]
		
		with tf.variable_scope('Arcs',):
			arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
			arc_output = self.output(arc_logits, head_labels_one_hot)
			if self.is_training:
				predictions = head_labels_one_hot
			else:
				predictions = arc_output['predictions']				

		with tf.variable_scope('Rels'):
			rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_rel_labels, predictions)
			rel_output = self.output(rel_logits, rel_labels_one_hot)
					
		output = {}
		if not self.is_training:			
			output['rel_probabilities'] = self.conditional_probabilities(rel_logits_cond)
			output['arc_probabilities'] = arc_output['probabilities']
			output['arc_accuracy'] = arc_output['accuracy']
			output['rel_accuracy'] = rel_output['accuracy']
			output['head_labels_one_hot'] = tf.math.argmax(head_labels_one_hot, -1)
			output['rel_labels_one_hot'] = tf.math.argmax(rel_labels_one_hot, -1)

		output['arc_predictions'] = arc_output['predictions']
		output['rel_predictions'] = rel_output['predictions']		
		output['loss'] = arc_output['loss'] + rel_output['loss'] 
		
		return output


	def output(self, logits, labels_one_hot):		
		predictions = tf.math.argmax(logits, -1) # (batch_size, bucket_size) 
		loss = tf.losses.softmax_cross_entropy(labels_one_hot, logits, label_smoothing=0.9)    
		
		output = {
			'predictions': predictions,
			'loss': loss
		}

		if not self.is_training:
			probabilities = tf.nn.softmax(logits)
			output['probabilities'] = probabilities

			targets_for_accuracy = tf.math.argmax(labels_one_hot, -1)
			accuracy = tf.metrics.accuracy(targets_for_accuracy, predictions, self.token_start_mask)
			output['accuracy'] = accuracy

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
		if not self.is_training:
			# is not training
			probs = tf.to_float(tf.one_hot(tf.to_int32(probs), bucket_size))
		else:
			# is training
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
		# (batch_size, bucket_size, n_classes, 1)
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


