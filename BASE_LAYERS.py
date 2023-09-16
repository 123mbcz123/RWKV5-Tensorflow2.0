import tensorflow as tf
from keras.layers import *

class RWKVModelError(Exception):
    def __init__(self, error_msg):
        self.error_msg = error_msg

    def __str__(self):
        return self.error_msg

class CustomLayerNormalization(Layer):
    def __init__(self, scale_weight, center_weight, axis=-1, epsilon=1e-5, name="layerNorm"):
        super(CustomLayerNormalization, self).__init__(name=name)

        self.scale = scale_weight  # tf.expand_dims(tf.expand_dims(scale_weight, axis=0), axis=0)
        self.center = center_weight  # tf.expand_dims(tf.expand_dims(center_weight, axis=0), axis=0)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs):
        input_dtype = inputs.dtype
        inputs = tf.cast(inputs,dtype=tf.float32)
        mean, variance = tf.nn.moments(inputs, axes=self.axis, keepdims=True)
        #print(self.center.dtype,self.scale.dtype)
        outputs = tf.nn.batch_normalization(inputs, mean, variance, tf.cast(self.center,tf.float32), tf.cast(self.scale,tf.float32), self.epsilon)
        outputs = tf.cast(outputs,dtype=input_dtype)
        return outputs


class CustomEmbedding(Layer):
    def __init__(self, config, name="embeddingLayer"):
        super(CustomEmbedding, self).__init__(name=name)
        self.embedding_weight = self.add_weight(shape=(config['vocabulary_size'], config['hidden_size']),
                                                           dtype=tf.float32, name="emb.weight")
        ln_scale_var = self.add_weight(shape=(1,1,config['hidden_size'],), dtype=tf.float32,
                                                  name='blocks.0.ln0.weight')
        ln_center_var =self.add_weight(shape=(1,1,config['hidden_size'],), dtype=tf.float32,
                                                  name='blocks.0.ln0.bias')
        self.embedding_layerNorm = CustomLayerNormalization(ln_scale_var, ln_center_var, axis=-1, epsilon=1e-5,name="ln0")

    def call(self, inputs):
        x = tf.nn.embedding_lookup(self.embedding_weight, inputs)
        outputs = self.embedding_layerNorm(x)
        return outputs

class OutputLayer(Layer):
    def __init__(self, config, name="outputLayer"):
        super(OutputLayer, self).__init__(name=name)

        ln_scale_var = self.add_weight(shape=(1,1,config['hidden_size'],), dtype=tf.float32,
                                                  name=f'ln_out.weight')
        ln_center_var = self.add_weight(shape=(1,1,config['hidden_size'],), dtype=tf.float32,
                                                  name=f'ln_out.bias')

        self.output_norm = CustomLayerNormalization(ln_scale_var, ln_center_var, axis=-1, epsilon=1e-5,
                                                    name="output_ln")
        self.head = self.add_weight(shape=(config['hidden_size'], config['vocabulary_size']),
                                                            dtype=tf.float32, name="head.weight")

    def call(self, inputs, *args, **kwargs):
        inputs = tf.cast(inputs, dtype=tf.float32)

        x_norm = self.output_norm(inputs)
        outputs = tf.matmul(x_norm, self.head)

        return outputs




















