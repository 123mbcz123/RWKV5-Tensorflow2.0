import gc
import json
import os.path
import time
from typing import Dict, Union, Tuple, List

import numpy as np
from keras import Model

#import comparer

from BASE_LAYERS import *


"""
  获取模型一共有多少层
"""
def init_time_mix_k(layer_idx,num_layers,hidden_size):
    indices = np.reshape(np.arange(0,hidden_size,dtype=float) / hidden_size,(1,1,hidden_size))
    a10 = 1. - layer_idx / num_layers
    mix_k = np.power(indices,a10)
    return mix_k

def init_time_mix_r(layer_idx,num_layers,hidden_size):
    indices = np.reshape(np.arange(0,hidden_size,dtype=float) / hidden_size,(1,1,hidden_size))
    a10 = 1. - layer_idx / num_layers
    mix_r = np.power(indices,0.5 * a10)
    return mix_r

def init_time_mix_v(layer_idx,num_layers,hidden_size):
    indices = np.reshape(np.arange(0,hidden_size,dtype=float) / hidden_size,(1,1,hidden_size))
    a10 = 1. - layer_idx / num_layers
    a01 = layer_idx / (num_layers - 1 + 1e-8)
    mix_v = np.power(indices,a10) + 0.3 * a01
    return mix_v

def init_time_decay(layer_idx,num_heads,num_layers):
    a01 = layer_idx / (num_layers - 1 + 1e-8)
    decay = -8 + 7 * (np.arange(0,num_heads,dtype=float) / (num_heads - 1. + 1e-7)) **(0.7 + 1.3 * a01)
    return decay
def init_time_first(num_heads):
    first = np.ones(shape=(num_heads,),dtype=float) * (-3.)
    return first

def get_model_layers_count(model_dict: Dict):
    max_layer = 0
    for var_name in model_dict.keys():
        if 'blocks' in var_name:
            max_layer = max(max_layer, int(var_name.split('.')[1]))
    return max_layer + 1

def custom_initialize(key,layer_idx,num_heads,hidden_size,num_layers):
    if 'decay' in key:
        return tf.constant_initializer(init_time_decay(layer_idx,num_heads,num_layers))
    elif 'first' in key:
        return tf.constant_initializer(init_time_first(num_heads))
    elif 'mix_k' in key:
        return tf.constant_initializer(init_time_mix_k(layer_idx,num_layers,hidden_size))
    elif 'mix_r' in key:
        return tf.constant_initializer(init_time_mix_r(layer_idx,num_layers,hidden_size))
    else:
        return tf.constant_initializer(init_time_mix_v(layer_idx,num_layers,hidden_size))



class TimeMixChunkParallel(Layer):
    def __init__(self, layer_idx,config, trainable=True):
        super(TimeMixChunkParallel, self).__init__(name=f"time_mix_{layer_idx}", trainable=trainable)
        self.chunk_size = config['seq_chunk']
        self.d_model = config['hidden_size']
        self.num_heads = config['num_heads']
        self.d_head = config['head_size']

        self.time_mix_r =self.add_weight(initializer=custom_initialize('time_mix_r',layer_idx,config['num_heads'], config['hidden_size'], config['num_layers']),
                                         shape=(1, 1, config['hidden_size'],), dtype=tf.float32,name=f'blocks.{layer_idx}.att.time_mix_r')
        self.time_mix_k =self.add_weight(initializer=custom_initialize('time_mix_k',layer_idx,config['num_heads'], config['hidden_size'], config['num_layers']),
                                         shape=(1, 1, config['hidden_size'],), dtype=tf.float32,name=f'blocks.{layer_idx}.att.time_mix_k')
        self.time_mix_v = self.add_weight(initializer=custom_initialize('time_mix_v',layer_idx,config['num_heads'], config['hidden_size'], config['num_layers']),
                                         shape=(1, 1, config['hidden_size'],), dtype=tf.float32,name=f'blocks.{layer_idx}.att.time_mix_v')


        self.key_weight = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),dtype=tf.float32,
                                          trainable=trainable, name=f'blocks.{layer_idx}.att.key.weight')
        self.value_weight = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),dtype=tf.float32,
                                            trainable=trainable, name=f'blocks.{layer_idx}.att.value.weight')
        self.receptance_weight = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),dtype=tf.float32,
                                                 trainable=trainable, name=f'blocks.{layer_idx}.att.receptance.weight')
        self.output_weight = self.add_weight(shape=(config['hidden_size'], config['hidden_size']), dtype=tf.float32,
                                             trainable=trainable, name=f'blocks.{layer_idx}.att.output.weight')


        self.time_decay =  self.add_weight(initializer=custom_initialize('time_decay', layer_idx, config['num_heads'], config['hidden_size'], config['num_layers']),
                                           shape=(config['num_heads'],),dtype=tf.float32,name=f'blocks.{layer_idx}.att.time_decay')
        self.time_first = self.add_weight(initializer=custom_initialize('time_first', layer_idx, config['num_heads'], config['hidden_size'], config['num_layers']),
                                          shape=(config['num_heads'],),dtype=tf.float32,name=f'blocks.{layer_idx}.att.time_first')




        input_ln_scale_var = self.add_weight(shape=(1, 1, config['hidden_size']),dtype=tf.float32,name=f'blocks.{layer_idx}.ln1.weight')
        input_ln_center_var =self.add_weight(shape=(1, 1, config['hidden_size']),dtype=tf.float32,name=f'blocks.{layer_idx}.ln1.bias')
        self.input_norm = CustomLayerNormalization(input_ln_scale_var,input_ln_center_var,axis=-1,name=f"att_input_norm_{layer_idx}")


        group_ln_scale_var = self.add_weight(shape=(1, 1,1, config['num_heads'], config['head_size']),dtype=tf.float32,name=f'blocks.{layer_idx}.att.ln_x.weight')
        group_ln_center_var =self.add_weight(shape=(1, 1,1, config['num_heads'], config['head_size']),dtype=tf.float32,name=f'blocks.{layer_idx}.att.ln_x.bias')
        self.group_norm = CustomLayerNormalization(group_ln_scale_var,group_ln_center_var,axis=-1,name=f"att_output_norm_{layer_idx}")



    def get_initial_state(self,batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32
        if batch_size is None:
            batch_size = 1
        """
        生成初始的state cell,其中上一状态的输入state_x,分子state_a,分母state_b使用全零初始化
        缩放因子state p初始化到负无穷 经过maximum基本上不影响输入的值,在这里初始化到负-1e8
        """

        state = tf.zeros(shape=(batch_size, self.num_heads, self.d_head, self.d_head), dtype=dtype)
        return state

    def call(self, inputs, initial_state_dict=None):
        bz, seq_len, d_model = tf.TensorShape(inputs.shape).as_list()
        tf.assert_equal(seq_len % self.chunk_size, 0, "输入文本序列长度必须是分块长度的整数倍")
        num_chunks = seq_len // self.chunk_size
        input_dtype = inputs.dtype

        inputs_norm = self.input_norm(inputs)

        reversed_id = tf.reshape(tf.range(self.chunk_size-1,-1,-1,dtype=tf.float32),(1,self.chunk_size))# shape=(1,chunk_size)
        w = tf.cast(tf.reshape(tf.exp(-tf.exp(self.time_decay)),(self.num_heads,1)),dtype=tf.float32)#shape=(num_heads,1)


        #w_chunk用于衰减过去时间步的信息,前一个chunk的state每个时间步衰减pow(W,chunk_size)次幂,得到过去state对下一个state的贡献
        w_chunk = tf.cast(tf.reshape(tf.pow(w,tf.cast(self.chunk_size,dtype=w.dtype)),(1,self.num_heads,1,1)),dtype=input_dtype)

        w = tf.cast(tf.pow(w,reversed_id),dtype=input_dtype)#shape=(num_heads,chunk_size)

        w_k = tf.reshape(w,(1,self.num_heads,1,self.chunk_size))

        w_b = tf.reverse(tf.reshape(w,(1,self.num_heads,self.chunk_size,1)),axis=[2])

        u = tf.cast(tf.exp(tf.reshape(self.time_first,(self.num_heads,1))),dtype=inputs_norm.dtype)#shape=(num_heads,1)

        w_mask = tf.concat([w[:,1:],u],axis=-1)
        w_mask = tf.pad(w_mask,[[0,0],[0,self.chunk_size]],constant_values=0.)
        w_mask = tf.tile(w_mask,[1,self.chunk_size])
        w_mask = tf.reshape(w_mask[:,:-self.chunk_size],(self.num_heads,self.chunk_size,self.chunk_size *2 -1))
        w_mask = tf.reshape(w_mask[:,:,self.chunk_size-1:],(1,1,self.num_heads,self.chunk_size,self.chunk_size))

        if initial_state_dict is None:
            norm_x = tf.zeros(shape=(bz,1,self.d_model),dtype=inputs_norm.dtype)
            state = self.get_initial_state(bz,inputs.dtype)
        else:
            norm_x = tf.expand_dims(initial_state_dict['time_mix_state_x'],axis=1)
            state = initial_state_dict['time_mix_state_chunk']

        shift_x = tf.concat([norm_x,inputs_norm[:,:-1,:]],axis=1)

        rx_mixed = inputs_norm * self.time_mix_r + (1. - self.time_mix_r) * shift_x
        kx_mixed = inputs_norm * self.time_mix_k + (1. - self.time_mix_k) * shift_x
        vx_mixed = inputs_norm * self.time_mix_v + (1. - self.time_mix_v) * shift_x



        k = tf.matmul(kx_mixed, tf.cast(self.key_weight, dtype=kx_mixed.dtype))
        r = tf.matmul(rx_mixed, tf.cast(self.receptance_weight, dtype=rx_mixed.dtype))
        v = tf.matmul(vx_mixed, tf.cast(self.value_weight, dtype=vx_mixed.dtype))
        #                 0  1          2               3             4

        r = tf.reshape(r, (bz, num_chunks, self.chunk_size, self.num_heads, self.d_head))
        k = tf.reshape(k, (bz, num_chunks, self.chunk_size, self.num_heads, self.d_head))
        v = tf.reshape(v, (bz, num_chunks, self.chunk_size, self.num_heads, self.d_head))



        r = tf.transpose(r, [0, 1, 3, 2, 4])
        k = tf.transpose(k, [0, 1, 3, 4, 2])
        v = tf.transpose(v, [0, 1, 3, 2, 4])




        rkw = tf.matmul(r,k) * w_mask

        rwkv_chunk_separate = tf.matmul(rkw,v) #shape=(bz,num_chunks,num_heads,chunk_size,d_head)

        chunk_bias_array = tf.TensorArray(dtype=inputs_norm.dtype,size=num_chunks,element_shape=(bz,self.num_heads,self.chunk_size,self.d_head))
        for idx in range(num_chunks):
            chunk_bias = tf.matmul(r[:,idx,:,:,:],state) * w_b
            chunk_bias_array = chunk_bias_array.write(idx, chunk_bias)


            state = state * w_chunk + tf.matmul((k[:,idx,:,:,:] * w_k),v[:,idx,:,:,:])

        rwkv_chunk_state = tf.transpose(chunk_bias_array.stack(),[1,0,2,3,4])


        att_outputs = tf.transpose(rwkv_chunk_separate + rwkv_chunk_state,[0,1,3,2,4])

        att_norm = tf.reshape(self.group_norm(att_outputs),(bz,seq_len,self.d_model))#shape=(bz,num_chunks,chunk_size,num_heads,d_head)

        outputs = tf.matmul(att_norm,tf.cast(self.output_weight,dtype=att_norm.dtype))

        return inputs + outputs,(inputs_norm[:,-1,:],state)



class ChannelMix(Layer):
    def __init__(self, layer_idx,config, trainable=True):
        super(ChannelMix, self).__init__(name=f'channel_mix_{layer_idx}', trainable=trainable)
        self.hidden_width = config['hidden_size']

        ln_scale_var = self.add_weight(shape=(1, 1,  config['hidden_size']), dtype=tf.float32,trainable=trainable,name=f'blocks.{layer_idx}.ln2.weight')
        ln_center_var = self.add_weight(shape=(1, 1, config['hidden_size']), dtype=tf.float32,trainable=trainable,name=f'blocks.{layer_idx}.ln2.bias')
        self.input_norm = CustomLayerNormalization(ln_scale_var, ln_center_var, axis=-1, epsilon=1e-5, name="ln3")

        self.time_mix_k = self.add_weight(shape=(1, 1, config['hidden_size']), dtype=tf.float32,trainable=trainable, name=f'blocks.{layer_idx}.ffn.time_mix_k')
        self.time_mix_r = self.add_weight(shape=(1, 1, config['hidden_size']), dtype=tf.float32,trainable=trainable,name=f'blocks.{layer_idx}.ffn.time_mix_r')

        self.key_weight = self.add_weight(shape=(config['hidden_size'], config['expand_size']),dtype=tf.float32,trainable=trainable, name=f'blocks.{layer_idx}.ffn.key.weight')
        self.value_weight = self.add_weight(shape=(config['expand_size'], config['hidden_size']),trainable=trainable,dtype=tf.float32, name=f'blocks.{layer_idx}.ffn.value.weight')
        self.receptance_weight = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),trainable=trainable,dtype=tf.float32, name=f'blocks.{layer_idx}.ffn.receptance.weight')


    def call(self, inputs, initial_state_dict=None):
        x_norm = self.input_norm(inputs)
        bz = tf.shape(x_norm)[0]
        if initial_state_dict is None:
            initial_state = self.get_initial_state(bz, x_norm.dtype)
        else:
            initial_state = initial_state_dict['channel_mix_state_x']

        initial_state = tf.concat([initial_state, x_norm[:, :-1, :]], axis=1)  # 在timestamp维度上拼接

        kx_mixed = x_norm * self.time_mix_k + initial_state * (1. - self.time_mix_k)
        rx_mixed = x_norm * self.time_mix_r + initial_state * (1. - self.time_mix_r)

        r = tf.nn.sigmoid(tf.matmul(rx_mixed, self.receptance_weight))
        vx = tf.matmul(kx_mixed, self.key_weight)
        vx = tf.square(tf.nn.relu(vx))

        out_v = r * tf.matmul(vx, self.value_weight)

        outputs = out_v + inputs

        return outputs, x_norm[:, -1:, :]

    def get_initial_state(self, batch_size=None, dtype=None):

        if batch_size is None:
            batch_size = 1
        if dtype is None:
            dtype = tf.float32
        """
        生成初始的state cell,其中上一状态的输入state_x 使用全零初始化
        """

        state_x = tf.zeros(shape=(batch_size, 1, self.hidden_width), dtype=dtype)

        return state_x


class RWKVModelOutput:
    def __init__(self, rwkv_outputs, rwkv_states=None, kv_cache_states=None):
        self.rwkv_outputs = rwkv_outputs
        self.rwkv_states = rwkv_states
        self.kv_cache_states = kv_cache_states

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise KeyError


class RWKV5(Model):
    def __init__(self, model_config,parallel_mode=False, model_name="RWKV"):
        super(RWKV5, self).__init__(name=model_name)
        self.parallel_mode = parallel_mode

        assert model_config is not None
        self.config = self._build_model_from_config(config=model_config)
        self.num_layers = self.config['num_layers']
        if self.parallel_mode:
            print(f"当前TimeMix处于并行模式下,并行模式下要求输入序列长度必须为seq_chunk={self.config['seq_chunk']}的整数倍")


        start_time = time.time()
        print('开始构建模型')
        self._build_model()
        print('模型构建完成,耗时： %.2fs' % (time.time() - start_time))
        weights_dict = {weight.name.replace('.', '_').strip(':0'): weight for weight in self.trainable_weights}
        self.checkpoint = tf.train.Checkpoint(**weights_dict)
        print(f'模型初始化完成,会使用精度{self.compute_dtype}进行计算')

    def get_checkpoint(self):
        return self.checkpoint

    def _build_model(self, build_batch_size=2, test_seq_len=64):
        self.embedding_layer = CustomEmbedding(self.config)

        self.output_layer = OutputLayer(self.config)

        self.time_mix_layers = [TimeMixChunkParallel(idx,self.config) for idx in range(self.num_layers)]

        self.channel_mix_layers = [ChannelMix(idx,self.config) for idx in range(self.num_layers)]

        random_inputs = tf.zeros(shape=(2,self.config['seq_chunk']*2),dtype=tf.int32)
        outputs = self(random_inputs)


    def export_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _build_model_from_config(self, config: Union[str, Dict[str, Union[int, List[bool]]]]):
        if type(config) == str:
            if not os.path.exists(config):
                raise RWKVModelError("如果传入的config是一个字符串类型，则默认这个config是一个配置文件路径,代码会使用尝试json打开. 但是您传入的文件路径不存在")
            with open(config, mode="r", encoding="utf-8") as fi:
                config = json.load(fi)

        elif type(config) == dict:
            pass
        else:
            raise RWKVModelError("您传入的config必须为一个json格式的配置文件路径或者一个字典类型的配置对象")

        if 'num_layers' not in config:
            raise RWKVModelError("您的config必须包含一个num_layers字段,来标记模型的层数")
        if 'hidden_size' not in config:
            raise RWKVModelError("您的config必须包含一个hidden_size字段,来标记模型的隐藏层宽度")
        if 'vocabulary_size' not in config:
            raise RWKVModelError("您的config必须包含一个vocabulary_size字段,来标记模型的模型的词汇表大小")
        if 'num_heads' not in config:
            raise RWKVModelError("您的config必须包含一个num_heads字段,来标记模型的头的个数,并且确保hidden_size能被num_heads整除")
        else:
            if config['hidden_size'] % config['num_heads'] != 0:
                raise RWKVModelError("请确保config里的hidden_size能被num_heads整除")

        if 'expand_size' not in config:
            config['expand_size'] = 4 * config['hidden_size']
            print(f'没有在config里找到expand_size字段,我们默认ffn里隐藏层宽度扩大四倍,则expand_size={config["expand_size"]}')

        config['head_size'] = config['hidden_size'] // config['num_heads']
        if self.parallel_mode and 'seq_chunk' not in config:
            raise RWKVModelError("在RWKV并行模式下,seq_chunk必须设置,且输入序列长度必须为seq_chunk的整数倍")

        return config


    def forward_sequence(self, inputs, rwkv_states=None, training=False):  # inputs=batch,timestamp

        outputs_sequence = inputs
        if rwkv_states is None:
            rwkv_states = [None for _ in range(self.num_layers)]

        next_states = []
        for idx, (time_mix_layer, channel_mix_layer, layer_states) in enumerate(
                zip(self.time_mix_layers, self.channel_mix_layers, rwkv_states)):
            outputs_sequence, (time_output_state_x,time_output_state_chunk) = time_mix_layer(
                outputs_sequence, initial_state_dict=layer_states)

            outputs_sequence, channel_output_state_x = channel_mix_layer(outputs_sequence,
                                                                         initial_state_dict=layer_states)

            channel_output_state_x = None
            output_states = {
                'time_mix_state_chunk': time_output_state_chunk,
                'time_mix_state_x':time_output_state_x,
                'channel_mix_state_x': channel_output_state_x
            }
            next_states.append(output_states)

        return outputs_sequence, next_states


    def call(self, inputs, rwkv_states=None, return_states=False, training=False):
        """
        :param inputs: 输入维度,如果输入维度为1,即只有batch一个维度,则认为是循环模式,如果inputs是二维即batch,timestamp则认为是并行模式(fake
        :return:outputs,final_states
         outputs的形状与inputs相同
         final_states是四元组结构 由time_mix产生,channel_mix借用四元组里的last_input
        """

        assert len(tf.shape(inputs)) == 2
        embedded_inputs = tf.cast(self.embedding_layer(inputs),self.compute_dtype)


        rwkv_outputs, next_rwkv_states = self.forward_sequence(embedded_inputs, rwkv_states=rwkv_states,
                                                               training=training)

        outputs = self.output_layer(rwkv_outputs)

        if return_states:
            return RWKVModelOutput(rwkv_outputs=outputs, rwkv_states=rwkv_states, kv_cache_states=None)
        else:
            return outputs
