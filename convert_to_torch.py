import gc
import re
from typing import Dict
from tensorflow.python.training import py_checkpoint_reader
import numpy as np
import tensorflow as tf
from keras import Model
import torch

def replace_key(key_name):
    key_name = key_name.replace("time.decay","time_decay")
    key_name = key_name.replace("time.first","time_first")
    key_name = key_name.replace("time.mix.r","time_mix_r")
    key_name = key_name.replace("time.mix.k","time_mix_k")
    key_name = key_name.replace("time.mix.v","time_mix_v")
    key_name = key_name.replace("ln.x","ln_x")
    key_name = key_name.replace("ln.out","ln_out")
    return key_name

class CheckpointConvertor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.reader = py_checkpoint_reader.NewCheckpointReader(self.file_path)
        self.weight_dict = {key.strip('/.ATTRIBUTES/VARIABLE_VALUE').replace('_','.'): self.reader.get_tensor(key) for
                            key, _ in self.reader.get_variable_to_shape_map().items() if 'save_counter' not in key and 'VARIABLE_VALUE' in key}


        self.weight_dict = {replace_key(key): var  for key,var in self.weight_dict.items()}
        print(self.weight_dict.keys())
        #print(self.reader.get_variable_to_shape_map().items())
    def save_to_ckpt(self, out_path):
        weight_dict_torch = {}
        for var_name, var_array in self.weight_dict.items():

            if 'time_decay' in var_name or 'time_first' in var_name:  # 这两个的shape=(hidden_size,)需要在前面补充一个batch_size维度

                var_array = np.reshape(var_array, (-1,))

            if 'time_mix' in var_name:
                var_array = np.reshape(var_array, (1, 1, -1))

            if any([k in var_name for k in ['ln_out', 'ln0', 'ln1', 'ln2']]):
                var_array = np.reshape(var_array, (-1,))

            if 'ln_x' in var_name:
                var_array = np.reshape(var_array, (-1,))

            if 'key.weight' in var_name or 'value.weight' in var_name \
                    or 'receptance.weight' in var_name or 'output.weight' in var_name or 'head.weight' in var_name:
                var_array = var_array.T
            var_array = var_array.astype('float32')
            #print(var_name,type(var_array))
            weight_dict_torch[var_name] = torch.from_numpy(var_array)
        torch.save(weight_dict_torch,out_path)



if __name__ == '__main__':
    #weight_filepath2 = r"out.pth"
    weight_filepath2 = r"F:/translate/RWKV-5-World-0.1B-v1-OnlyForTest_37%_trained-20230728-ctx4096.pth"

    weight_dict = torch.load(weight_filepath2, map_location='cpu')
    print({key: var.shape for key, var in weight_dict.items()})
    weight_filepath = r"F:\translate\checkpoint\ckpt-65"
    convertor = CheckpointConvertor(weight_filepath)
    convertor.save_to_ckpt('out.pth')
