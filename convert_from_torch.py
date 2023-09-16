import gc
from typing import Dict

import numpy as np
import tensorflow as tf
from keras import Model


class CheckpointSaver(Model):
    def __init__(self,file_path):
        self.file_path = file_path
        super(CheckpointSaver, self).__init__()

        try:
            import torch
            ckpt_dict = torch.load(file_path, map_location="cpu")
            gc.collect()
            hidden_size = ckpt_dict['emb.weight'].shape[-1]
            num_heads = ckpt_dict['blocks.0.att.time_first'].shape[0]
            assert hidden_size % num_heads ==0,"隐藏层宽度必须是num_heads头的个数的整数倍,一般情况下请检查您的num_heads个数"
            var_keys = set(ckpt_dict.keys())
            tf_variables_dict = {}
            for var_name in var_keys:
                tensor = ckpt_dict[var_name]
                tensor = tensor.detach().to('cpu', dtype=torch.float32).numpy()

                if 'time_decay' in var_name or  'time_first' in var_name: #这两个的shape=(hidden_size,)需要在前面补充一个batch_size维度
                    tensor = np.reshape(tensor,(-1,))

                if 'time_mix' in var_name: #这些tensor的shape=(1,1,hidden_size) 我在这里使用循环展开处理rwkv 因此中间的时间维度需要去除
                    tensor = np.reshape(tensor,(1,1,-1))
                if any([k in var_name for k in ['ln_out','ln0','ln1','ln2']]):
                    tensor = np.reshape(tensor,(1,1,-1))

                if 'ln_x' in var_name:
                    tensor = np.reshape(tensor,(1,1,1,num_heads,hidden_size // num_heads))

                if 'key.weight' in var_name or 'value.weight' in var_name\
                        or 'receptance.weight' in var_name or 'output.weight' in var_name or 'head.weight' in var_name:
                    tensor =tensor.T


                tf_weights = self.add_weight(name=var_name,initializer=tf.constant_initializer(tensor),shape=tensor.shape)
                del ckpt_dict[var_name]
                tf_variables_dict[var_name] = tf_weights
            gc.collect()
            self.checkpoint = tf.train.Checkpoint(**{key.replace('.', '_'): var for key, var in tf_variables_dict.items()})
        except ModuleNotFoundError as e:
            print('如果要从pth载入模型,请安装pytorch,cpu版本即可')
    def save_to_ckpt(self,filepath_with_prefix):
        self.checkpoint.save(filepath_with_prefix)

if __name__ == '__main__':
    weight_filepath = r"F:/translate/RWKV-5-World-0.1B-v1-OnlyForTest_37%_trained-20230728-ctx4096.pth"
    CheckpointSaver(weight_filepath).save_to_ckpt("./converted_model/world-0.1B")
