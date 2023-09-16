import glob
import json
import os.path
import random
import re
from collections import defaultdict

import tensorflow as tf
from tqdm import tqdm

import RWKV_5_MODEL
from BASE_LAYERS import RWKVModelError


class RWKVDataset:
    def __init__(self,args):
        self.seq_len = args.seq_len
        #self.batch_size = args.batch_size
        self.train_dataset_dir = args.train_dataset_dir
        self.valid_dataset_dir = args.valid_dataset_dir
        self.test_dataset_dir = args.test_dataset_dir
        self.dataset_weights = args.dataset_weights
        if self.dataset_weights == '':
            self.dataset_weights = None
        else:
            self.dataset_weights = json.loads(self.dataset_weights)
    def map_fun(self,record):
            feature = {
                "inputs_and_labels": tf.io.FixedLenFeature((self.seq_len+1,), dtype=tf.int64),
                "label_masks": tf.io.FixedLenFeature((self.seq_len,), dtype=tf.float32)
            }
            example = tf.io.parse_single_example(record, feature)
            inputs_and_labels = tf.cast(example['inputs_and_labels'],dtype=tf.int32)
            label_masks = example['label_masks']
            inputs = inputs_and_labels[:-1]
            labels = inputs_and_labels[1:]
            return inputs, labels,label_masks

    def __create_dataset(self,dataset_files):
        datasets = tf.data.TFRecordDataset(dataset_files, num_parallel_reads=2).shuffle(buffer_size=10240, seed=1234) \
            .map(self.map_fun, num_parallel_calls=3)
        if dataset_files in ['train', 'valid']:
            datasets = datasets.repeat()
        return datasets

    def load_dataset(self,dataset_prefix):
        if dataset_prefix == 'train':
            data_dir = self.train_dataset_dir
        elif dataset_prefix == 'valid':
            data_dir = self.valid_dataset_dir
        else:
            data_dir = self.test_dataset_dir

        filepaths = glob.glob(data_dir)
        filenames = [os.path.basename(fp) for fp in filepaths]
        pattern = r"^(.+?)_\d+\.tfrecord"

        group_files = defaultdict(list)
        for filepath,filename in zip(filepaths,filenames):
            res = re.match(pattern,filename)
            if res is None:
                raise RWKVModelError("tfrecords文件必须满足 前缀_文件下标索引.tfrecord格式")
            group_files[res.group(1)].append(filepath)
            #print(res.group(1))

        if self.dataset_weights is None:
            if len(group_files.keys()) !=1:
                raise RWKVModelError("--dataset-weights 如果您使用多种tfrecords数据集必须传入一个json字符串格式的字典,key为文件的前缀,value为这个数据集的权重")
            datasets = self.__create_dataset(filepaths)

        else:
            if set(group_files.keys()) != set(self.dataset_weights.keys()):
                raise RWKVModelError("--dataset-weights 如果您使用多种tfrecords数据集必须传入一个json字符串格式的字典,并且这个字典的key与tfrecords文件的前缀相同")
            #if :
            task_weights = []
            task_datasets = []
            for task_key,task_weight in self.dataset_weights.items():
                task_dataset = self.__create_dataset(group_files[task_key])
                task_weights.append(task_weight)
                task_datasets.append(task_dataset)



            datasets = tf.data.Dataset.sample_from_datasets(task_datasets,weights=task_weights)
        datasets = datasets.prefetch(tf.data.AUTOTUNE)#.batch(self.batch_size)
        return datasets

