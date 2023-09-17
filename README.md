# RWKV5-Tensorflow2.0
全新的RWKV5 Tensorflow2.0实现,支持多GPU训练，这个版本绝对不鸽。

  RWKV5相比于RWKV4最大的进步在于K从万恶的EXP上下来了（RWKV4 EXPK会在float32下出现数字上溢为inf，如果使用logsumexp技巧又会在float32下出现数值下溢为0），并且RWKV5使用GroupNormalization代替了RWKV4中加权softmax实现归一化，大大降低了计算复杂度。

  数据集使用tfrecord格式存储、每个记录包含两个标签：inputs_and_labels（int64类型 shape=(seq_len+1,)）与label_masks（float32类型 shape=(seq_len,)），分别对应训练数据与位置权重。 其中inputs_and_labels[:-1]会被作为训练数据的输入、inputs_and_labels[1:]会被作为训练数据的输出。
  tfrecord文件必须统一按照: 前缀_数字id.tfrecord 格式命名，否则无法加载。

  代码支持同时加载多种数据集并为其分配不同的权重,使用--dataset-weights "json"参数指定数据集的权重,其中json代表字符串类型的json字典,其中key数据集的tfrecord文件前缀、value则是对应数据集的权重。如果只是用一种数据集请确保参数: --dataset-weights的值为""

  代码支持半精度fp16训练，请使用--fp16参数来开启；代码同样支持分布式训练，请使用--distribute参数来开启，默认的分布式策略是mirror镜像模式，但是梯度求和会发送到CPU上进行，抑制峰值显存需求。

  添加--accumulate-batch 积累步数>0 参数可以开启梯度累积功能，添加--accumulate-batch参数后使用完全不同的分布式策略，模型张量会保存在CPU、并与显卡显存内的副本实时同步。模型计算完成的梯度会直接发送到CPU上，梯度累积将在CPU上完成，紧接着的梯度更新也将在CPU上完成，最后由CPU向显卡同步模型副本，开启下一轮更新循环。

  其中convert_from_torch.py可以把pytorch的pth权重转化为tensorflow的checkpoint权重，而convert_to_torch.py文件则可以把训练得到的checkpoint权重转换成pytorch的pth权重。注意！ 使用这两个文件需要确保同时安装了tensorflow与pytorch库。

  这是我小说翻译模型的一部分，训练好的翻译模型将会拥有300M的参数量，并且在拥有连续上下文的平行语料上进行微调。
