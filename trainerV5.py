import os.path
import time


from tqdm import tqdm
from keras.metrics import Metric
from RWKV_5_MODEL import RWKV5
from keras.optimizers import Adam
try:
    from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule, CosineDecay
except:
    from keras.optimizers.schedules import LearningRateSchedule, CosineDecay
from keras.losses import categorical_crossentropy
from keras.mixed_precision import set_global_policy, global_policy
import tensorflow as tf
import argparse
from keras.mixed_precision import LossScaleOptimizer
from TranslationDataset import RWKVDataset


class CosineDecayWarmup(LearningRateSchedule):

    def __init__(self, init_lr, steps, warmup_steps, min_lr):
        super(CosineDecayWarmup, self).__init__()

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.cosine_decay = CosineDecay(
            init_lr, steps - warmup_steps, min_lr)

    def __call__(self, step):
        linear_increase = self.init_lr * tf.cast(step, tf.float32) / (
                tf.cast(self.warmup_steps, tf.float32) + 1e-5)
        cosine_decay = self.cosine_decay(step)
        return tf.cond(pred=step <= self.warmup_steps,
                       true_fn=lambda: linear_increase,
                       false_fn=lambda: cosine_decay)

    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'init_lr': self.init_lr
        }

class MySchedule(LearningRateSchedule):
    def __init__(self,learning_rate,final_learning_rate,warmup_steps,decay_steps):
        super(MySchedule,self).__init__()
        self.lr = tf.convert_to_tensor(learning_rate,dtype=tf.float32)
        self.final_lr= tf.convert_to_tensor(final_learning_rate,dtype=tf.float32)
        tf.assert_greater(self.lr,self.final_lr)
        self.warmup_steps =  warmup_steps
        self.decay_steps =  decay_steps
        self.cosine_decay = CosineDecay(self.lr - self.final_lr,self.decay_steps - self.warmup_steps)
    def __call__(self,step):
        cur_step = tf.cast(step,dtype=tf.float32)
        def warmup():
            return self.lr * cur_step / (tf.cast(self.warmup_steps,dtype=tf.float32) + 1e-8)
        def cosine():
            return self.cosine_decay(cur_step - self.warmup_steps) + self.final_lr
        def constant():
            return self.final_lr
        learning_rate = tf.case([(tf.less_equal(step,self.warmup_steps),warmup),
                (tf.less_equal(step,self.decay_steps),cosine)],default=constant)
        return learning_rate

class MeanWithCapacity(Metric):
    def __init__(self, capacity):
        super(MeanWithCapacity, self).__init__()
        self.capacity = capacity
        super(MeanWithCapacity, self).__init__()
        self.num_list = []

    def update_state(self, value):
        value = tf.cast(value, dtype=tf.float32)
        if len(self.num_list) == self.capacity:
            self.num_list.pop(0)
        self.num_list.append(value)

    def result(self):
        if len(self.num_list) == 0:
            return tf.constant(0., dtype=tf.float32)
        stacked = tf.stack(self.num_list, axis=0)
        return tf.reduce_mean(stacked)




class RWKV5Trainer:
    def __init__(self, args):

        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.train_steps = args.train_steps
        self.warmup_steps = args.warmup_steps
        self.start_steps = args.start_steps
        self.save_freq = args.save_freq
        self.final_learning_rate = args.final_learning_rate
        self.decay_steps = args.decay_steps
        self.num_heads = args.num_heads
        self.vocabulary_size = args.vocabulary_size
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.seq_chunk = args.seq_chunk
        self.enable_distribute = args.distribute
        self.enable_fp16 = args.fp16
        self.global_clip_var = args.global_clip_var
        self.accumulate_batch = args.accumulate_batch


        if self.enable_fp16:
            set_global_policy('mixed_float16')
            print('您启用了fp16训练')

        model_config = {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'vocabulary_size': self.vocabulary_size,
            'num_heads': self.num_heads,
            'seq_chunk': self.seq_chunk,
        }
        def create_model():
            return RWKV5(model_config=model_config, parallel_mode=True)

        def create_optimizer():
            lr_schedule = MySchedule(self.learning_rate, self.final_learning_rate, self.warmup_steps, self.decay_steps)
            try:
                optimizer = Adam(learning_rate=lr_schedule, global_clipnorm=self.global_clip_var,jit_compile=False)
            except:
                optimizer = Adam(learning_rate=lr_schedule, global_clipnorm=self.global_clip_var )
            # optimizer = Lion(learning_rate=lr_schedule,global_clipnorm=self.global_clip_var,jit_compile=True)
            optimizer.iterations.assign(self.start_steps)
            if self.enable_fp16:
                optimizer = LossScaleOptimizer(optimizer)
            return optimizer

        if self.enable_distribute:
            self.strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice('/cpu:0'))
            if self.accumulate_batch > 0:
                with self.strategy.scope():
                    self.rwkv_model = create_model()
                with tf.device('/cpu:0'):
                    self.optimizer = create_optimizer()
                print('您的策略选择为开启分布式训练+梯度累积,在这部分中模型会在CPU上保存并在每个GPU上保存一个副本. 梯度会在从GPU上计算完成后发回CPU.在CPU上完成累积更新,最终同步到GPU的副本上')
            else:
                with self.strategy.scope():
                    self.rwkv_model, self.optimizer = create_model(),create_optimizer()
                print('mirror策略分布式训练 优化器模型均保存在GPU上，求和操作会发送到CPU上，由CPU完成，抑制峰值显存消耗')
        else:
            self.rwkv_model, self.optimizer = create_model(), create_optimizer()






        self.logs_dir = os.path.join(args.logs_dir, '%d-%m-%d_%H_%M_%S')
        self.models_dir = args.models_dir
        self.ckpt_keeps = args.ckpt_keeps
        self.ckpt_manager = tf.train.CheckpointManager(self.rwkv_model.get_checkpoint(), self.models_dir,
                                                       max_to_keep=self.ckpt_keeps)
        self.rwkv_dataset = RWKVDataset(args)
        if self.accumulate_batch == 0:
            self.train_on_step = tf.function(func=self.train_on_step_mirror if self.enable_distribute else self.train_on_step_inst,jit_compile=False)

        else:
            with tf.device('/cpu:0'):
                self.zero_variables = [tf.zeros(var.shape) for var in self.rwkv_model.trainable_weights]
                self.copy_variables = [tf.Variable(var) for var in self.rwkv_model.trainable_weights]
    def train(self):
        self.ckpt_manager.restore_or_initialize()
        logs_dir = time.strftime(self.logs_dir)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        print(f'日志文件将保存在{logs_dir}目录下')

        writer = tf.summary.create_file_writer(logs_dir)
        capacity = 200
        loss_mean = MeanWithCapacity(capacity=capacity)
        top5_mean = MeanWithCapacity(capacity=capacity)

        train_dataset = self.rwkv_dataset.load_dataset("train")
        if self.enable_distribute:
            if self.accumulate_batch >0:
                train_dataset = train_dataset.batch(self.accumulate_batch).batch(self.batch_size)
            else:
                train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            train_dataset = train_dataset.batch(self.batch_size)

        train_dataset = iter(train_dataset)
        with tqdm(initial=self.start_steps, total=self.train_steps) as bar:
            for step in range(self.start_steps, self.train_steps):
                # print(next(train_dataset))
                if self.accumulate_batch>0 and self.enable_distribute:
                    step_loss,step_top5=self.train_on_batch_accumulation_mirror(*next(train_dataset))
                else:
                    step_loss,step_top5 = self.train_on_step(*next(train_dataset))

                loss_mean.update_state(step_loss)
                top5_mean.update_state(step_top5)
                bar.set_postfix({
                    'train_loss': '%.2f' % loss_mean.result(),
                    'train_top5': '%.2f' % top5_mean.result()
                })
                bar.update()
                with writer.as_default(step=step):
                    tf.summary.scalar("train_loss", step_loss)
                    tf.summary.scalar("train_top5_acc", step_top5)
                if (step + 1) % self.save_freq == 0:
                    self.ckpt_manager.save()

    def top_k_accuracy_metric(self, logits, labels, top_k=5):
        labels = tf.expand_dims(labels, axis=-1)  # bz,seq_len,1
        _, indices = tf.nn.top_k(logits, k=top_k, sorted=False)
        indices = tf.cast(indices, labels.dtype)
        is_equal_top_k = tf.cast(tf.reduce_any(tf.equal(labels, indices), axis=-1), dtype=tf.float32)
        return tf.reduce_mean(is_equal_top_k)

    """
        下面四个函数实现的逻辑是模型在GPU计算梯度,然后返回到CPU上 进行梯度累计
        当达到累积的步数以后把梯度送回GPU更新
    """

    def grad_on_step_mirror(self, inputs, labels, label_masks,idx):
        def grad_accumulate_batch(_inputs, _labels, _label_masks,_idx):
            return self.train_on_step_inst(_inputs[:,_idx,...], _labels[:,_idx,...], _label_masks[:,_idx,...])
        per_grads,per_step_loss, per_step_top5 = self.strategy.run(grad_accumulate_batch,args=(inputs, labels, label_masks,idx))

        grads = [self.strategy.reduce(tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in per_grads]
        step_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_step_loss, axis=None)
        step_top5 = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_step_top5, axis=None)
        #tf.print('out grad',grads[0].device)
        return grads,step_loss,step_top5


    
    def sum_grads(self,grads1,grads2):
        grads = [tf.add_n(grad_zip) / self.accumulate_batch for grad_zip in zip(grads1,grads2)]
        #print('sum device:',grads[0].device)
        return grads


    
    def update_accumulate_mirror(self,grads):
        tf.print('final_d',grads[0].device)
        self.optimizer.apply_gradients(zip(grads, self.copy_variables))
        def update_model(_variables):
            for gpu_var,cpu_var in zip(self.rwkv_model.trainable_weights,_variables):
                gpu_var.assign(cpu_var)
        print('w_device:',self.copy_variables[0].device)
        self.strategy.run(update_model, args=(self.copy_variables,))

    @tf.function
    def train_on_batch_accumulation_mirror(self,inputs, labels, label_masks):
        accumulation_grads = self.zero_variables
        step_loss_acc = 0.
        step_top5_acc = 0.
        for idx in range(self.accumulate_batch):
            
            step_grads,step_loss,step_top5 = self.grad_on_step_mirror(inputs,labels,label_masks,idx)
            #tf.print(step_grads[0].device)
            accumulation_grads = self.sum_grads(accumulation_grads,step_grads)
            step_loss_acc += step_loss
            step_top5_acc += step_top5
            tf.print('acc_grads:',accumulation_grads[0].device)
        self.update_accumulate_mirror(accumulation_grads)
        return step_loss_acc / self.accumulate_batch,step_top5_acc/self.accumulate_batch

    @tf.function(jit_compile=False)
    def train_on_step_mirror(self,inputs, labels, label_masks):
        per_step_loss, per_step_top5 = self.strategy.run(self.train_on_step_inst,args=(inputs, labels, label_masks))
        step_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_step_loss, axis=None)
        step_top5 = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_step_top5, axis=None)
        return step_loss,step_top5


    def train_on_step_inst(self, inputs, labels, label_masks):
        with tf.GradientTape() as tape:
            logits = self.rwkv_model(inputs, rwkv_states=None, return_states=False, training=True)
            label_masks = tf.cast(label_masks, dtype=logits.dtype)  #
            labels_one_hot = tf.one_hot(labels, tf.shape(logits)[-1])
            cross_loss = categorical_crossentropy(labels_one_hot, logits, from_logits=True, label_smoothing=0.05)

            losses_a = tf.reduce_sum(label_masks * cross_loss, axis=-1)
            losses_b = tf.reduce_sum(label_masks, axis=-1)

            loss = tf.math.divide_no_nan(losses_a, losses_b)
            loss = tf.nn.compute_average_loss(loss,global_batch_size=self.batch_size)
            if self.enable_fp16:
                loss_scaled = self.optimizer.get_scaled_loss(loss)
            else:
                loss_scaled = loss

        grads = tape.gradient(loss_scaled, self.rwkv_model.trainable_weights)
        top5_acc = self.top_k_accuracy_metric(logits, labels)

        if self.enable_fp16:
            grads = self.optimizer.get_unscaled_gradients(grads)

        if self.accumulate_batch > 0:
            return grads, loss, top5_acc
        else:
            self.optimizer.apply_gradients(zip(grads, self.rwkv_model.trainable_weights))
            return loss,top5_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', default=192, type=int, required=False, help="训练时候使用的序列长度")
    parser.add_argument('--vocabulary-size', default=65536, type=int, required=False, help="RWkV5模型的词汇表大小")
    parser.add_argument('--num-layers', default=24, type=int, required=False, help="RWKV5模型的层数")
    parser.add_argument('--hidden-size', default=1024, type=int, required=False, help="RWKV5模型的隐藏层宽度")
    parser.add_argument('--num-heads', default=16, type=int, required=False, help="RWKV5模型的头的个数")
    parser.add_argument('--seq-chunk', default=32, type=int, required=False, help="RWKV5模型在并行模式下,并行运算的序列长度")
    parser.add_argument('--ckpt-keeps', default=5, type=int, required=False, help="模型保存的最大副本数量")
    parser.add_argument('--accumulate-batch',default=0,type=int,required=False,help='是由启用梯度累积功能,0代表关闭这个功能,>=1'
        '代表累积的batch次数.区别在于如果关闭此功能优化器更新将在GPU间完成,>=1时候CPU会参与计算,把返回的梯度在CPU上累积后返回GPU最终使用优化器更新权重')
    parser.add_argument('--batch-size', default=6, type=int, required=False, help='训练时候的批处理大小')
    parser.add_argument('--save-freq', default=6000, type=int, required=False, help='训练保存模型的频率,模型保存的周期')
    parser.add_argument('--train-steps', default=800000, type=int, required=False, help='训练时候总共训练的步数')
    parser.add_argument('--learning-rate', default=2.5e-4, type=float, required=False,
                        help='训练的学习率,这个开始指的是step=warnup-steps时的学习率')
    parser.add_argument('--decay-steps',default=100000,type=int,required=False,help='训练使用余弦衰减的步数,步数超过这个值会使用恒定学习率')
    parser.add_argument('--final-learning-rate',default=4.e-5,type=float,required=False,help='当迭代步数超过decay-steps时使用的学习率')
    parser.add_argument('--global-clip-var', default=2.5, type=float, required=False,
                        help='使用GlobalNormal进行梯度裁剪的阈值，超过这个阈值梯度被缩放，有利于训练稳定')
    parser.add_argument('--warmup-steps', default=4000, type=int, required=False,
                        help='热启动的步数,学习率从0上升到learning-rate指定的学习率,在step=warmup-steps时实现')
    parser.add_argument('--start-steps', default=0, type=int, required=False,
                        help='是否跳过一些训练steps,从中间开始训练.设置为0从头开始训练')
    parser.add_argument('--train', action='store_true', default=True, help='添加这个参数表示将进行训练')
    parser.add_argument('--logs-dir', default='logs/', type=str, required=False, help='设置日志文件的目录')
    parser.add_argument('--models-dir', default='/home/niconiconi/diskF/translate/checkpoint', type=str, required=False,
                        help='设置输出模型的保存目录')
    parser.add_argument("--train-dataset-dir",
                        #default=r"C:/Users/a1313/PycharmProjects/MyTEXTCleaner/translation/*.tfrecord", type=str,
                        default=r"/home/niconiconi/diskF/rwkv_dataset/translation/*.tfrecord", type=str,
                        required=False)
    parser.add_argument("--valid-dataset-dir",
                        default=r"C:\Users\a1313\PycharmProjects\MyTEXTCleaner\translation\*.tfrecord", type=str,
                        required=False)
    parser.add_argument("--test-dataset-dir",
                        default=r"C:\Users\a1313\PycharmProjects\MyTEXTCleaner\translation\*.tfrecord", type=str,
                        required=False)
    parser.add_argument("--dataset-weights",default='', type=str, required=False)
    parser.add_argument("--distribute", action='store_true', help="是否启用分布式训练,添加此参数意味着启用分布式训练")
    parser.add_argument("--fp16", action='store_true', help="是否启用fp16精度训练,添加此参数意味着启用半精度训练")
    parser.add_argument("--stateful", action='store_true', help="是否允许状态在不同的batch之间传递,添加此参数意味着启用状态传递,并且会禁用数据集的随机打乱功能(只允许使用一种数据集).")
    args = parser.parse_args()
    trainer = RWKV5Trainer(args)
    if args.train:
        print('开始进行训练..')
        trainer.train()
