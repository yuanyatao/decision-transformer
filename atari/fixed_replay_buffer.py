# source: https://github.com/google-research/batch_rl/blob/master/batch_rl/fixed_replay/replay_memory/fixed_replay_buffer.py

import collections
from concurrent import futures
from dopamine.replay_memory import circular_replay_buffer
import numpy as np
import tensorflow.compat.v1 as tf
import gin

gfile = tf.gfile

# 存储文件名前缀，来自 circular_replay_buffer 模块
STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX

class FixedReplayBuffer(object):
    """用于加载和管理多个回放缓冲区的类，由一组 OutofGraphReplayBuffers 对象组成。"""

    def __init__(self, data_dir, replay_suffix, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """
        初始化 FixedReplayBuffer 类。
        参数：
          data_dir: str，存储回放缓冲区数据的目录。
          replay_suffix: int，指定加载哪个缓冲区文件的后缀编号。如果为 None，则加载所有缓冲区。
          *args: 传递给回放缓冲区的额外位置参数。
          **kwargs: 传递给回放缓冲区的额外关键字参数。
        """
        self._args = args
        self._kwargs = kwargs
        self._data_dir = data_dir
        self._loaded_buffers = False  # 标记是否已加载缓冲区
        self.add_count = np.array(0)  # 初始化已添加的记录数
        self._replay_suffix = replay_suffix  # 设置要加载的缓冲区后缀编号
        
        # 根据 replay_suffix 是否为空，决定加载单个还是多个缓冲区
        if not self._loaded_buffers:
            if replay_suffix is not None:
                assert replay_suffix >= 0, '请输入非负的缓冲区后缀编号'
                self.load_single_buffer(replay_suffix)  # 加载单个缓冲区
            else:
                self._load_replay_buffers(num_buffers=50)  # 默认加载 50 个缓冲区

    def load_single_buffer(self, suffix):
        """加载指定后缀编号的单个回放缓冲区文件。
        
        参数：
          suffix: int，要加载的回放缓冲区文件的后缀编号。
        """
        replay_buffer = self._load_buffer(suffix)
        if replay_buffer is not None:
            self._replay_buffers = [replay_buffer]  # 只保存单个缓冲区
            self.add_count = replay_buffer.add_count  # 更新添加记录数
            self._num_replay_buffers = 1
            self._loaded_buffers = True  # 标记缓冲区已加载

    def _load_buffer(self, suffix):
        """加载一个具有指定后缀编号的 OutOfGraphReplayBuffer 缓冲区。
        
        参数：
          suffix: int，缓冲区的后缀编号。
          
        返回：
          replay_buffer: OutOfGraphReplayBuffer 对象，如果文件加载失败，则返回 None。
        """
        try:
            # 创建 OutOfGraphReplayBuffer 实例，并从指定后缀文件加载数据
            replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
                *self._args, **self._kwargs)
            replay_buffer.load(self._data_dir, suffix)  # 从文件加载数据
            tf.logging.info('已加载回放缓冲区检查点 {} 从 {}'.format(suffix, self._data_dir))
            return replay_buffer
        except tf.errors.NotFoundError:
            # 文件未找到时返回 None
            return None

    def _load_replay_buffers(self, num_buffers=None):
        """加载指定数量的回放缓冲区文件到缓冲区列表中。
        
        参数：
          num_buffers: int，要加载的缓冲区数量。如果为 None，则加载所有缓冲区。
        """
        if not self._loaded_buffers:
            # 列出数据目录中的所有检查点文件
            ckpts = gfile.ListDirectory(self._data_dir)
            
            # 统计每个后缀编号的文件个数
            ckpt_counters = collections.Counter(
                [name.split('.')[-2] for name in ckpts])
            
            # 筛选出有效的后缀编号，通常每个缓冲区包含 6-7 个文件
            ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
            
            # 如果指定了加载的缓冲区数量，则随机选取相应数量的后缀编号
            if num_buffers is not None:
                ckpt_suffixes = np.random.choice(
                    ckpt_suffixes, num_buffers, replace=False)
            
            self._replay_buffers = []
            
            # 使用多线程并行加载缓冲区文件
            with futures.ThreadPoolExecutor(
                max_workers=num_buffers) as thread_pool_executor:
                replay_futures = [thread_pool_executor.submit(
                    self._load_buffer, suffix) for suffix in ckpt_suffixes]
            
            # 收集所有加载的缓冲区
            for f in replay_futures:
                replay_buffer = f.result()
                if replay_buffer is not None:
                    self._replay_buffers.append(replay_buffer)  # 将加载的缓冲区添加到列表
                    self.add_count = max(replay_buffer.add_count, self.add_count)  # 更新最大记录数
            
            # 更新已加载缓冲区的数量和标记
            self._num_replay_buffers = len(self._replay_buffers)
            if self._num_replay_buffers:
                self._loaded_buffers = True  # 标记已加载

    def get_transition_elements(self):
        """返回转移元素的结构描述，用于采样。
        
        返回：
          List，描述转移元素（如状态、动作、奖励等）的结构。
        """
        return self._replay_buffers[0].get_transition_elements()

    def sample_transition_batch(self, batch_size=None, indices=None):
        """从回放缓冲区中随机抽样一批转移。
        
        参数：
          batch_size: int，采样的批次大小。
          indices: list，可选的指定索引列表。
          
        返回：
          从随机选择的回放缓冲区中采样的转移批次。
        """
        # 随机选择一个缓冲区进行采样
        buffer_index = np.random.randint(self._num_replay_buffers)
        return self._replay_buffers[buffer_index].sample_transition_batch(
            batch_size=batch_size, indices=indices)

    def load(self, *args, **kwargs):  # pylint: disable=unused-argument
        """未实现的加载方法，作为占位符。"""
        pass

    def reload_buffer(self, num_buffers=None):
        """重新加载缓冲区数据。
        
        参数：
          num_buffers: int，要加载的缓冲区数量。
        """
        self._loaded_buffers = False
        self._load_replay_buffers(num_buffers)  # 重新加载缓冲区

    def save(self, *args, **kwargs):  # pylint: disable=unused-argument
        """未实现的保存方法，作为占位符。"""
        pass

    def add(self, *args, **kwargs):  # pylint: disable=unused-argument
        """未实现的添加方法，作为占位符。"""
        pass
