import numpy as np

from torch.utils.data import Sampler
from typing import Sized, Optional


class OpusSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sized,
        temperature: float = 1.0,
        seed: int = 42,
        num_samples: Optional[int] = None,
    ):
        self.data_source = data_source
        self.temperature = temperature

        self.seed = seed
        self._num_samples = num_samples

        self.utility = np.zeros(len(self.data_source))
        self.rng = np.random.default_rng(self.seed)
        self.epoch = 0

        self.sample_counter = np.zeros(len(self.data_source))
        self.update_step = 0


    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples


    def debug_info(self):
        u = self.utility
        print("===== OPUS Sampler Stats =====")
        print(f"utility mean : {u.mean():.6f}")
        print(f"utility std  : {u.std():.6f}")
        print(f"utility max  : {u.max():.6f}")
        print(f"utility p90  : {np.percentile(u, 90):.6f}")
        print(f"utility p95  : {np.percentile(u, 95):.6f}")
        print(f"utility p99  : {np.percentile(u, 99):.6f}")

        probs = self.get_probs()
        print("===== OPUS Sampling =====")
        print(f"prob mean : {probs.mean():.6f}")
        print(f"prob max  : {probs.max():.6f}")
        print(f"prob min  : {probs.min():.6f}")
        print(f"top1 prob : {np.sort(probs)[-1]:.6f}")
        print(f"top10 prob : {np.sort(probs)[-10:].sum():.6f}")

        print("===== Sampling Frequency =====")
        print(f"mean freq : {self.sample_counter.mean():.2f}")
        print(f"max freq  : {self.sample_counter.max():.2f}")
        print(f"p90 freq  : {np.percentile(self.sample_counter, 90):.2f}")
        print(f"p80 freq  : {np.percentile(self.sample_counter, 80):.2f}")

        topk = np.argsort(self.utility)[-10:]
        print("===== Hardest Samples =====")
        for idx in topk:
            print(idx, self.utility[idx])


    def update(self, indices, values, ema=0.9):
        for idx, val in zip(indices, values):
            old = self.utility[idx]
            self.utility[idx] = ema * old + (1 - ema) * val

        self.update_step += 1
        if self.update_step % 20 == 0:
            self.debug_info()

    def get_probs(self):
        u = self.utility

        p95 = np.percentile(u, 95)
        u_clipped = np.clip(u, 0, p95)

        # 2. 避免 Softmax 的指数崩塌，改用成比例映射 (Proportional Mapping)
        # 加上一个小常数 epsilon 防止概率完全为 0
        epsilon = 1e-5
        weights = (u_clipped + epsilon) ** self.temperature
        dynamic_probs = weights / weights.sum()

        # 3. 引入保底均匀分布 (例如 20% 的概率从均匀分布中采，80% 按困难度采)
        # 这能确保即便是不太重要的样本 (如 p85 的那批数据) 也能被复习到
        exploration_ratio = 0.35
        uniform_probs = np.ones_like(dynamic_probs) / len(dynamic_probs)

        probs = exploration_ratio * uniform_probs + (1 - exploration_ratio) * dynamic_probs

        return probs

    def __iter__(self):
        self.epoch += 1

        if self.epoch == 1:
            indices = self.rng.choice(
                self.num_samples,
                size=self.num_samples,
                replace=False
            ).tolist()
            counts = np.ones(self.num_samples)
        else:
            probs = self.get_probs()

            # 4. 高效硬截断：通过直接限制最大概率来软性限制最大采样频次
            # 假设我们希望一个样本最多被采样约 max_freq 次 (比如 3 次)
            max_freq = 3.0
            max_prob_threshold = max_freq / self.num_samples

            # 迭代归一化，确保没有样本概率超过阈值 (通常循环 2-3 次即可收敛)
            for _ in range(3):
                probs = np.minimum(probs, max_prob_threshold)
                probs /= probs.sum()

            # 5. 向量化一次性采样，彻底消除低效的 for 循环
            indices = self.rng.choice(
                self.num_samples,
                size=self.num_samples,
                p=probs,
                replace=True
            ).tolist()

            # 统计本次采样的频次分布用于 debug_info
            counts = np.zeros(self.num_samples)
            unique, counts_vals = np.unique(indices, return_counts=True)
            counts[unique] = counts_vals

        self.sample_counter = counts
        return iter(indices)


    def __len__(self) -> int:
        return self.num_samples
