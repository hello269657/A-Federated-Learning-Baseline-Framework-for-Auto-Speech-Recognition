import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # 继承Client类的所有属性

    def train(self):
        self.model.train()
        trainloader = self.load_train_data()
        start_time = time.time()

        # 核心新增：输出当前客户端的样本数（训练+测试）
        print("=" * 50)
        print(f"【客户端{self.id} 样本信息】")
        print(f"训练样本总数：{self.train_samples}")
        print(f"测试样本总数：{self.test_samples}")
        print(f"本轮训练批次数量：{len(trainloader)}")  # 输出批次数量
        print(f"批次大小：{self.batch_size}")  # 输出批次大小
        print("=" * 50)

        # 修复：确保慢客户端epoch范围合法（low < high）
        max_local_epochs = self.local_epochs
        if self.train_slow:
            high = max(self.local_epochs // 2 + 1, 2)  # 当local_epochs=1时，high=2
            max_local_epochs = np.random.randint(1, high)  # 范围[1, high)，确保至少1轮

        for epoch in range(max_local_epochs):
            for batch_idx, batch in enumerate(trainloader):
                mel, mel_len, text_ids, text_len = [x.to(self.device) for x in batch]

                # 慢客户端延迟模拟
                if self.train_slow:
                    time.sleep(0.1 * np.random.rand())

                # 模型前向传播（CTC需转置log_probs维度）
                log_probs = self.model(mel).permute(1, 0, 2)  # (T', B, vocab_size)
                input_lengths = mel_len // 8  # CNN两次下采样（stride=2）

                # 计算CTC损失
                loss = self.loss(log_probs, text_ids, input_lengths, text_len)
                print(f"客户端{self.id} 第{epoch}轮 第{batch_idx}批次 损失：{loss.item()}")

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.save_client_model()
        # 学习率衰减
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # 记录耗时
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time