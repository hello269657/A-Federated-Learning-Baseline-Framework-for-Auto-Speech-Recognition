import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 初始化慢客户端
        self.set_slow_clients()
        # 初始化客户端
        self.set_clients(clientAVG)

        print(f"\n参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")
        print("服务器和客户端初始化完成。")

        self.Budget = []  # 记录每轮耗时

    def train(self):
        """执行联邦平均训练流程"""
        for round in range(self.global_rounds + 1):
            start_time = time.time()

            # 1. 选择本轮参与客户端
            self.selected_clients = self.select_clients()
            # 2. 向客户端发送全局模型
            self.send_models(round)

            # 3. 每eval_gap轮评估一次全局模型
            if round % self.eval_gap == 0:
                print(f"\n-------------第 {round} 轮-------------")
                print("\n评估全局模型...")
                self.evaluate()  # 评估WER
            if round != self.global_rounds:
            # 4. 客户端本地训练（使用CTC损失）
             for client in self.selected_clients:
                print(f"客户端{client.id} ")
                client.train()
                # 5. 接收客户端模型并聚合
             self.receive_models()
             self.aggregate_parameters()
             # 记录本轮耗时
             self.Budget.append(time.time() - start_time)
             print(f'本轮耗时: {self.Budget[-1]:.2f}s')
            # 6. 检查是否早停
             if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_wer], top_cnt=self.top_cnt
             ):
                print(f"\n在第 {round} 轮满足早停条件，停止训练。")
                break

        # 训练结束后输出关键结果
        print(f"\n最佳测试WER: {min(self.rs_test_wer):.4f}")

        # 保存结果和全局模型
        self.save_results()
        self.save_global_model()