import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data

class Server(object):
    def __init__(self, args, times):
        # 核心参数初始化
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.auto_break = args.auto_break
        self.top_cnt = args.top_cnt

        # 客户端相关
        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        # 模型聚合相关
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # ASR任务核心评估指标
        self.rs_test_wer = []
        self.rs_test_cer = []
        self.rs_train_loss = []


        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name

    def set_clients(self, clientObj):

        for i, train_slow, send_slow in zip(
                range(self.num_clients), self.train_slow_clients, self.send_slow_clients
        ):
            train_data = read_client_data(
                self.dataset, i, is_train=True,
                few_shot=self.args.few_shot,
                max_mel_len=self.args.max_mel_len,
                max_text_len=self.args.max_text_len,
                dict_path=self.args.dict_path
            )
            test_data = read_client_data(
                self.dataset, i, is_train=False,
                few_shot=self.args.few_shot,
                max_mel_len=self.args.max_mel_len,
                max_text_len=self.args.max_text_len,
                dict_path=self.args.dict_path
            )

            client = clientObj(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=train_slow,
                send_slow=send_slow
            )
            self.clients.append(client)

    def select_slow_clients(self, slow_rate):

        slow_clients = [False] * self.num_clients
        idx = np.random.choice(
            range(self.num_clients),
            int(slow_rate * self.num_clients),
            replace=False
        )
        for i in idx:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):

        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):

        specify_ids = [0,1,2,3,4,5,6,7,8,9]
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
            join_mode = "随机调整后"
        else:
            self.current_num_join_clients = self.num_join_clients
            join_mode = "固定"

        self.selected_clients = [
            client for client in self.clients
            if client.id in specify_ids
        ]

        selected_client_ids = [client.id for client in self.selected_clients]
        print("=" * 40)
        print(f"[本轮客户端选择结果]")
        print(f"总客户端数：{self.num_clients}")
        print(f"参与数量模式：{join_mode}")
        print(f"本轮参与客户端数：{self.current_num_join_clients}")
        print(f"参与客户端ID列表：{sorted(selected_client_ids)}")
        print("=" * 40)

        return self.selected_clients

    def send_models(self, round):

        key_params = []
        for name, param in self.global_model.named_parameters():
            if any(layer in name for layer in [
                'conv_block1.0.weight',
                'conv_block3.1.weight',
                'dense2.weight'
            ]):
                key_params.append((name, param))

        print("\n" + "=" * 50)
        print(f"【第{round}轮 发送全局模型参数（部分）】")
        for name, param in key_params:
            print(f"\n参数名称：{name}")
            print(f"参数形状：{param.shape}")
            print(f"部分数值：{param.detach().cpu().numpy().flatten()[:5]}...")
        print("=" * 50 + "\n")

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            print(f"已向客户端{client.id}发送模型，通信耗时：{time.time() - start_time:.4f}秒")

    def receive_models(self):

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = (client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] +
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds'])
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)

        self.uploaded_weights = [w / tot_samples for w in self.uploaded_weights]
        print(f"本轮活跃客户端ID：{[client.id for client in active_clients]}")

    def aggregate_parameters(self):
        """联邦平均"""
        if not self.uploaded_models:
            print("无上传的客户端模型，跳过聚合")
            return

        # -------------------------- 聚合前打印功能 --------------------------
        print("=" * 60)
        print(" 聚合前 - 客户端模型权重及关键参数")
        print("=" * 60)
        target_params = ["conv_block1.0.weight", "dense1.weight", "dense1.bias", "dense2.weight", "dense2.bias"]

        for idx, (w, client_model) in enumerate(zip(self.uploaded_weights, self.uploaded_models)):
            print(f"\n【客户端 {idx + 1}】")
            print(f"  对应权重 w：{w:.4f}")
            client_state = client_model.state_dict()
            for param_name in target_params:
                if param_name in client_state:
                    param = client_state[param_name]
                    # 基础信息（形状、类型、均值）
                    base_info = f"  {param_name}：形状={param.shape}，类型={param.dtype}，均值={param.data.mean().item():.4f}"
                    if param_name == "conv_block1.0.weight":
                        flat_param = param.detach().flatten()
                        sample_values = flat_param[:5] if len(flat_param) >= 5 else flat_param
                        base_info += f"，前5个元素：{[f'{v:.6f}' for v in sample_values]}"
                    print(base_info)

        # -------------------------- 核心聚合逻辑 --------------------------
        global_state = self.global_model.state_dict()

        for param_name in global_state.keys():
            for client_model in self.uploaded_models:
                if param_name not in client_model.state_dict():
                    raise KeyError(f"客户端模型缺失参数：{param_name}")

            aggregated_param = torch.zeros_like(global_state[param_name], device=self.device, dtype=torch.float32)

            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                client_param = client_model.state_dict()[param_name].to(self.device)
                client_param_float = client_param.to(dtype=torch.float32)
                aggregated_param += w * client_param_float.clone()

            global_state[param_name] = aggregated_param

        self.global_model.load_state_dict(global_state)

        # -------------------------- 聚合后打印功能 --------------------------
        print("\n" + "=" * 60)
        print("合后 - 全局模型关键参数（部分）")
        print("=" * 60)
        for param_name in target_params:
            if param_name in global_state:
                param = global_state[param_name]
                # 基础信息（形状、类型、均值）
                base_info = f"{param_name}：形状={param.shape}，类型={param.dtype}，均值={param.data.mean().item():.4f}"
                if param_name == "conv_block1.0.weight":
                    flat_param = param.detach().flatten()
                    sample_values = flat_param[:5] if len(flat_param) >= 5 else flat_param
                    base_info += f"，前5个元素：{[f'{v:.6f}' for v in sample_values]}"
                print(base_info)
        print("=" * 60)

    def add_parameters(self, weight, client_model):

        for server_param, client_param in zip(
            self.global_model.parameters(), client_model.parameters()
        ):
            server_param.data += client_param.data.clone() * weight

    def save_global_model(self):

        model_path = os.path.join("models", self.dataset)
        os.makedirs(model_path, exist_ok=True)
        model_path = os.path.join(model_path, f"{self.algorithm}_server.pt")
        torch.save(self.global_model.state_dict(), model_path)
        print(f"全局模型已保存至：{model_path}")

    def load_model(self):

        model_path = os.path.join("models", self.dataset, f"{self.algorithm}_server.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.global_model.load_state_dict(state_dict)
            print(f"已加载预训练全局模型：{model_path}")

    def save_results(self):

        result_path = "../results/"
        os.makedirs(result_path, exist_ok=True)
        file_name = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.times}"
        file_path = os.path.join(result_path, f"{file_name}.h5")

        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_wer', data=self.rs_test_wer)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def test_metrics(self):

        total_wer = 0.0
        total_samples = 0

        for client in self.clients:
            wer, samples = client.test_metrics()
            total_wer += wer * samples

            total_samples += samples

        avg_wer = total_wer / total_samples if total_samples > 0 else 0.0

        return avg_wer, total_samples

    def train_metrics(self):

        num_samples = []
        losses = []
        for client in self.clients:
            loss, samples = client.train_metrics()
            num_samples.append(samples)
            losses.append(loss * samples)

        total_loss = sum(losses) / sum(num_samples) if sum(num_samples) > 0 else 0.0
        return total_loss

    def evaluate(self):

        avg_wer,  _ = self.test_metrics()
        train_loss = self.train_metrics()

        self.rs_test_wer.append(avg_wer)

        self.rs_train_loss.append(train_loss)

        print(f"Averaged Train Loss: {train_loss:.4f}")
        print(f"Averaged Test WER: {avg_wer:.4f}")


    def check_done(self, acc_lss, top_cnt=None):

        for acc_ls in acc_lss:
            if len(acc_ls) < top_cnt:
                return False
            recent = acc_ls[-top_cnt:]
            if np.std(recent) < 1e-3:
                continue
            else:
                return False
        return True