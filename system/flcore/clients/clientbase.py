import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data, ID_TO_CHAR


class Client(object):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.id = id
        self.dataset = args.dataset
        self.device = args.device
        self.algorithm = args.algorithm
        self.save_folder_name = args.save_folder_name

        # 数据相关参数
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot
        self.max_mel_len = args.max_mel_len
        self.max_text_len = args.max_text_len

        # 1. 初始化ASR模型
        self.model = copy.deepcopy(args.model).to(self.device)

        # 2. 初始化CTC损失
        self.loss = nn.CTCLoss(
            blank=0,
            reduction='mean',
            zero_infinity=True
        )

        # 3. 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.local_learning_rate
        )

        # 4. 学习率调度器
        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )

        # 客户端异构性相关
        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # 冗余字段
        self.has_BatchNorm = any(isinstance(layer, nn.BatchNorm2d) for layer in self.model.children())

    def load_train_data(self, batch_size=None):
        """加载客户端训练数据，补充传递梅尔/文本长度参数"""
        batch_size = batch_size or self.batch_size
        train_data = read_client_data(
            dataset=self.dataset,
            idx=self.id,
            is_train=True,
            few_shot=self.few_shot,
            max_mel_len=self.max_mel_len,
            max_text_len=self.max_text_len
        )
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        """加载客户端测试数据，补充传递梅尔/文本长度参数"""
        batch_size = batch_size or self.batch_size
        test_data = read_client_data(
            dataset=self.dataset,
            idx=self.id,
            is_train=False,
            few_shot=self.few_shot,
            max_mel_len=self.max_mel_len,
            max_text_len=self.max_text_len
        )
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model):
        """接收服务器发送的全局模型参数"""
        for server_param, client_param in zip(model.parameters(), self.model.parameters()):
            client_param.data = server_param.data.clone()
        for server_buf, client_buf in zip(model.buffers(), self.model.buffers()):
            client_buf.data = server_buf.data.clone()

    def test_metrics(self):
        """计算测试集WER"""
        self.model.eval()
        testloader = self.load_test_data()
        total_wer = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in testloader:
                # 解包批量数据
                mel, mel_len, text_ids, text_len = [x.to(self.device) for x in batch]
                try:
                    conv_weight = self.model.conv_block1[0].weight
                    conv_weight2=self.model.dense1.weight
                    conv_weight3 = self.model.dense1.bias
                    conv_weight_np = conv_weight.detach().cpu().numpy()
                    # 2. 转移到CPU并转为numpy
                    conv_weight_np2 = conv_weight2.detach().cpu().numpy()
                    conv_weight_np3 = conv_weight3.detach().cpu().numpy()
                    # 3. 展平参数并取前5个元素
                    first_five = conv_weight_np.flatten()[:5]
                    first_five2 = conv_weight_np2.flatten()[:5]
                    first_five3 = conv_weight_np3.flatten()[:5]
                    print(f"conv_block1.0.weight前五个元素：{first_five}")
                    print(f"dense1.weight前五个元素：{first_five2}")
                    print(f"dense1.bias前五个元素：{first_five3}")
                except AttributeError:
                    print("警告：未找到参数conv_block1.0.weight，请检查模型结构命名是否正确")
                # 模型推理：输出log_probs (B, T', vocab_size)
                self.model.eval()
                print(f"客户端{self.id}测试时模型模式：{'评估模式' if not self.model.training else '训练模式'}")  # 验证模式
                print(f"mel：{mel}")
                log_probs = self.model(mel)
                print(f"ids：{log_probs}")
                # 1. 贪心解码（CTC后处理：去空白、去重复）
                pred_ids = torch.argmax(log_probs, dim=-1).cpu().numpy()  # (B, T')

                true_ids = text_ids.cpu().numpy()  # (B, max_text_len)
                true_lens = text_len.cpu().numpy()  # (B,)

                # 2. 解码为文本（拼音序列）
                pred_texts = [self.decode_ctc(pred) for pred in pred_ids]
                true_texts = [self.decode_true_text(true,tl) for true, tl in zip(true_ids, true_lens)]

                # 3. 计算单样本WER并累加
                for pred, true in zip(pred_texts, true_texts):
                    total_wer += self.calculate_wer(pred, true)

                    total_samples += 1


        avg_wer = total_wer / max(total_samples, 1)
        return avg_wer, total_samples

    def decode_ctc(self, pred_ids: np.ndarray) -> str:
        """CTC解码：移除空白标签（0）和连续重复标签"""
        pred_chars = []
        prev_id = -1  # 记录前一个标签ID，避免重复
        for id in pred_ids:
            if id != 0 and id != prev_id:
                pred_chars.append(ID_TO_CHAR.get(id, ''))
                prev_id = id
        return ' '.join(pred_chars)  # 拼音用空格分隔

    def decode_true_text(self, true_ids: np.ndarray, true_len: int) -> str:
        """真实文本解码：移除补零的blank标签"""
        true_ids = true_ids[:true_len]
        true_chars = [ID_TO_CHAR.get(id, '') for id in true_ids if id !=0]
        return ' '.join(true_chars)  # 拼音用空格分隔

    def calculate_wer(self, pred: str, true: str) -> float:
        """计算词错误率（WER），处理空文本情况"""
        pred_words = pred.strip().split()
        true_words = true.strip().split()
        print(f"预测词列表: {pred_words}")
        print(f"真实词列表: {true_words}")
        # 编辑距离（Levenshtein）
        edit_dist = self.edit_distance(pred_words, true_words)
        # 分母取最大长度（避免除以零）
        denom = max(len(pred_words), len(true_words), 1)
        return edit_dist / denom

    @staticmethod
    def edit_distance(pred: list, true: list) -> int:
        """静态方法：计算Levenshtein编辑距离（插入、删除、替换）"""
        m, n = len(pred), len(true)
        # 初始化DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == true[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],    # 删除
                                      dp[i][j-1],    # 插入
                                      dp[i-1][j-1])  # 替换
        return dp[m][n]

    def train_metrics(self):
        """计算训练损失"""
        self.model.eval()
        trainloader = self.load_train_data(batch_size=1)  # 单样本计算避免批量误差
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in trainloader:
                mel, mel_len, text_ids, text_len = [x.to(self.device) for x in batch]
                log_probs = self.model(mel).permute(1, 0, 2)  # (T', B, vocab_size)
                input_lengths = mel_len // 8  # 与CNN降采样倍数一致

                # 计算损失
                loss = self.loss(log_probs, text_ids, input_lengths, text_len)
                total_loss += loss.item() * len(mel)  # 按样本数加权
                total_samples += len(mel)

        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss, total_samples


    def save_client_model(self):
       """保存客户端训练后的模型参数、优化器状态、训练轮次"""
       save_dict = {
           "model_state_dict": self.model.state_dict(),  # 模型权重
           "optimizer_state_dict": self.optimizer.state_dict(),  # 优化器状态

           "loss": self.train_metrics()[0],  # 最新训练损失
           "args": self.args  # 训练参数
       }
       model_path = os.path.join("client model", self.dataset)
       os.makedirs(model_path, exist_ok=True)
       model_path = os.path.join(model_path, f"client{self.id}.pt")

       torch.save(save_dict, model_path)
       print(f"【客户端{self.id}】已保存模型至：{model_path}")