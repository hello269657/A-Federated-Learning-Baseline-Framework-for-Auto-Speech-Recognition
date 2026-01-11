import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging
from flcore.servers.serveravg import FedAvg
from flcore.trainmodel.models import FedAvgASRModel
from utils.data_utils import VOCAB_SIZE
from utils.result_utils import average_data

import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = Logger('a.log', sys.stdout)  # 正常内容写入a.log
sys.stderr = Logger('error.log', sys.stderr)  # 错误内容写入error.log
# 初始化日志和随机种子
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    time_list = []
    for i in range(args.prev, args.times):
        print(f"\n============= 第 {i} 次实验 =============")
        start = time.time()

        # 初始化ASR模型
        args.model = FedAvgASRModel(
            in_channels=1,
            mel_dim=args.mel_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=VOCAB_SIZE,
        ).to(args.device)
        print(f"模型初始化完成：{args.model.__class__.__name__}（设备：{args.device}）")

        # 初始化联邦算法
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        else:
            raise NotImplementedError(f"未支持算法：{args.algorithm}")

        # 开始训练
        server.train()
        time_list.append(time.time() - start)

    # 输出实验结果
    print(f"\n平均单次实验耗时：{round(np.average(time_list), 2)}s")
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
    print("所有实验完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1. ASR核心参数
    parser.add_argument("-mel_dim", "--mel_dim", type=int, default=80, help="梅尔频谱维度")
    parser.add_argument("-hidden_dim", "--hidden_dim", type=int, default=256, help="LSTM隐藏层维度")
    parser.add_argument("-num_lstm", "--num_lstm_layers", type=int, default=3, help="LSTM层数")
    parser.add_argument("-max_mel_len", "--max_mel_len", type=int, default=1600, help="梅尔最大长度")
    parser.add_argument("-max_text_len", "--max_text_len", type=int, default=100, help="文本最大长度")
    parser.add_argument("-dict_path", "--dict_path", type=str, default=r"D:", help="词汇表dict.txt路径")  

    # 2. 联邦学习参数
    parser.add_argument("-data", "--dataset", type=str, default="THCHS30_ASR", help="数据集名称")
    parser.add_argument("-m", "--model", type=str, default="ASRModel", help="模型标识")
    parser.add_argument("-lr", "--local_learning_rate", type=float, default=0.003, help="客户端学习率")
    parser.add_argument("-go", "--goal", type=str, default="test", help="实验目标")
    parser.add_argument("-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="计算设备")
    parser.add_argument("-did", "--device_id", type=str, default="0", help="GPU设备ID（如0,1）")
    parser.add_argument("-lbs", "--batch_size", type=int, default=10, help="客户端批次大小")
    parser.add_argument("-ld", "--learning_rate_decay", type=bool, default=False, help="是否启用学习率衰减")
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.99, help="衰减系数")
    parser.add_argument("-gr", "--global_rounds", type=int, default=3, help="全局训练轮次")
    parser.add_argument("-tc", "--top_cnt", type=int, default=100, help="早停判断轮次（auto_break启用时）")
    parser.add_argument("-ls", "--local_epochs", type=int, default=10, help="客户端每轮训练epoch数")
    parser.add_argument("-algo", "--algorithm", type=str, default="FedAvg", help="联邦算法（仅支持FedAvg）")
    parser.add_argument("-jr", "--join_ratio", type=float, default=1, help="每轮参与客户端比例")
    parser.add_argument("-rjr", "--random_join_ratio", type=bool, default=False, help="是否随机参与比例")
    parser.add_argument("-nc", "--num_clients", type=int, default=10, help="总客户端数量")
    parser.add_argument("-pv", "--prev", type=int, default=0, help="起始实验序号")
    parser.add_argument("-t", "--times", type=int, default=1, help="实验重复次数")
    parser.add_argument("-eg", "--eval_gap", type=int, default=1, help="评估间隔轮次")
    parser.add_argument("-sfn", "--save_folder_name", type=str, default="items", help="结果保存文件夹")
    parser.add_argument("-ab", "--auto_break", type=bool, default=False, help="是否启用早停")
    parser.add_argument("-cdr", "--client_drop_rate", type=float, default=0.0, help="客户端掉线率")
    parser.add_argument("-tsr", "--train_slow_rate", type=float, default=0.0, help="慢客户端比例（训练延迟）")
    parser.add_argument("-ssr", "--send_slow_rate", type=float, default=0.0, help="慢客户端比例（通信延迟）")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    args = parser.parse_args()

    # 配置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n警告：CUDA不可用，自动切换到CPU")
        args.device = "cpu"

    # 打印参数配置
    print("=" * 50)
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    print("=" * 50)

    # 启动实验
    run(args)
