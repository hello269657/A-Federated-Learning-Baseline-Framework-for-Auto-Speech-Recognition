import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import numpy as np
from utils. all_data_utils import load_vocab, read_global_data, ID_TO_CHAR
from flcore.trainmodel.models import FedAvgASRModel

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
sys.stdout = Logger('a.log', sys.stdout)
sys.stderr = Logger('error.log', sys.stderr)
# ASR工具类
class ASREvaluator:
    @staticmethod
    def decode_ctc(pred_ids: np.ndarray, id_to_char) -> str:
        """CTC解码：去空白、去重复"""
        pred_chars = []
        prev_id = -1
        for id in pred_ids:
            if id != 0 and id != prev_id:
                pred_chars.append(id_to_char.get(id, ''))
                prev_id = id
        return ' '.join(pred_chars)

    @staticmethod
    def decode_true_text(true_ids: np.ndarray, id_to_char) -> str:
        """真实文本解码：去空白"""
        true_chars = [id_to_char.get(id, '') for id in true_ids if id != 0]
        return ' '.join(true_chars)

    @staticmethod
    def edit_distance(pred: list, true: list) -> int:
        """Levenshtein编辑距离"""
        m, n = len(pred), len(true)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == true[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    @staticmethod
    def calculate_wer(pred: str, true: str) -> float:
        """计算词错误率（WER）"""
        pred_words = pred.strip().split()
        true_words = true.strip().split()
        edit_dist = ASREvaluator.edit_distance(pred_words, true_words)
        return edit_dist / max(len(pred_words), len(true_words), 1)

    @staticmethod
    def calculate_cer(pred: str, true: str) -> float:
        """计算字符错误率（CER）"""
        pred_chars = pred.replace(' ', '')
        true_chars = true.replace(' ', '')
        edit_dist = ASREvaluator.edit_distance(list(pred_chars), list(true_chars))
        return edit_dist / max(len(pred_chars), len(true_chars), 1)


def evaluate_model(model, test_loader, device, id_to_char):
    """评估ASR模型：计算测试集平均WER/CER"""
    model.eval()
    total_wer = 0.0
    total_cer = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            mel, mel_len, text_ids, text_len = [x.to(device) for x in batch]
            # 模型推理（输出log_probs: [batch, seq_len//8, vocab_size]）
            log_probs = model(mel)
            # 贪心解码（取概率最大的ID）
            pred_ids = torch.argmax(log_probs, dim=-1).cpu().numpy()
            true_ids = text_ids.cpu().numpy()
            true_lens = text_len.cpu().numpy()

            # 解码为文本并计算WER/CER
            for pred_id, true_id, tl in zip(pred_ids, true_ids, true_lens):
                pred_text = ASREvaluator.decode_ctc(pred_id, id_to_char)
                true_text = ASREvaluator.decode_true_text(true_id[:tl], id_to_char)
                total_wer += ASREvaluator.calculate_wer(pred_text, true_text)
                total_cer += ASREvaluator.calculate_cer(pred_text, true_text)
                total_samples += 1

    avg_wer = total_wer / total_samples if total_samples > 0 else 0.0
    avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
    return avg_wer, avg_cer


def train_asr(args):
    # 1. 初始化配置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 2. 加载词汇表
    vocab, char_to_id, id_to_char = load_vocab(args.dict_path)
    vocab_size = len(vocab)

    # 3. 加载全局训练/测试集
    train_samples = read_global_data(
        dataset_dir=args.dataset_dir,
        is_train=True,
        max_mel_len=args.max_mel_len,
        max_text_len=args.max_text_len
    )
    test_samples = read_global_data(
        dataset_dir=args.dataset_dir,
        is_train=False,
        max_mel_len=args.max_mel_len,
        max_text_len=args.max_text_len
    )

    # 4. 构建DataLoader
    train_loader = DataLoader(
        train_samples, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_samples, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # 5. 初始化ASR模型、CTC损失、优化器
    model = FedAvgASRModel(
        in_channels=1,
        mel_dim=args.mel_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size
    ).to(device)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)  # CTC损失（blank=0）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_decay_gamma
    ) if args.lr_decay else None

    # 6. 训练循环
    best_wer = float('inf')
    train_losses = []
    test_wers = []
    test_cers = []

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        # 训练批次
        for batch_idx, batch in enumerate(train_loader):
            mel, mel_len, text_ids, text_len = [x.to(device) for x in batch]
            # 模型前向传播（CTC损失要求log_probs维度：[seq_len, batch, vocab_size]）
            log_probs = model(mel).permute(1, 0, 2)  # 转置为[seq_len//8, batch, vocab_size]
            input_lengths = mel_len // 8  # 匹配CNN下采样倍数（8倍）

            # 计算损失
            loss = criterion(log_probs, text_ids, input_lengths, text_len)
            total_loss += loss.item() * mel.size(0)  # 按批次大小加权

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印批次信息
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # 计算epoch平均损失
        avg_train_loss = total_loss / len(train_samples)
        train_losses.append(avg_train_loss)

        # 评估测试集WER/CER
        avg_wer, avg_cer = evaluate_model(model, test_loader, device, id_to_char)
        test_wers.append(avg_wer)
        test_cers.append(avg_cer)

        # 学习率衰减
        if args.lr_decay:
            scheduler.step()

        # 保存最佳模型（按WER最低）
        if avg_wer < best_wer:
            best_wer = avg_wer
            torch.save(model.state_dict(), args.save_model_path)
            print(f"Epoch [{epoch + 1}] | 保存最佳模型（WER: {best_wer:.4f}）")

        # 打印epoch总结
        epoch_time = time.time() - start_time
        print(f"=" * 50)
        print(f"Epoch [{epoch + 1}/{args.epochs}] | 耗时: {epoch_time:.2f}s")
        print(f"平均训练损失: {avg_train_loss:.4f}")
        print(f"测试集WER: {avg_wer:.4f} | 测试集CER: {avg_cer:.4f}")
        print(f"当前最佳WER: {best_wer:.4f}")
        print("=" * 50 + "\n")

    # 7. 保存训练结果（损失、WER、CER）
    np.savez(args.save_result_path,
             train_losses=train_losses,
             test_wers=test_wers,
             test_cers=test_cers)
    print(f"训练完成！结果保存至：{args.save_result_path}")
    print(f"最佳模型保存至：{args.save_model_path}")
    print(f"最终测试WER: {test_wers[-1]:.4f} | 最终测试CER: {test_cers[-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立ASR训练脚本（剔除联邦学习）")
    # ASR数据配置
    parser.add_argument("--dataset_dir", type=str, default=r"D:", help="数据集根目录（含train/test子目录）")
    parser.add_argument("-dict_path", "--dict_path", type=str, default=r"D:",
                        help="词汇表dict.txt路径")
    parser.add_argument("--max_mel_len", type=int, default=1600, help="梅尔频谱最大长度")
    parser.add_argument("--max_text_len", type=int, default=100, help="文本标签最大长度")
    parser.add_argument("--mel_dim", type=int, default=80, help="梅尔频谱特征维度（或MFCC维度）")

    # 模型与训练配置
    parser.add_argument("--hidden_dim", type=int, default=256, help="ASR模型全连接层隐藏维度")
    parser.add_argument("--batch_size", type=int, default=16, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练总epoch数")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="学习率")
    parser.add_argument("--lr_decay", type=bool, default=False, help="是否启用学习率衰减")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.99, help="学习率衰减系数")

    # 设备与保存配置
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="计算设备")
    parser.add_argument("--save_model_path", type=str, default="./best_asr_model.pth", help="最佳模型保存路径")
    parser.add_argument("--save_result_path", type=str, default="./asr_train_results.npz", help="训练结果保存路径")

    args = parser.parse_args()
    train_asr(args)