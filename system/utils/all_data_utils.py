import numpy as np
import os
import torch
from typing import List, Tuple

VOCAB, CHAR_TO_ID, ID_TO_CHAR, VOCAB_SIZE = [], {}, {}, 941
# ========================== 核心修改：从dict.txt加载词汇表 ==========================
def load_vocab(dict_path):
    """从指定路径加载词汇表，返回VOCAB、CHAR_TO_ID、ID_TO_CHAR"""
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if lines[0].split("\t")[0] != "<blank>" or int(lines[0].split("\t")[1]) != 0:
        raise ValueError("dict.txt首行必须为<blank>\t0（CTC要求）")

    vocab = [line.split("\t")[0] for line in lines]
    char_to_id = {char: int(idx) for char, idx in [line.split("\t") for line in lines]}
    id_to_char = {int(idx): char for char, idx in [line.split("\t") for line in lines]}
    return vocab, char_to_id, id_to_char


# 加载词汇表
DICT_PATH = r"D:"
VOCAB, CHAR_TO_ID, ID_TO_CHAR = load_vocab(DICT_PATH)
VOCAB_SIZE = len(VOCAB)
print(f"成功加载词汇表：共{VOCAB_SIZE}个符号（含空白标签）")

# ========================== 解码函数（CTC解码+真实标签解码）==========================
def decode_ctc(pred_ids: np.ndarray, id_to_char: dict) -> str:
    """CTC解码：移除空白标签（ID=0）和连续重复标签，转为可读文本"""
    pred_chars = []
    prev_id = -1  # 记录前一个标签ID，避免连续重复
    for id in pred_ids:
        # 跳过空白标签（0）和与前一个相同的标签
        if id != 0 and id != prev_id:
            pred_chars.append(id_to_char.get(id, ''))
            prev_id = id
    return ' '.join(pred_chars)

def decode_true_text(true_ids: np.ndarray, id_to_char: dict) -> str:
    # 过滤空白标签，仅保留有效字符ID
    true_chars = [id_to_char.get(id, '') for id in true_ids if id != 0]
    return ' '.join(true_chars)

# ========================== 数据读取与处理逻辑==========================
def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('..', 'dataset', dataset, 'train')
    else:
        data_dir = os.path.join('..', 'dataset', dataset, 'test')
    # 拼接文件名（如"0.npz"）
    file = os.path.join(data_dir, f"{idx}.npz")
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_global_data(dataset_dir, is_train=True, max_mel_len=1600, max_text_len=100):
    """读取全局训练/测试集（不分客户端）"""
    data_type = "train" if is_train else "test"
    data_dir = os.path.join(dataset_dir, data_type)

    all_samples = []
    # 遍历所有客户端的npz文件，合并样本
    for npz_file in os.listdir(data_dir):
        if npz_file.endswith(".npz"):
            file_path = os.path.join(data_dir, npz_file)
            with open(file_path, 'rb') as f:
                data = np.load(f, allow_pickle=True)['data'].tolist()
            # 处理每个样本
            samples = process_asr_data(data, max_mel_len, max_text_len)
            all_samples.extend(samples)

    print(f"加载{data_type}集完成：共{len(all_samples)}个样本")
    return all_samples


def process_asr_data(data: dict, max_mel_len: int, max_text_len: int) -> List[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    mels = data["feature"]
    label_ids_list = data["label_ids"]
    processed = []

    for mel, label_ids in zip(mels, label_ids_list):
        # 1. 处理梅尔频谱（截断/补零）
        mel_len = mel.shape[0]
        if mel_len > max_mel_len:
            mel = mel[:max_mel_len, :]
            mel_len = max_mel_len
        else:
            pad = np.zeros((max_mel_len - mel_len, mel.shape[1]), dtype=np.float32)
            mel = np.concatenate([mel, pad], axis=0)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, T, 80)
        mel_len_tensor = torch.tensor(mel_len, dtype=torch.int32)

        # 2. 处理label_ids
        text_len = np.count_nonzero(label_ids != CHAR_TO_ID["<blank>"])
        if text_len > max_text_len:
            label_ids = label_ids[:max_text_len]
            text_len = max_text_len
        else:
            label_ids = np.pad(
                label_ids, (0, max_text_len - len(label_ids)),
                mode="constant", constant_values=CHAR_TO_ID["<blank>"]
            )
        text_ids_tensor = torch.tensor(label_ids, dtype=torch.int32)
        text_len_tensor = torch.tensor(text_len, dtype=torch.int32)

        processed.append((mel_tensor, mel_len_tensor, text_ids_tensor, text_len_tensor))
    return processed