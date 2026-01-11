import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from flcore.trainmodel.models import FedAvgASRModel
from utils.data_utils import read_client_data, load_vocab
from flcore.clients.clientbase import Client


def parse_args():
    """解析参数：包含模型、数据、运行相关配置"""
    parser = argparse.ArgumentParser(description="独立WER测试程序（提取自原有代码核心逻辑）")
    # -------------------------- 1. 模型相关 --------------------------
    parser.add_argument("-model_path", "--model_path", type=str,
                        default=r"D:",
                        help="最终聚合模型的权重路径（如FedAvg_server.pt）")
    parser.add_argument("-mel_dim", "--mel_dim", type=int, default=80,
                        help="梅尔频谱维度（训练时设为80，不可修改）")
    parser.add_argument("-hidden_dim", "--hidden_dim", type=int, default=256,
                        help="模型隐藏层维度（与训练时一致）")
    parser.add_argument("-vocab_size", "--vocab_size", type=int, default=941,
                        help="词汇表大小（含<blank>，从dict.txt读取的实际大小）")


    # -------------------------- 2. 数据相关 --------------------------
    parser.add_argument("-dataset", "--dataset", type=str, default="THCHS30_ASR",
                        help="数据集名称（需与data_utils中读取的数据集名匹配）")
    parser.add_argument("-dict_path", "--dict_path", type=str,
                        default=r"D:",
                        help="词汇表路径（与训练时使用的dict.txt一致）")
    parser.add_argument("-test_client_ids", "--test_client_ids", type=str, default="all",
                        help="测试的客户端ID（如'0,1,2'或'all'表示所有客户端）")
    parser.add_argument("-max_mel_len", "--max_mel_len", type=int, default=1600,
                        help="梅尔特征最大长度（与data_utils中process_asr_data配置一致）")
    parser.add_argument("-max_text_len", "--max_text_len", type=int, default=100,
                        help="文本最大长度（与data_utils中process_asr_data配置一致）")

    # -------------------------- 3. 运行相关 --------------------------
    parser.add_argument("-device", "--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="计算设备（优先GPU，无GPU自动切CPU）")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=10,
                        help="测试批次大小（根据显存调整，不影响结果）")
    return parser.parse_args()


def init_wer_tools(dict_path):
    """初始化WER计算必需的工具：词汇表映射、解码函数"""
    # 加载词汇表
    _, CHAR_TO_ID, ID_TO_CHAR = load_vocab(dict_path)
    print(f"✅ 加载词汇表成功：共{len(ID_TO_CHAR)}个符号（<blank> ID=0）")

    class WERToolkit:
        @staticmethod
        def decode_ctc(pred_ids: np.ndarray) -> str:
            """CTC预测结果解码：移除<blank>（ID=0）和连续重复标签"""
            pred_chars = []
            prev_id = -1
            for id in pred_ids:
                if id != CHAR_TO_ID["<blank>"] and id != prev_id:
                    pred_chars.append(ID_TO_CHAR.get(id, ""))
                    prev_id = id
            return " ".join(pred_chars)

        @staticmethod
        def decode_true_text(true_ids: np.ndarray, true_len: int) -> str:
            """真实标签解码：仅移除<blank>，取有效长度"""
            valid_true_ids = true_ids[:true_len]
            true_chars = [ID_TO_CHAR.get(id, "") for id in valid_true_ids if id != CHAR_TO_ID["<blank>"]]
            return " ".join(true_chars)

        @staticmethod
        def calculate_wer(pred_text: str, true_text: str) -> float:
            """计算WER：基于Levenshtein编辑距离"""
            # 分割为词（拼音）列表
            pred_words = pred_text.strip().split()
            true_words = true_text.strip().split()
            # 计算编辑距离
            print(f"预测词列表: {pred_words}")
            print(f"真实词列表: {true_words}")
            edit_dist = Client.edit_distance(pred_words, true_words)
            # 分母取最大长度
            denom = max(len(pred_words), len(true_words), 1)
            return edit_dist / denom

    return WERToolkit, ID_TO_CHAR


def load_final_model(args):
    """加载最终生成的聚合模型"""
    try:
        # 初始化模型
        model = FedAvgASRModel(
            in_channels=1,
            mel_dim=args.mel_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=args.vocab_size
        ).to(args.device)

        # 加载模型权重
        if torch.cuda.is_available() and args.device == "cuda":
            state_dict = torch.load(args.model_path)
        else:
            state_dict = torch.load(args.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()  # 切换到评估模式
        print(f"✅ 加载最终模型成功：{args.model_path}（运行设备：{args.device}）")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        print(f"请检查：1. 模型路径是否正确；2. 模型参数（mel_dim/hidden_dim/vocab_size）是否与训练一致")
        return None


def test_model_wer(args, model, wer_toolkit):
    # 1. 确定要测试的客户端ID列表
    if args.test_client_ids == "all":
        test_client_ids = list(range(10))  # 默认总客户端数为10
    else:
        test_client_ids = [int(cid.strip()) for cid in args.test_client_ids.split(",")]
    print(f"\n📌 测试客户端列表：{test_client_ids}（共{len(test_client_ids)}个客户端）")

    # 2. 遍历客户端测试集，计算总WER
    total_wer = 0.0
    total_samples = 0
    client_wer_detail = []  # 记录每个客户端的WER详情

    with torch.no_grad():  # 禁用梯度计算，加速推理并减少内存占用
        for client_id in test_client_ids:
            # 读取当前客户端的测试数据
            try:
                test_data = read_client_data(
                    dataset=args.dataset,
                    idx=client_id,
                    is_train=False,  # 读取测试集
                    max_mel_len=args.max_mel_len,
                    max_text_len=args.max_text_len,
                    dict_path=args.dict_path
                )
                if len(test_data) == 0:
                    print(f"⚠️  客户端{client_id}：无测试数据，跳过")
                    continue
            except Exception as e:
                print(f"⚠️  客户端{client_id}：数据读取失败（{str(e)}），跳过")
                continue

            # 构建测试数据加载器
            test_loader = DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False, drop_last=False
            )

            # 计算当前客户端的WER
            client_wer = 0.0
            client_sample_cnt = 0

            for batch in test_loader:
                # 解包数据
                mel, mel_len, text_ids, text_len = [x.to(args.device) for x in batch]
                try:
                    conv_weight = model.conv_block1[0].weight
                    conv_weight2=model.dense1.weight
                    conv_weight3 = model.dense1.bias
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
                # 模型推理（输出log_probs：(batch, T', vocab_size)）
                model.eval()
                print(f"mel：{mel}")
                log_probs = model(mel)
                print(f"ids：{log_probs}")
                # 贪心解码（取每个时间步概率最大的字符ID）
                pred_ids = torch.argmax(log_probs, dim=-1).cpu().numpy()

                # 解码为文本（拼音序列）
                true_ids_np = text_ids.cpu().numpy()  # 真实标签ID
                true_len_np = text_len.cpu().numpy()  # 真实文本有效长度
                pred_texts = [wer_toolkit.decode_ctc(pred) for pred in pred_ids]
                true_texts = [wer_toolkit.decode_true_text(true, tl)
                              for true, tl in zip(true_ids_np, true_len_np)]

                # 累加WER和样本数
                for pred_txt, true_txt in zip(pred_texts, true_texts):
                    client_wer += wer_toolkit.calculate_wer(pred_txt, true_txt)
                    client_sample_cnt += 1

            # 计算当前客户端平均WER
            avg_client_wer = client_wer / client_sample_cnt if client_sample_cnt > 0 else 0.0
            client_wer_detail.append({
                "client_id": client_id,
                "sample_cnt": client_sample_cnt,
                "avg_wer": avg_client_wer
            })
            total_wer += client_wer
            total_samples += client_sample_cnt

            print(f"✅ 客户端{client_id}：测试样本数={client_sample_cnt:3d}，平均WER={avg_client_wer:.4f}")

    # 3. 计算整体平均WER并输出汇总
    if total_samples == 0:
        print("\n❌ 无有效测试样本，无法计算整体WER")
        return

    overall_avg_wer = total_wer / total_samples
    print("\n" + "=" * 80)
    print(f"📊 最终模型WER测试结果汇总")
    print(f"=" * 80)
    print(f"测试客户端数量：{len(test_client_ids)}")
    print(f"总测试样本数量：{total_samples}")
    print(f"所有客户端整体平均WER：{overall_avg_wer:.4f}")
    print(f"\n各客户端详细WER：")
    for detail in client_wer_detail:
        print(f"  客户端{detail['client_id']}：样本数={detail['sample_cnt']:3d}，WER={detail['avg_wer']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    # 1. 解析参数
    args = parse_args()

    # 2. 适配设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  未检测到可用CUDA设备，自动切换到CPU运行")
        args.device = "cpu"

    # 3. 初始化WER计算工具
    wer_toolkit, _ = init_wer_tools(args.dict_path)

    # 4. 加载最终模型
    final_model = load_final_model(args)
    if not final_model:
        exit(1)  # 模型加载失败则退出

    # 5. 执行WER测试
    test_model_wer(args, final_model, wer_toolkit)