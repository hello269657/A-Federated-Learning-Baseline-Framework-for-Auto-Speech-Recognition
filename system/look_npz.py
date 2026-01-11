import numpy as np

# 设置NumPy打印选项
np.set_printoptions(
    threshold=np.inf,  # 打印所有元素
    suppress=True,
    precision=6  # 保留6位小数
)

# npz文件路径
npz_path = r"D:"

with np.load(npz_path, allow_pickle=True) as npz_data:
    # 1. 查看 npz 文件里的所有顶层键
    all_keys = npz_data.files
    print("1. npz 文件顶层所有键：", all_keys, "\n")

    # 2. 提取 'data' 键对应的内容
    data_obj = npz_data['data']
    data_dict = data_obj.item()  # 转成字典，键为 'mel'、'transcript'、'label_ids'
    print("2. 'data' 内部字典包含的键：", list(data_dict.keys()), "\n")

    # 3. 查看 'mel' 列表
    mel_list = data_dict['feature']
    print("3. 查看 'mel' 列表（每个元素是一个音频的梅尔频谱，NumPy 数组）：")
    print(f"   - 列表总长度（音频数量）：{len(mel_list)}")

    # 选择第一个样本，输出完整梅尔频谱数值
    target_sample_idx = 3  # 样本索引
    target_mel = mel_list[target_sample_idx]
    print(f"\n4. 第{target_sample_idx + 1}个样本的完整梅尔频谱数值（形状：{target_mel.shape}）：")
    print("=" * 80)
    print(target_mel)
    print("=" * 80)
    print(f"   - 样本梅尔频谱：{target_mel.shape[0]}行（时序帧）×{target_mel.shape[1]}列（梅尔维度）")
    print(f"   - 数值范围：{target_mel.min():.6f} ~ {target_mel.max():.6f}")
    print(f"   - 数据类型：{target_mel.dtype}\n")

    # 5. 查看该样本对应的文本和label_ids
    transcript_list = data_dict['transcript']
    label_ids_list = data_dict['label_ids']
    print(f"5. 第{target_sample_idx + 1}个样本的配套信息：")
    print(f"   - 原始文本转录：{transcript_list[target_sample_idx]}")
    print(
        f"   - label_ids（前20个，完整长度{len(label_ids_list[target_sample_idx])}）：{label_ids_list[target_sample_idx][:20]}...\n")

    # 6. 验证数据一致性（确保梅尔列表、文本列表、label_ids列表长度相同）
    if len(mel_list) == len(transcript_list) == len(label_ids_list):
        print("6. 数据一致性验证：✅ 所有列表长度一致")
        print(f"   - 总有效音频-文本对数量：{len(mel_list)}")
    else:
        print("6. 数据一致性验证：❌ 长度不一致！请检查数据生成过程")