import numpy as np


def levenshtein_distance_with_align(s1, s2):
    """计算编辑距离并返回对齐结果"""
    m = len(s1)
    n = len(s2)
    # 创建DP表
    dp = np.zeros((m + 1, n + 1), dtype=int)
    # 初始化边界
    for i in range(m + 1):
        dp[i][0] = i  # s2为空，删除s1的i个元素
    for j in range(n + 1):
        dp[0][j] = j  # s1为空，插入s2的j个元素
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i][j - 1] + 1,  # 插入
                    dp[i - 1][j] + 1,  # 删除
                    dp[i - 1][j - 1] + 1  # 替换
                )
    # 获取对齐结果
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            # 匹配操作
            alignment.append((s1[i - 1], s2[j - 1], "匹配"))
            i -= 1
            j -= 1
        else:
            current = dp[i][j]
            # 判断操作类型
            if i > 0 and j > 0 and current == dp[i - 1][j - 1] + 1:
                # 替换操作
                alignment.append((s1[i - 1], s2[j - 1], "替换"))
                i -= 1
                j -= 1
            elif j > 0 and current == dp[i][j - 1] + 1:
                # 插入操作（s1插入s2的元素）
                alignment.append((None, s2[j - 1], "插入"))
                j -= 1
            elif i > 0 and current == dp[i - 1][j] + 1:
                # 删除操作（s1删除元素）
                alignment.append((s1[i - 1], None, "删除"))
                i -= 1

    # 反转对齐结果
    alignment = alignment[::-1]
    return dp[m][n], alignment


def calculate_wer(true_text, pred_text):
    """计算词错误率（WER）"""
    true_words = true_text.split()
    pred_words = pred_text.split()
    edit_dist, alignment = levenshtein_distance_with_align(pred_words, true_words)  # 注意顺序：pred→true
    true_len = max(len(true_words), 1)
    wer = edit_dist / true_len
    return wer, edit_dist, alignment


def calculate_cer(true_text, pred_text):
    """计算字符错误率（CER）"""
    true_chars = list(true_text.replace(' ', ''))  # 去空格后转字符列表
    pred_chars = list(pred_text.replace(' ', ''))
    edit_dist, alignment = levenshtein_distance_with_align(pred_chars, true_chars)
    true_len = max(len(true_chars), 1)
    cer = edit_dist / true_len
    return cer, edit_dist, alignment


if __name__ == "__main__":
    # 输入文本
    true_text = ""
    pred_text = ""

    # 计算WER及词对齐
    wer, wer_edit_dist, word_alignment = calculate_wer(true_text, pred_text)
    # 计算CER及字符对齐
    cer, cer_edit_dist, char_alignment = calculate_cer(true_text, pred_text)

    # 输出结果
    print("=" * 80)
    print("真实标签文本:", true_text)
    print("\n模型预测文本:", pred_text)
    print("=" * 80)

    # 词级结果
    print("\n【词级评估】")
    print(f"词编辑距离: {wer_edit_dist}")
    print(f"词错误率（WER）: {wer:.4f} ({wer * 100:.2f}%)")
    print("词对齐结果（预测词 → 真实词）:")
    for i, (pred, true, op) in enumerate(word_alignment, 1):
        print(f"  步骤{i}: {pred!r:8} → {true!r:8} （{op}）")
    print("-" * 60)

    # 字符级结果
    print("\n【字符级评估】")
    print(f"字符编辑距离: {cer_edit_dist}")
    print(f"字符错误率（CER）: {cer:.4f} ({cer * 100:.2f}%)")
    print("字符对齐结果:")
    for i, (pred, true, op) in enumerate(char_alignment[:10], 1):
        print(f"  步骤{i}: {pred!r:4} → {true!r:4} （{op}）")
    print("  ...（省略后续步骤）")
    print("=" * 80)