import h5py
import numpy as np
import os

def average_data(algorithm="", dataset="", goal="", times=10):
    # 读取WER和CER结果
    test_wer = get_all_results_for_one_algo(algorithm, dataset, goal, times, metric='rs_test_wer')


    # 计算最佳WER/CER的均值和标准差
    min_wer = [wer.min() for wer in test_wer]


    print("std for best WER:", np.std(min_wer))
    print("mean for best WER:", np.mean(min_wer))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, metric='rs_test_wer'):
    results = []
    for i in range(times):
        file_name = f"{dataset}_{algorithm}_{goal}_{i}"
        results.append(np.array(read_data_then_delete(file_name, metric=metric, delete=False)))
    return results

def read_data_then_delete(file_name, metric='rs_test_wer', delete=False):
    file_path = f"../results/{file_name}.h5"
    with h5py.File(file_path, 'r') as hf:
        data = np.array(hf.get(metric))
    if delete:
        os.remove(file_path)
    print(f"Length of {metric}: {len(data)}")
    return data