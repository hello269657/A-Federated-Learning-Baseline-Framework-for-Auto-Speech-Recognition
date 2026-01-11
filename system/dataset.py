import os
import numpy as np
import random
from collections import defaultdict
from scipy.fftpack import dct

# ========================== 1. 配置参数==========================
AUDIO_PATH_FILE = r"D:"  # 训练集/测试集路径
LABEL_FILE = r"D:"
RAW_AUDIO_ROOT = r"D:"

DATA_TYPE = "train"
OUTPUT_ROOT = "../dataset/THCHS30_ASR"
DATA_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, DATA_TYPE)
DICT_PATH = r"D:"

# 音频基础参数
SAMPLE_RATE = 16000
WINLEN = 0.025
WINSTEP = 0.01
N_FFT = int(WINLEN * SAMPLE_RATE)
HOP_LENGTH = int(WINSTEP * SAMPLE_RATE)

# MFCC特征参数
NUMCEP = 27
NFILT = 54
PREEMPH = 0.97
CEPLIFTER = 22
APPEND_ENERGY = True

# 客户端与标签参数
CLIENT_SPLIT_MODE = "uniform"  # 全量均匀划分
MIN_FILES_PER_CLIENT = 5
TARGET_CLIENT_NUM = 10  # 固定划分10个客户端
MAX_LABEL_LEN = 100
SPECIAL_TOKENS = {"<blank>": 0, "UNK": 1}


# ========================== 2. 信号处理工具函数==========================
def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    slen = len(sig)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))

    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(np.ceil((slen - frame_len) / frame_step))

    padlen = (numframes - 1) * frame_step + frame_len
    padsignal = np.concatenate((sig, np.zeros((padlen - slen,))))

    shape = padsignal.shape[:-1] + (padsignal.shape[-1] - frame_len + 1, frame_len)
    strides = padsignal.strides + (padsignal.strides[-1],)
    frames = np.lib.stride_tricks.as_strided(padsignal, shape=shape, strides=strides)[::frame_step]

    win = winfunc(frame_len)
    return frames * win


def powspec(frames, nfft):
    complex_spec = np.fft.rfft(frames, nfft)
    mag_spec = np.absolute(complex_spec)
    return 1.0 / nfft * np.square(mag_spec)


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(nfilt=26, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def lifter(cepstra, L=22):
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
        return lift * cepstra
    return cepstra


def delta(feat, N=2):
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')

    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N + 1), padded[t: t + 2 * N + 1]) / denominator
    return delta_feat


# ========================== 3. 特征提取基类与MFCC子类（不变）==========================
class SpeechFeatureMeta:
    def __init__(self, framesamplerate=16000):
        self.framesamplerate = framesamplerate

    def run(self, wavsignal, fs=16000):
        raise NotImplementedError('[ASRT] `run()` method is not implemented.')


class MFCC(SpeechFeatureMeta):
    def __init__(self, framesamplerate=16000, winlen=0.025, winstep=0.01,
                 numcep=13, nfilt=26, preemph=0.97, ceplifter=22, appendEnergy=True):
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.preemph = preemph
        self.ceplifter = ceplifter
        self.appendEnergy = appendEnergy
        super().__init__(framesamplerate)

    def compute_mfcc(self, signal):
        signal = preemphasis(signal, self.preemph)

        frame_len = self.winlen * self.framesamplerate
        frame_step = self.winstep * self.framesamplerate
        frames = framesig(signal, frame_len, frame_step)

        nfft = int(2 ** np.ceil(np.log2(frame_len)))
        pspec = powspec(frames, nfft)

        fb = get_filterbanks(self.nfilt, nfft, self.framesamplerate)
        feat = np.dot(pspec, fb.T)
        feat = np.where(feat == 0, np.finfo(float).eps, feat)

        feat = np.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:, :self.numcep]
        feat = lifter(feat, self.ceplifter)

        if self.appendEnergy:
            energy = np.sum(pspec, 1)
            energy = np.where(energy == 0, np.finfo(float).eps, energy)
            feat[:, 0] = np.log(energy)

        return feat

    def run(self, wavsignal, fs=16000):
        wavsignal = np.array(wavsignal, dtype=np.float64)

        feat_mfcc = self.compute_mfcc(wavsignal)
        feat_mfcc_d = delta(feat_mfcc, 2)
        feat_mfcc_dd = delta(feat_mfcc_d, 2)

        wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
        return wav_feature


# ========================== 4. 加载词汇表==========================
def load_vocab(dict_path):
    char_to_id = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"dict.txt第{line_num}行格式错误：{line}（需用\t分隔符号和ID）")
            char, id_str = parts
            try:
                idx = int(id_str)
            except ValueError:
                raise ValueError(f"dict.txt第{line_num}行ID非整数：{line}")
            if char in char_to_id:
                raise ValueError(f"dict.txt第{line_num}行符号{char}重复（首次ID={char_to_id[char]}）")
            char_to_id[char] = idx

    missing_tokens = [tok for tok in SPECIAL_TOKENS if tok not in char_to_id]
    if missing_tokens:
        raise ValueError(f"dict.txt缺少必需符号：{missing_tokens}（需添加<blank>\t0和UNK\t1）")
    for tok, exp_idx in SPECIAL_TOKENS.items():
        if char_to_id[tok] != exp_idx:
            raise ValueError(f"dict.txt中{tok}的ID必须为{exp_idx}（CTC要求），当前为{char_to_id[tok]}")

    print(f"词汇表加载成功：共{len(char_to_id)}个符号，blank=0，UNK=1")
    return char_to_id


# ========================== 5. 文本→ID==========================
def transcript_to_ids(transcript, char_to_id, max_len):
    chars = transcript.strip().split() if transcript.strip() else []
    if not chars:
        return np.full(max_len, char_to_id["<blank>"], dtype=np.int32)

    id_seq = [char_to_id.get(char, char_to_id["UNK"]) for char in chars]

    if len(id_seq) > max_len:
        id_seq = id_seq[:max_len]
    else:
        id_seq += [char_to_id["<blank>"]] * (max_len - len(id_seq))

    return np.array(id_seq, dtype=np.int32)


# ========================== 6. 音频特征提取==========================
def load_wav(wav_path, sample_rate=16000):
    import wave
    with wave.open(wav_path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        data = wf.readframes(n_frames)
        if sample_width == 2:
            signal = np.frombuffer(data, dtype=np.int16)
        elif sample_width == 4:
            signal = np.frombuffer(data, dtype=np.int32)
        else:
            raise ValueError(f"不支持的采样宽度：{sample_width}")

        if channels > 1:
            signal = np.mean(signal.reshape(-1, channels), axis=1)

        if framerate != sample_rate:
            from scipy.signal import resample
            signal = resample(signal, int(n_frames * sample_rate / framerate))

        signal = signal.astype(np.float64) / np.iinfo(signal.dtype).max
        return signal


def extract_feature(wav_path, feature_extractor):
    """提取特征"""
    signal = load_wav(wav_path, sample_rate=feature_extractor.framesamplerate)
    if len(signal) == 0:
        raise ValueError(f"音频文件为空：{wav_path}")

    feature = feature_extractor.run(signal)
    if feature.shape[-1] > 80:
        feature = feature[:, :80]  # 保留前80维，丢弃最后1维
    if feature.shape[0] == 0:
        raise ValueError(f"特征提取失败（音频过短）：{wav_path}")

    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature)) * 2 - 1
    return feature


# ========================== 7. 建立音频-标签映射==========================
def build_audio_label_map(audio_path_file, label_file, raw_audio_root):
    audio_id_to_path = {}
    with open(audio_path_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            audio_id, wav_rel_path = line.split(maxsplit=1)
            wav_abs_path = os.path.join(raw_audio_root, wav_rel_path)
            if not os.path.exists(wav_abs_path):
                print(f"警告：音频不存在，跳过：{wav_abs_path}")
                continue
            audio_id_to_path[audio_id] = wav_abs_path

    audio_id_to_label = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"警告：第{line_num}行标签格式错误，跳过：{line}")
                continue
            audio_id, transcript = parts
            if not isinstance(transcript, str) or transcript.replace(" ", "").isdigit():
                print(f"警告：第{line_num}行标签非文本，跳过：{transcript}")
                continue
            audio_id_to_label[audio_id] = transcript

    common_ids = set(audio_id_to_path.keys()) & set(audio_id_to_label.keys())
    audio_label_map = {
        aid: {"wav_path": audio_id_to_path[aid], "transcript": audio_id_to_label[aid]}
        for aid in common_ids
    }

    if len(audio_label_map) > 0:
        sample_transcripts = list(audio_label_map.values())[:3]
        print("原始文本标签示例（前3个）：")
        for i, info in enumerate(sample_transcripts):
            print(f"  样本{i + 1}：{info['transcript']}（类型：{type(info['transcript']).__name__}）")

    print(f"成功匹配音频-标签对：{len(audio_label_map)} 个（全量参与划分）")
    return audio_label_map


# ========================== 8. 划分客户端==========================
def split_audio_to_clients(audio_label_map, split_mode="uniform", min_files=5, target_num=10):
    clients = defaultdict(list)
    all_audio_ids = list(audio_label_map.keys())
    random.shuffle(all_audio_ids)  # 打乱顺序，保证分配随机性
    total_audio = len(all_audio_ids)

    if split_mode == "uniform":
        # 模式1：全量均匀分配
        audio_per_client = total_audio // target_num
        remainder = total_audio % target_num  # 余数音频均匀分配给前N个客户端

        start_idx = 0
        for client_id in range(target_num):
            end_idx = start_idx + audio_per_client + (1 if client_id < remainder else 0)
            clients[client_id] = all_audio_ids[start_idx:end_idx]
            start_idx = end_idx
            print(f"客户端{client_id}：{len(clients[client_id])}个音频")

    elif split_mode == "speaker":
        # 模式2：按说话人全量分配
        audio_id_to_speaker = {}
        for audio_id in all_audio_ids:
            # 提取说话人ID
            speaker_id = audio_id.split("_")[0] if "_" in audio_id else audio_id[:4]
            audio_id_to_speaker[audio_id] = speaker_id
        # 按说话人分组
        speaker_to_audio = defaultdict(list)
        for audio_id, speaker in audio_id_to_speaker.items():
            speaker_to_audio[speaker].append(audio_id)
        all_speakers = list(speaker_to_audio.items())
        # 均匀分配说话人到10个客户端
        speaker_per_client = len(all_speakers) // target_num
        remainder = len(all_speakers) % target_num
        start_idx = 0
        for client_id in range(target_num):
            end_idx = start_idx + speaker_per_client + (1 if client_id < remainder else 0)
            client_speakers = all_speakers[start_idx:end_idx]
            # 收集该客户端所有说话人的音频
            client_audio_ids = []
            for speaker, audio_ids in client_speakers:
                client_audio_ids.extend(audio_ids)
            clients[client_id] = client_audio_ids
            start_idx = end_idx
            print(f"客户端{client_id}：包含{len(client_speakers)}个说话人，{len(client_audio_ids)}个音频")

    else:
        for i, aid in enumerate(all_audio_ids):
            clients[i % target_num].append(aid)

    # 验证数据完整性
    total_assigned = sum(len(audio_ids) for audio_ids in clients.values())
    print(f"\n分配验证：总有效音频数={total_audio}，已分配音频数={total_assigned}")
    if total_assigned != total_audio:
        print(f"警告：有{total_audio - total_assigned}个音频未分配！")

    print(f"最终客户端数：{len(clients)} 个（目标：{target_num}个）")
    return clients


# ========================== 9. 生成npz==========================
def generate_client_npz(clients, audio_label_map, char_to_id, output_dir, feature_extractor, max_label_len=100):
    os.makedirs(output_dir, exist_ok=True)
    for client_id, audio_ids in clients.items():
        feat_list = []
        transcript_list = []
        label_ids_list = []

        for idx, audio_id in enumerate(audio_ids):
            info = audio_label_map[audio_id]
            wav_path = info["wav_path"]
            transcript = info["transcript"]

            if not isinstance(transcript, str):
                print(f"错误：客户端{client_id}音频{audio_id}的transcript非文本，跳过")
                continue

            try:
                feat = extract_feature(wav_path, feature_extractor)
                label_ids = transcript_to_ids(transcript, char_to_id, max_label_len)

                feat_list.append(feat)
                transcript_list.append(transcript)
                label_ids_list.append(label_ids)

                if idx < 3:  # 打印前3个样本信息
                    print(f"\n客户端{client_id}样本{idx + 1}（音频ID：{audio_id}）：")
                    print(f"  原始文本：{transcript}")
                    print(f"  label_ids（前15个）：{label_ids[:15]}")
                    print(f"  特征形状：{feat.shape}（时间步×特征维度）")

            except Exception as e:
                print(f"处理音频{audio_id}出错：{str(e)}，跳过")
                continue

        if len(feat_list) == 0:
            print(f"警告：客户端{client_id}无有效数据，跳过")
            continue

        data_dict = {
            "feature": feat_list,  # 80维特征
            "transcript": transcript_list,
            "label_ids": label_ids_list
        }

        npz_path = os.path.join(output_dir, f"{client_id}.npz")
        np.savez_compressed(npz_path, data=data_dict, allow_pickle=True)
        print(f"\n已生成客户端{client_id}npz文件：{npz_path}（{len(feat_list)}个有效样本）")


# ========================== 10. 主函数==========================
if __name__ == "__main__":
    print("=" * 60)
    print(f"开始处理{DATA_TYPE}集数据集（全量划分模式）")
    print("=" * 60)

    print(f"\n1. 正在加载词汇表：{DICT_PATH}")
    CHAR_TO_ID = load_vocab(DICT_PATH)

    print(f"\n2. 正在建立音频-标签映射（全量匹配，不丢弃有效数据）...")
    audio_label_map = build_audio_label_map(AUDIO_PATH_FILE, LABEL_FILE, RAW_AUDIO_ROOT)
    if len(audio_label_map) == 0:
        print("错误：无有效音频-标签对，终止程序")
        exit(1)

    print(f"\n3. 初始化MFCC特征提取器...")
    mfcc_extractor = MFCC(
        framesamplerate=SAMPLE_RATE,
        winlen=WINLEN,
        winstep=WINSTEP,
        numcep=NUMCEP,
        nfilt=NFILT,
        preemph=PREEMPH,
        ceplifter=CEPLIFTER,
        appendEnergy=APPEND_ENERGY
    )
    print(f"MFCC配置：采样率={SAMPLE_RATE}Hz，最终特征维度：80（81维截断）")

    print(f"\n4. 正在全量划分{TARGET_CLIENT_NUM}个客户端（划分模式：{CLIENT_SPLIT_MODE}）...")
    clients = split_audio_to_clients(audio_label_map, CLIENT_SPLIT_MODE, MIN_FILES_PER_CLIENT, TARGET_CLIENT_NUM)
    if len(clients) == 0:
        print("错误：无有效客户端，终止程序")
        exit(1)

    print(f"\n5. 正在生成{DATA_TYPE}集客户端npz文件...")
    generate_client_npz(clients, audio_label_map, CHAR_TO_ID, DATA_OUTPUT_DIR, mfcc_extractor, MAX_LABEL_LEN)

    print(f"\n" + "=" * 60)
    print(f"{DATA_TYPE}集数据集制作完成！")
    print(f"客户端npz输出路径：{DATA_OUTPUT_DIR}")
    print(f"关键信息：10个客户端，全量数据分配，特征维度80")
    print(f"词汇表路径：{DICT_PATH}")
    print("=" * 60)