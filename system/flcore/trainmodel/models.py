import torch
import torch.nn as nn
import torch.nn.functional as F


class FedAvgASRModel(nn.Module):
    def __init__(self,
                 in_channels=1,  # 语音特征通道
                 mel_dim=80,  # 梅尔频谱特征维度
                 hidden_dim=256,  # 全连接层隐藏维度
                 vocab_size=941,  # 词汇表大小
                 pool_size=8):  # 总下采样倍数
        super().__init__()
        self.pool_size = pool_size
        self.mel_dim = mel_dim

        # 卷积块1：输入(batch, 1, seq_len, mel_dim)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 时序/频率各下采样2倍
        )

        # 卷积块2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 累计下采样4倍
        )

        # 卷积块3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 时序下采样2倍（累计8倍），频率维度不变
        )

        # 卷积块4（加深特征提取，仅时序下采样）
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128, eps=0.0002),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))  # 不额外下采样，保持特征图尺寸
        )

        # 全连接层：将CNN特征映射到词汇表
        # 计算全连接层输入维度：128（通道） * (mel_dim//4)（频率下采样后）
        self.fc_input_dim = 128 * (mel_dim // 4)  # 频率维度经2次2倍下采样后为mel_dim//4
        self.dense1 = nn.Linear(self.fc_input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, vocab_size)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        # 输入x形状：(batch, in_channels, seq_len, mel_dim) → 例如(32, 1, 1600, 80)
        batch_size = x.shape[0]
        seq_len = x.shape[2]

        # 经过4个卷积块提取特征
        x = self.conv_block1(x)  # (batch, 32, seq_len//2, mel_dim//2)
        x = self.conv_block2(x)  # (batch, 64, seq_len//4, mel_dim//4)
        x = self.conv_block3(x)  # (batch, 128, seq_len//8, mel_dim//4)
        x = self.conv_block4(x)  # (batch, 128, seq_len//8, mel_dim//4)

        # 调整维度：将频率维度与通道维度合并，保留时序维度
        # 输出形状：(batch, seq_len//8, 128 * (mel_dim//4))
        x = x.permute(0, 2, 3, 1)  # (batch, seq_len//8, mel_dim//4, 128)
        x = x.reshape(batch_size, -1, self.fc_input_dim)  # (batch, seq_len//8, 128*(mel_dim//4))

        # 全连接层映射到词汇表
        x = F.relu(self.dense1(x))  # (batch, seq_len//8, hidden_dim)
        logits = self.dense2(x)  # (batch, seq_len//8, vocab_size)

        # 计算log_probs（CTC损失需要）
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs