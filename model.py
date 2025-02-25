import torch
import torch.nn as nn


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        self.avgpool = nn.AdaptiveAvgPool3d((10, 10, 10))
        self.fc1 = nn.Linear(64 * 10 * 10, 256)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.fc1(x)
        x = self.relu6(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_weights = self.softmax(torch.bmm(query, key.transpose(1, 2)))
        y = torch.bmm(attention_weights, value)
        return y, attention_weights


class C3D_LSTM_Attention(nn.Module):
    def __init__(self, c3d, lstm, attention):
        super(C3D_LSTM_Attention, self).__init__()
        self.c3d = c3d
        self.lstm = lstm
        self.attention1 = attention
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # C3D
        x = self.c3d(x)
        # print(x.shape)

        # LSTM
        lstm_output, _ = self.lstm(x)
        # print(lstm_output.shape)
        x = torch.add(x, lstm_output)

        # Self-attention
        att_output, attention_weights = self.attention1(x)
        x = torch.add(att_output, lstm_output)

        # FC
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=64):
        super(ResNet3D, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3))
        self.bn1 = nn.BatchNorm3d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=(1, 2, 2))
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=(1, 2, 2))
        self.avgpool = nn.AdaptiveAvgPool3d((10, 10, 10))
        self.fc1 = nn.Linear(64 * 10 * 10, 512)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu7 = nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = out.reshape(out.size(0), out.size(1), -1)
        out = self.fc1(out)
        out = self.relu6(out)
        out = self.fc2(out)
        out = self.relu7(out)
        return out


def ResNet3D18(num_classes=64):
    return ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class Transformer(nn.Module):
    def __init__(self, input_size=64, hidden_size=256, heads=1, num_layers=num, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers

        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=64, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm([64, ])

        self.fc1 = nn.Linear(640, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(64, 64 * 4),
            nn.ReLU(),
            nn.Linear(64 * 4, 64)
        )

    def forward(self, query, key, value):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        for i in range(self.num_layers):
            query = self.norm(query)
            key = self.norm(key)
            value = self.norm(value)
            attn_output, attn_output_weights = self.multi_head_attention(query, key, value)
            x = self.dropout(self.norm(attn_output + query))
            feed_forward = self.feed_forward(x)
            query = self.dropout(feed_forward + x)

        x = query.reshape(query.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x,attn_output_weights


class MyModel(nn.Module):
    def __init__(self, c3d, lstm, attention):
        super(MyModel, self).__init__()
        self.model_1 = C3D_LSTM_Attention(c3d, lstm, attention)
        self.model_2 = ResNet3D18()
        self.model_3 = Transformer()

    def forward(self, num, x1, x2):
        x1 = self.model_1(x1)
        x2 = self.model_2(x2)
        # 被试、刺激、标签
        x3,attn_output_weights = self.model_3(x1, x2, x2)

        return x3

