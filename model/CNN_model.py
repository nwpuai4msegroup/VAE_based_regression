from torch import nn

class CNN_model(nn.Module):
    def __init__(self, l1=20):
        super(CNN_model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.drop_1 = nn.Dropout(p=0.3)
        #self.drop_2 = nn.Dropout(p=0.2)
        self.gmp = nn.AdaptiveAvgPool2d((1,1))
        self.linear_1 = nn.Linear(64, l1)
        #self.linear_2 = nn.Linear(l1, l2)
        self.linear_3 = nn.Linear(l1, 1)
        self.flat = nn.Flatten()
    def forward(self, x):
        out = self.features(x)
        out = self.gmp(out)
        #print(out.shape)
        out = self.flat(out)
        #print(out.shape)
        out = self.linear_1(out)
        out = self.drop_1(out)
        #out = self.linear_2(out)
        #out = self.drop_2(out)
        out = self.linear_3(out)

        feature_map = []
        for name, module in self.features.named_children():
            x = module(x)
            if name in ["0", "3", "7", "10", "14", "17"]:
                feature_map.append(x)
        return out, feature_map

# 模型构建
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                    stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                    stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_result=1):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                    stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(5)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.gmp = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()
        self.drop_1 = nn.Dropout(p=0.5)
        self.drop_2 = nn.Dropout(p=0.2)
        self.linear_1 = nn.Linear(64, 20)
        self.linear_2 = nn.Linear(20, 12)
        self.linear_3 = nn.Linear(12, num_result)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.pool2(out)
        out = self.layer2(out)
        out = self.pool2(out)
        out = self.layer3(out)
        #out = self.pool2(out)
        out = self.layer4(out)
        out = self.gmp(out)
        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        out = self.flat(out)
        out = self.linear_1(out)
        out = self.drop_1(out)
        out = self.linear_2(out)
        out = self.drop_2(out)
        out = self.linear_3(out)
        #out = out.squeeze(-1)
        return out



def ResNet18():
    return ResNet(BasicBlock, [1,1,1,1])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])
