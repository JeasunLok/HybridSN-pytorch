import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

class HybridSN(nn.Module):
    def __init__(self, in_channels, patch_size, num_classes):
        super().__init__()
        self.in_chs = in_channels
        self.patch_size = patch_size
        
        # 3D卷积部分
        self.conv1 = nn.Sequential(
                    nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
                    nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
                    nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
                    nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
                    nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
                    nn.ReLU(inplace=True))
        
        # 获取3D卷积后张量形状
        self.x1_shape = self.get_shape_after_3dconv()
        
        # 2D卷积部分
        self.conv4 = nn.Sequential(
                    nn.Conv2d(in_channels=self.x1_shape[1] * self.x1_shape[2], out_channels=64, kernel_size=(3, 3)),
                    nn.ReLU(inplace=True))
        
        # 获取2D卷积后张量形状
        self.x2_shape = self.get_shape_after_2dconv()
        
        # 全连接层
        self.dense1 = nn.Sequential(
                    nn.Linear(self.x2_shape, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4))
        
        self.dense2 = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4))
        
        self.dense3 = nn.Linear(128, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 使用 He 初始化卷积层和全连接层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def get_shape_after_3dconv(self):
        # 计算通过 3D 卷积后的形状
        x = torch.zeros((1, 1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape
    
    def get_shape_after_2dconv(self):
        # 计算通过 2D 卷积后的形状
        x = torch.zeros((1, self.x1_shape[1] * self.x1_shape[2], self.x1_shape[3], self.x1_shape[4]))
        with torch.no_grad():
            x = self.conv4(x)
        return x.shape[1] * x.shape[2] * x.shape[3]
    
    def forward(self, X):
        # 前向传播
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 将3D卷积的输出展平以适应2D卷积
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.conv4(x)
        
        # 将2D卷积的输出展平并通过全连接层
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        
        return out


if __name__ == '__main__':
    model = HybridSN(166,33,num_classes=11).cuda()
    input = torch.randn([2,1,166,33,33]).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(model(input).shape)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}M".format(flops / 1e6))
    print("params:{:.3f}M".format(params / 1e6))
    # --------------------------------------------------#
    #   用来测试网络能否跑通，同时可查看FLOPs和params
    # --------------------------------------------------#
    summary(model, input_size=(1,166,33,33), batch_size=-1)
