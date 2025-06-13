import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class MiniRatUNet(nn.Module):
    def __init__(self, num_features=32):
        super(MiniRatUNet, self).__init__()

        # Input conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # Encoder
        self.enc1 = ConvBlock(num_features, num_features * 2)
        self.enc2 = ConvBlock(num_features * 2, num_features * 4)

        # Bottleneck
        self.bottleneck = ConvBlock(num_features * 4, num_features * 8)

        # Decoder
        self.up2 = UpBlock(num_features * 8 + num_features * 4, num_features * 4)  # 256 + 128 = 384
        self.up1 = UpBlock(num_features * 4 + num_features * 2, num_features * 2)  # 128 + 64 = 192
        self.up0 = UpBlock(num_features * 2 + num_features, num_features)          # 64 + 32 = 96


        # Final attention + conv
        self.attn = SpatialAttention()
        self.final_conv = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.enc1(F.avg_pool2d(x1, 2))  # down 1/2
        x3 = self.enc2(F.avg_pool2d(x2, 2))  # down 1/4

        xb = self.bottleneck(F.avg_pool2d(x3, 2))  # down 1/8

        x = self.up2(xb, x3)  # up to 1/4
        x = self.up1(x, x2)   # up to 1/2
        x = self.up0(x, x1)   # up to 1

        x = self.attn(x)
        x = self.final_conv(x)
        return x - x  # or: return x for clean output

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(attn))
        return x * attn


if __name__ == "__main__":
    model = MiniRatUNet(num_features=32).to("cuda")
    x = torch.randn(1, 1, 256, 256).to("cuda")

    # get time for one forward pass
    start_time = time.time()
    output = model(x)
    end_time = time.time()

    print(f"Time taken for one forward pass: {end_time - start_time:.6f} seconds")


    print(f"Output shape: {output.shape}")
    
    # Check if the output shape is correct
    assert output.shape == (1, 1, 256, 256), "Output shape is incorrect"
    print("Output shape is correct.")

    # check how many parameters the model has
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")