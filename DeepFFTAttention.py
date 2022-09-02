from layers import *

class EBlock(nn.Module):
    def __init__(self, out_channel, height, width, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, height, width) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class DBlock(nn.Module):
    def __init__(self, channel, height, width, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, height, width) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    
class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel,
                      kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel,
                      kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane//4,
                      kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2,
                      kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2,
                      kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-inchannel,
                      kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane,
                              kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(
            channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class DeepFFTAttention(nn.Module):
    def __init__(self, num_res=8, inference=False):
        super(DeepFFTAttention, self).__init__()
        self.inference = inference
        BasicConv = BasicConv_do
        ResBlock = FFTAttentionBlock
        base_channel = 32
        height, width = 256,256

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, height, width, num_res, ResBlock=ResBlock),
            EBlock(base_channel*2, height//2, width //
                   2, num_res, ResBlock=ResBlock),
            EBlock(base_channel*4, height//4, width //
                   4, num_res, ResBlock=ResBlock),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2,
                      kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4,
                      kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4,
                      relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4,
                      relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, height//4, width //
                   4, num_res, ResBlock=ResBlock),
            DBlock(base_channel * 2, height//2, width //
                   2, num_res, ResBlock=ResBlock),
            DBlock(base_channel, height, width, num_res, ResBlock=ResBlock)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2,
                      kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel,
                      kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3,
                          relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3,
                          relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel*2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x) # 32,256,256
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)  # 64
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)  # 224, 64
        res1 = self.AFFs[0](res1, z21, z41)  # 224, 32

        # print(z.shape)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if not self.inference:
            outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        # print(z.shape)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if not self.inference:
            outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        # print(z.shape)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            outputs.append(z+x)
            # print(outputs)
            return outputs[::-1]
        else:
            return z+x