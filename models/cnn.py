import torch.nn as nn
from .modules import SelfAttention, Down, Mid, Up

class CNNs(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, channels = 32, dropout = 0.5, skip=True, data_size = (112, 112, 112)):
        super(CNNs, self).__init__()

        channel_nums = [channels, channels * 2, channels * 4, channels * 8]
        self.d1, self.d2, self.d3 = [s // 16 for s in data_size]
        
        self.skip = skip # set True if using U-Net
        
        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        
        self.st = nn.Sequential(
            nn.Linear(self.d1*self.d2*self.d3*2, self.d1*self.d2*self.d3),
            nn.GELU()
        )

        self.encoder1 = nn.ModuleList().append(Down(in_channels, channels))
        self.encoder2 = nn.ModuleList().append(Down(in_channels, channels))
        self.decoder = nn.ModuleList()

        for i in range(3):
            self.encoder1.append(Down(channel_nums[i], channel_nums[i+1], dropout=0.1))
            self.encoder2.append(Down(channel_nums[i], channel_nums[i+1], dropout=0.1))
            self.decoder.append(Up(channel_nums[-(i+1)], channel_nums[-(i+2)], dropout=0.1))

        self.sa1 = SelfAttention(channels * 8)
        self.sa2 = SelfAttention(channels * 8)
        self.sa3 = SelfAttention(channels * 8)
        self.sa_t = SelfAttention(1, 1)
        
        self.mid1 = Mid(channels * 8, channels * 8, dropout=dropout)
        self.mid2 = Mid(channels * 8, channels * 8, dropout=dropout)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(channels, out_channels, 3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, field, skull, tinput):

        field = self.norm1(field)
        skull = self.norm2(skull)

        f_skips = []
        s_skips = []

        for i, (enc1, enc2) in enumerate(zip(self.encoder1[:-1], self.encoder2[:-1])):
            field = enc1(field)
            f_skips.append(field)

            skull = enc2(skull)
            s_skips.append(skull)

        field = self.sa1(self.encoder1[-1](field))
        skull = self.sa2(self.encoder2[-1](skull))

        tinput = self.st(tinput).view(-1, 1, self.d1, self.d2, self.d3)
        tinput = self.sa_t(tinput)
        
        x = field + tinput + skull

        x = x + self.mid1(x) if self.skip else self.mid1(x)
        x = x + self.mid2(x) if self.skip else self.mid2(x)
        
        x = self.sa3(x)

        f_skips.reverse()
        s_skips.reverse()

        for (dec, f_skip, s_skip) in zip(self.decoder, f_skips[:-1], s_skips[:-1]):
            x = dec(x) + f_skip + s_skip if self.skip else dec(x)

        x = self.decoder[-1](x) + f_skips[-1] if self.skip else self.decoder[-1](x)

        return self.final(x)