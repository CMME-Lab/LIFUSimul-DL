import torch.nn as nn
from einops import rearrange
from .modules import SelfAttention, PatchEmbed3D, StageModule, CNNDouble

class SwinUNet(nn.Module):
    def __init__(self, device='cpu', in_channels=1, out_channels=1, in_dim=96, dropout_attn=0.1, dropout_mlp=0.1, data_size=(112, 112, 112)):
        super().__init__()

        self.d1, self.d2, self.d3 = [s // 16 for s in data_size]

        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        self.sa = SelfAttention(1, 1)

        self.linear = nn.Sequential(
            nn.Linear(self.d1*self.d2*self.d3*2, self.d1*self.d2*self.d3),
            nn.GELU()
        )
        
        self.patchemb1 = PatchEmbed3D(patch_size=(4,4,4), in_chans=1, embed_dim=in_dim)
        self.patchemb2 = PatchEmbed3D(patch_size=(4,4,4), in_chans=1, embed_dim=in_dim)

        self.stage1 = StageModule(in_channels=in_channels, hidden_dimension=in_dim,
                                  downscaling_factor=2, num_heads=3, window_size=(7,7,7), device=device,
                                  dropout_attn=0.0, dropout_mlp=0.0, first_block=True, depth=2)
        self.stage2 = StageModule(in_channels=in_dim, hidden_dimension=in_dim*2,
                                  downscaling_factor=2, num_heads=6, window_size=(7,7,7), device=device,
                                  dropout_attn=dropout_attn, dropout_mlp=dropout_mlp, depth=2)
        self.stage3 = StageModule(in_channels=in_dim*2, hidden_dimension=in_dim*4,
                                  downscaling_factor=2, num_heads=12, window_size=(7,7,7), device=device,
                                  dropout_attn=dropout_attn, dropout_mlp=dropout_mlp, depth=2)

        self.Sstage1 = StageModule(in_channels=in_channels, hidden_dimension=in_dim,
                                  downscaling_factor=2, num_heads=3, window_size=(7,7,7), device=device,
                                  dropout_attn=0.0, dropout_mlp=0.0, first_block=True, depth=2)
        self.Sstage2 = StageModule(in_channels=in_dim, hidden_dimension=in_dim*2,
                                  downscaling_factor=2, num_heads=6, window_size=(7,7,7), device=device,
                                  dropout_attn=dropout_attn, dropout_mlp=dropout_mlp, depth=2)
        self.Sstage3 = StageModule(in_channels=in_dim*2, hidden_dimension=in_dim*4,
                                  downscaling_factor=2, num_heads=12, window_size=(7,7,7), device=device,
                                  dropout_attn=dropout_attn, dropout_mlp=dropout_mlp, depth=2)
        
        self.upstage1 = CNNDouble(scale_factor=2, in_size=in_dim*4, out_size=in_dim*2, dropout=0.1)
        self.upstage2 = CNNDouble(scale_factor=2, in_size=in_dim*2, out_size=in_dim, dropout=0.1)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
            nn.Conv3d(in_dim, out_channels, 3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, field, skull, tinput):

        # InstanceNorm
        field = self.patchemb1(self.norm1(field))
        skull = self.patchemb2(self.norm2(skull))

        # pressure field
        d1 = self.stage1(field) # 28
        d2 = self.stage2(d1) # 14
        d3 = self.stage3(d2) # 7

        # medical image
        s1 = self.Sstage1(skull)
        s2 = self.Sstage2(s1)
        s3 = self.Sstage3(s2)

        # tinput
        tinput = self.linear(tinput).view(-1, 1, self.d1, self.d2, self.d3)
        tinput = self.sa(tinput).view(-1, self.d1, self.d2, self.d3, 1)

        # merge
        x = d3 + s3 + tinput

        # upsample + skip connection
        x = self.upstage1(x)
        x = x + d2 + s2
        x = self.upstage2(x)
        x = x + d1 + s1
        
        x = rearrange(x, 'b d h w c -> b c d h w')

        return self.final(x)