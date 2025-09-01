import torch
import torch.nn as nn
import torch.nn.functional as F
from ResBlock import ResBlock
class DTSM(nn.Module):
    def __init__(self, in_channels, out_channels, tau, resblock_num, time_list):
        super(DTSM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.resblocks = []
        for _ in range(resblock_num):
            self.resblocks.append(ResBlock(in_channels=out_channels, out_channels=out_channels, stride=1))
        self.resblocks = nn.Sequential(*self.resblocks)
    
        self.time_list = time_list
        
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 128 * 128, 128),  # 看了眼src：在forward里面是全部缩放到 128 * 128 的
            nn.ReLU(),
            nn.Linear(128, len(time_list))
        )
        
        self.tau = tau
        
    def forward(self, x):
        x = F.interpolate(x, size=(128, 128))
        f_shallow = self.conv(x)
        f_deep = self.resblocks(f_shallow)
        v = self.mlp(f_deep.flatten(1))  # (len, )
        time_probs = F.gumbel_softmax(v, tau=self.tau, hard=True)  # (len,)

        time_list = torch.tensor(self.time_list, device=x.device, dtype=torch.float32)  # (len,)
        t_star = (time_probs @ time_list).sum() 

        return t_star
