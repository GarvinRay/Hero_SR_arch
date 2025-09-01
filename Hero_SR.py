# Hero-SR 里面的 U-net 以及 VAE 可以直接调 API
import torch
import torch.nn as nn
import torch.nn.functional as F
from OWMS import OWMS_Loss
from DTSM import DTSM
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel

# Image_lr ->(vae encoder) z_lr
# Image_lr ->(DTSM) t_star
# z_lr + t_star ->(U-net) z_sr
# z_sr ->(vae decoder) Image_sr
# Image_gt + Image_sr ->(OWMS) loss

class Hero_SR(nn.modules):
    def __init__(self, in_channels, sd_model_name_or_path, 
                 out_channels, tau, resblock_num, time_list, 
                 p_prompt, n_prompt, lambda_list, device):
        super(Hero_SR, self).__init__()
        self.sd_model_name_or_path = sd_model_name_or_path
        self.DtsmBlock = DTSM(in_channels, out_channels, tau, resblock_num, time_list)
        self.VaeEncoder = AutoencoderKL.from_pretrained(
            self.sd_model_name_or_path, subfolder='vae').to(device)
        self.Unet = UNet2DConditionModel.from_pretrained(
            self.sd_model_name_or_path, subfolder='unet').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sd_model_name_or_path, subfolder="tokenizer").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.sd_model_name_or_path, subfolder="text_encoder").to(device)
        # self.text_encoder.requires_grad_(False)
        
        self.p_prompt = p_prompt
        self.n_prompt = n_prompt
        self.lambda_list = lambda_list
        
    def forward(self, image_lr, image_gt):
        z_lr = self.VaeEncoder.encode(image_lr) # a tuple
        t_star = self.DtsmBlock(image_lr)
        z_sr = self.Unet(z_lr, t_star)
        image_sr = self.VaeEncoder.decode(z_sr)
        loss = OWMS_Loss(image_sr, image_gt, self.tokenizer, self.text_encoder, self.p_prompt, self.n_prompt, self.lambda_list)

        


