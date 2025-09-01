import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
 
def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=-1)

def normalize(x):
    return x / (x.norm(dim=-1, keepdim=True))


def OWMS_Loss(Image_sr, Image_gt, f_image, f_text, p_prompt, n_prompt,lambda_list):   
    mse_fn = nn.MSELoss()
    mse = mse_fn(Image_sr,Image_gt)
    
    lpips_fn = lpips.LPIPS(net='vgg')
    loss_lpips = lpips_fn(Image_sr, Image_gt)
    
    td_pal = TD_PAL(Image_sr, f_image, f_text, p_prompt, n_prompt)
    id_sal = ID_SAL(Image_sr, Image_gt, f_image)
    lambda_1 = lambda_list[0]
    lambda_2 = lambda_list[1]
    lambda_3 = lambda_list[2]
    lambda_4 = lambda_list[3]
    loss = lambda_1 * mse + lambda_2 * loss_lpips+ lambda_3 * td_pal + lambda_4 * id_sal
    return loss

def TD_PAL(Image_sr, f_image, f_text, p_prompt, n_prompt):
    e_sr = normalize(f_image(Image_sr))       
    e_p  = normalize(f_text(p_prompt))       
    e_n  = normalize(f_text(n_prompt))        

    s_p = cosine_similarity(e_sr, e_p)       
    s_n = cosine_similarity(e_sr, e_n)       
    
    s_hat_p = torch.exp(s_p) / (torch.exp(s_p) + torch.exp(s_n))

    loss = 1 - s_hat_p.mean()
    return loss
def ID_SAL(Image_sr, Image_gt, f_image):
    e_sr = normalize(f_image(Image_sr))   
    e_gt = normalize(f_image(Image_gt))  

    s = cosine_similarity(e_sr, e_gt) 
    loss = 1 - s.mean()
    return loss

