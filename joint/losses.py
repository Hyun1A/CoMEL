import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, n_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(n_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.n_classes = n_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, cos, T, mask=None):  
        pos_one_hot = F.one_hot(T, num_classes=self.n_classes)
        neg_one_hot = 1-pos_one_hot
        
        if T.sum() == 0:
            num_valid_proxies=1
        else:
            num_valid_proxies=2

        # breakpoint()
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        # P_sim_sum = torch.where(pos_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).mean(dim=0)       
        # N_sim_sum = torch.where(neg_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).mean(dim=0)
        
        if mask is not None:
            P_sim_sum = (mask.unsqueeze(-1) * torch.where(pos_one_hot == 1, pos_exp, torch.zeros_like(pos_exp))).sum(dim=0) / (pos_one_hot.sum(dim=0)+1e-10)       
            N_sim_sum = (mask.unsqueeze(-1) * torch.where(neg_one_hot == 1, neg_exp, torch.zeros_like(neg_exp))).sum(dim=0) / (neg_one_hot.sum(dim=0)+1e-10)
        else:
            P_sim_sum = torch.where(pos_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) / (pos_one_hot.sum(dim=0)+1e-10)       
            N_sim_sum = torch.where(neg_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0) / (neg_one_hot.sum(dim=0)+1e-10)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / num_valid_proxies
        loss = pos_term + neg_term
        
        return loss
    
    
class Proxy_Reg(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        
    def forward(self, proxies):  
        proxies_norm = proxies/proxies.norm(dim=1, keepdim=True)
        
        sim_mat = proxies_norm@proxies_norm.T

        loss = sim_mat.norm(2,dim=1).sum()

        return loss    
    
    
class TokenContrast(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        
        # attention_modules = {}
        # for k,v in model.named_modules(): 
        #     #if v.__class__.__name__ == "InnerAttention":
        #     if "attn.qkv" in k:
        #         attention_modules[k] = v
                
        # attn_info_modules = []
        # for k,v in model.named_modules(): 
        #     if v.__class__.__name__ == "InnerAttention":
        #         attn_info_modules.append([k,v]) 

        # attention_outputs = []
        
        # def hook_fn(module, input, output):
        #     attention_outputs.append(output)
        
        # for k,v in attention_modules.items():        
        #     hook = v.register_forward_hook(hook_fn)
                        
        
        
        
        
    def forward(self, attention_outputs, patch_labels):  
        
        # ### token contrast loss
        # M, N, C = attention_outputs[0].shape
        # num_heads = attn_info_modules[0][1].num_heads
        # head_dim = attn_info_modules[0][1].head_dim

        # qkv = attention_outputs[0].reshape(M, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        # q, k, _ = qkv[0], qkv[1], qkv[2]        
        
        features = attention_outputs[0] ## 
        features_normed=features/(features.norm(2, dim=2, keepdim=True)+1e-10)
        sim_mat = torch.einsum("ijk, ilk -> ijl", features_normed, features_normed)
        
        add_length = features.shape[0]*features.shape[1] - patch_labels.shape[0]
        patch_labels_cat  = torch.cat([patch_labels, torch.zeros(add_length,device=patch_labels.device)], dim=0)
        num_region = int(math.sqrt(features.shape[0]))
        region_size = int(math.sqrt(features.shape[1]))
        patch_labels_cat = patch_labels_cat.view(num_region*region_size, num_region*region_size)
        patch_labels_cat = patch_labels_cat.view(num_region, region_size, num_region, region_size)
        patch_labels_cat = patch_labels_cat.permute(0,2,1,3).contiguous().view(-1, region_size, region_size)
        patch_labels_cat = patch_labels_cat.view(-1, region_size*region_size)
        
        pos_map = torch.einsum("ij,ik->ijk", patch_labels_cat, patch_labels_cat) + torch.einsum("ij,ik->ijk", 1-patch_labels_cat, 1-patch_labels_cat)
        neg_map = 1-pos_map
        
        num_pos = pos_map.view(-1).sum()
        num_neg = neg_map.view(-1).sum()
        
        pos_cos = (pos_map*sim_mat).view(-1)
        neg_cos = (neg_map*sim_mat).view(-1)
        
        toco_loss = (1-pos_cos).sum() / (num_pos+1e-10) + (neg_cos).sum() / (num_neg+1e-10)
        
        return toco_loss    
    
    
    