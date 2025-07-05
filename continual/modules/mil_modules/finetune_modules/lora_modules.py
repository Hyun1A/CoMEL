
import os
from copy import deepcopy
import math
from typing import Optional, List
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file

from modules.emb_position import *
from modules.datten import *
from modules.rmsa import *
from ..nystrom_attention import NystromAttention
from modules.datten import DAttention
from timm.models.layers import DropPath


class LoRALayer(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.dim = dim

        if org_module.__class__.__name__ == "Linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, dim, bias=False)
            self.lora_up = nn.Linear(dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{lora_name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )
        
        

class Arc_LoRALayer(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.dim = dim

        if org_module.__class__.__name__ == "Linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, dim, bias=False)
            self.lora_up = nn.Linear(dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{lora_name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.scale
                
        
        
        
        
        
        
class MoRALayer(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.dim = dim
        
        self.module_type = org_module.__class__.__name__
        
        
        if org_module.__class__.__name__ == "Linear":
            self.in_dim = in_dim = org_module.in_features
            self.out_dim = out_dim = org_module.out_features
            self.new_dim = int(math.sqrt((self.in_dim + self.out_dim)*self.dim)+0.5)
            self.new_dim  = self.new_dim//2*2 # require to be even for RoPE
            
            self.mora = nn.Linear(self.new_dim, self.new_dim, bias=False)
            
            
        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{lora_name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

            if type(alpha) == torch.Tensor:
                alpha = alpha.detach().numpy()
            alpha = dim if alpha is None or alpha == 0 else alpha
            self.scale = alpha / self.dim
            self.register_buffer("alpha", torch.tensor(alpha))

            # same as microsoft's
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        x_org = x
        if self.module_type == 'Linear':
            if self.multiplier == 1.:
                sum_inter = self.in_dim // self.new_dim
                rb1 = self.in_dim//self.new_dim if self.in_dim % self.new_dim == 0 else self.in_dim//self.new_dim + 1
                if self.in_dim % self.new_dim != 0:
                    pad_size = self.new_dim - self.in_dim % self.new_dim
                    x = torch.cat([x, x[..., :pad_size]], dim=-1)
                    sum_inter += 1
                in_x = x.view(*x.shape[:-1], sum_inter, self.new_dim)
                if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
                    inv_freq = 1.0 / (10000 ** (torch.arange(0, self.new_dim, 2).float() / self.new_dim))
                    t = torch.arange(rb1)
                    freqs = torch.outer(t, inv_freq)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
                    self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
                rh_in_x = torch.cat((-in_x[..., self.new_dim//2:], in_x[..., :self.new_dim//2]), dim=-1)
                in_x = in_x*self.cos + rh_in_x*self.sin                
                
                out_x = self.mora(in_x)
                
                out_x = out_x.view(*x.shape[:-1], -1)[..., :self.out_dim]
                if out_x.shape[-1] < self.out_dim:
                    repeat_time = self.out_dim // out_x.shape[-1]
                    if self.out_dim % out_x.shape[-1] != 0:
                        repeat_time += 1
                    out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :self.out_dim]
                return self.org_forward(x_org) + out_x            

                
            else:
                return self.org_forward(x)    

        else:
            return (
                self.org_forward(x)
                + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
            )
        


class Arc_MoRALayer(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.dim = dim
        
        self.module_type = org_module.__class__.__name__
        
        
        if org_module.__class__.__name__ == "Linear":
            self.in_dim = in_dim = org_module.in_features
            self.out_dim = out_dim = org_module.out_features
            self.new_dim = int(math.sqrt((self.in_dim + self.out_dim)*self.dim)+0.5)
            self.new_dim  = self.new_dim//2*2 # require to be even for RoPE
            
            self.mora = nn.Linear(self.new_dim, self.new_dim, bias=False)
            
            
        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{lora_name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

            if type(alpha) == torch.Tensor:
                alpha = alpha.detach().numpy()
            alpha = dim if alpha is None or alpha == 0 else alpha
            self.scale = alpha / self.dim
            self.register_buffer("alpha", torch.tensor(alpha))

            # same as microsoft's
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        x_org = x
        if self.module_type == 'Linear':
            sum_inter = self.in_dim // self.new_dim
            rb1 = self.in_dim//self.new_dim if self.in_dim % self.new_dim == 0 else self.in_dim//self.new_dim + 1
            if self.in_dim % self.new_dim != 0:
                pad_size = self.new_dim - self.in_dim % self.new_dim
                x = torch.cat([x, x[..., :pad_size]], dim=-1)
                sum_inter += 1
            in_x = x.view(*x.shape[:-1], sum_inter, self.new_dim)
            if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
                inv_freq = 1.0 / (10000 ** (torch.arange(0, self.new_dim, 2).float() / self.new_dim))
                t = torch.arange(rb1)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
                self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
            rh_in_x = torch.cat((-in_x[..., self.new_dim//2:], in_x[..., :self.new_dim//2]), dim=-1)
            in_x = in_x*self.cos + rh_in_x*self.sin                

            out_x = self.mora(in_x)

            out_x = out_x.view(*x.shape[:-1], -1)[..., :self.out_dim]
            if out_x.shape[-1] < self.out_dim:
                repeat_time = self.out_dim // out_x.shape[-1]
                if self.out_dim % out_x.shape[-1] != 0:
                    repeat_time += 1
                out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :self.out_dim]
                            
            return  out_x            


        else:            
            return self.lora_up(self.lora_down(x)) * self.scale
        
class Arc_MoRALayer_LayerNorm(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.dim = dim
        
        self.module_type = org_module.__class__.__name__
        

        if org_module.__class__.__name__ == "LayerNorm":
            self.layernorm_update = deepcopy(org_module)
            self.layernorm_update.requires_grad_(True)
            self.layernorm_update.train()        
            
        elif org_module.__class__.__name__ == "Linear":
            self.in_dim = in_dim = org_module.in_features
            self.out_dim = out_dim = org_module.out_features
            self.new_dim = int(math.sqrt((self.in_dim + self.out_dim)*self.dim)+0.5)
            self.new_dim  = self.new_dim//2*2 # require to be even for RoPE
            
            self.mora = nn.Linear(self.new_dim, self.new_dim, bias=False)
            
            
        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{lora_name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

            if type(alpha) == torch.Tensor:
                alpha = alpha.detach().numpy()
            alpha = dim if alpha is None or alpha == 0 else alpha
            self.scale = alpha / self.dim
            self.register_buffer("alpha", torch.tensor(alpha))

            # same as microsoft's
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        x_org = x
        
        if self.module_type == "LayerNorm":
            return (1-self.multiplier)*self.org_forward(x) + self.multiplier*self.layernorm_update(x)  
                
        elif self.module_type == 'Linear':
            sum_inter = self.in_dim // self.new_dim
            rb1 = self.in_dim//self.new_dim if self.in_dim % self.new_dim == 0 else self.in_dim//self.new_dim + 1
            if self.in_dim % self.new_dim != 0:
                pad_size = self.new_dim - self.in_dim % self.new_dim
                x = torch.cat([x, x[..., :pad_size]], dim=-1)
                sum_inter += 1
            in_x = x.view(*x.shape[:-1], sum_inter, self.new_dim)
            if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
                inv_freq = 1.0 / (10000 ** (torch.arange(0, self.new_dim, 2).float() / self.new_dim))
                t = torch.arange(rb1)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
                self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
            rh_in_x = torch.cat((-in_x[..., self.new_dim//2:], in_x[..., :self.new_dim//2]), dim=-1)
            in_x = in_x*self.cos + rh_in_x*self.sin                

            out_x = self.mora(in_x)

            out_x = out_x.view(*x.shape[:-1], -1)[..., :self.out_dim]
            if out_x.shape[-1] < self.out_dim:
                repeat_time = self.out_dim // out_x.shape[-1]
                if self.out_dim % out_x.shape[-1] != 0:
                    repeat_time += 1
                out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :self.out_dim]
                            
            return  out_x            

        else:            
            return self.lora_up(self.lora_down(x)) * self.scale
                
        

        

class LoRALayer_w_LayerNorm(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.dim = dim

        self.module_type = org_module.__class__.__name__

        if org_module.__class__.__name__ == "LayerNorm":
            self.layernorm_update = deepcopy(org_module)
            self.layernorm_update.requires_grad_(True)
            self.layernorm_update.train()

        else:
            if org_module.__class__.__name__ == "Linear":
                in_dim = org_module.in_features
                out_dim = org_module.out_features
                self.lora_down = nn.Linear(in_dim, dim, bias=False)
                self.lora_up = nn.Linear(dim, out_dim, bias=False)


            elif org_module.__class__.__name__ == "Conv2d":
                in_dim = org_module.in_channels
                out_dim = org_module.out_channels

                self.dim = min(self.dim, in_dim, out_dim)
                if self.dim != dim:
                    print(f"{lora_name} dim (rank) is changed to: {self.dim}")

                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = nn.Conv2d(
                    in_dim, self.dim, kernel_size, stride, padding, bias=False
                )
                self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

            if type(alpha) == torch.Tensor:
                alpha = alpha.detach().numpy()
            alpha = dim if alpha is None or alpha == 0 else alpha
            self.scale = alpha / self.dim
            self.register_buffer("alpha", torch.tensor(alpha))

            # same as microsoft's
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        if self.module_type == "LayerNorm":
            return (1-self.multiplier)*self.org_forward(x) + self.multiplier*self.layernorm_update(x)  
        
        else:    
            return (
                self.org_forward(x)
                + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
            )
        




class LoRAModules(nn.Module):
    TARGET_REPLACE_MODULES = ['Linear', 'Conv2d']

    MODULE_PREFIX_UNET = "lora"   # aligning with SD webui usage

    def __init__(
        self,
        base_model,
        rank: int = 4,
        multiplier: float = 0.0,
        alpha: float = 1.0,
        module = LoRALayer,
        module_kwargs = None,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.dim = rank
        self.alpha = alpha

        self.module = module
        self.module_kwargs = module_kwargs or {}

        # lora layer for base_model
        self.lora_layers = self.create_modules(
            LoRAModules.MODULE_PREFIX_UNET,
            base_model,
            LoRAModules.TARGET_REPLACE_MODULES,
            self.dim,
            self.multiplier,
        )
        print(f"Create LoRA for MIL: {len(self.lora_layers)} modules.")

        lora_names = set()
        for lora_layer in self.lora_layers:
            assert (
                lora_layer.lora_name not in lora_names
            ), f"duplicated LoRA layer name: {lora_layer.lora_name}. {lora_names}"
            lora_names.add(lora_layer.lora_name)

        for lora_layer in self.lora_layers:
            lora_layer.apply_to()
            self.add_module(
                lora_layer.lora_name,
                lora_layer,
            )

        del base_model

        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
    ) -> list:
        lora_layers = []

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                lora_name = prefix + "." + name
                lora_name = lora_name.replace(".", "_")
                print(f"{lora_name}")
                lora_layer = self.module(
                    lora_name, module, multiplier, rank, self.alpha, **self.module_kwargs
                )
                lora_layers.append(lora_layer)

        return lora_layers

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        all_params = []

        if self.lora_layers:
            params = []
            [params.extend(lora_layer.parameters()) for lora_layer in self.lora_layers]
            param_data = {"params": params}
            if default_lr is not None:
                param_data["lr"] = default_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        for key in list(state_dict.keys()):
            if not key.startswith("lora"):
                del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def __enter__(self):
        for lora_layer in self.lora_layers:
            lora_layer.multiplier = 1.0

    def __exit__(self, exc_type, exc_value, tb):
        for lora_layer in self.lora_layers:
            lora_layer.multiplier = 0
            
            
            
class LoRAModules_w_LayerNorm(nn.Module):
    TARGET_REPLACE_MODULES = ['Linear', 'Conv2d', 'LayerNorm']

    MODULE_PREFIX_UNET = "lora"   # aligning with SD webui usage

    def __init__(
        self,
        base_model,
        rank: int = 4,
        multiplier: float = 0.0,
        alpha: float = 1.0,
        module = LoRALayer,
        module_kwargs = None,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.dim = rank
        self.alpha = alpha

        self.module = module
        self.module_kwargs = module_kwargs or {}

        # lora layer for base_model
        self.lora_layers = self.create_modules(
            LoRAModules_w_LayerNorm.MODULE_PREFIX_UNET,
            base_model,
            LoRAModules_w_LayerNorm.TARGET_REPLACE_MODULES,
            self.dim,
            self.multiplier,
        )
        print(f"Create LoRA for MIL: {len(self.lora_layers)} modules.")

        lora_names = set()
        for lora_layer in self.lora_layers:
            assert (
                lora_layer.lora_name not in lora_names
            ), f"duplicated LoRA layer name: {lora_layer.lora_name}. {lora_names}"
            lora_names.add(lora_layer.lora_name)

        for lora_layer in self.lora_layers:
            lora_layer.apply_to()
            self.add_module(
                lora_layer.lora_name,
                lora_layer,
            )

        del base_model

        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
    ) -> list:
        lora_layers = []

        for name, module in root_module.named_modules():            
            if module.__class__.__name__ in target_replace_modules:
                lora_name = prefix + "." + name
                lora_name = lora_name.replace(".", "_")
                print(f"{lora_name}")
                lora_layer = self.module(
                    lora_name, module, multiplier, rank, self.alpha, **self.module_kwargs
                )
                lora_layers.append(lora_layer)

        return lora_layers

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        all_params = []

        if self.lora_layers:
            params = []
            [params.extend(lora_layer.parameters()) for lora_layer in self.lora_layers]
            param_data = {"params": params}
            if default_lr is not None:
                param_data["lr"] = default_lr
            all_params.append(param_data)
        
        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        for key in list(state_dict.keys()):
            if not key.startswith("lora"):
                del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def __enter__(self):
        for lora_layer in self.lora_layers:
            lora_layer.multiplier = 1.0

    def __exit__(self, exc_type, exc_value, tb):
        for lora_layer in self.lora_layers:
            lora_layer.multiplier = 0
            




class Arc_Demaging_MoRALayer_LayerNorm(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.dim = dim
        
        self.module_type = org_module.__class__.__name__
        self.prune_flag = False

        if org_module.__class__.__name__ == "LayerNorm":
            self.layernorm_update = deepcopy(org_module)
            self.layernorm_update.requires_grad_(True)
            self.layernorm_update.train()        
            
            self.register_buffer("prune_mask", torch.ones(list(self.layernorm_update.weight.shape)))
            self.register_buffer("prune_rank", 1e3*torch.ones(list(self.layernorm_update.weight.shape)))
            
            
        elif org_module.__class__.__name__ == "Linear":
            self.in_dim = in_dim = org_module.in_features
            self.out_dim = out_dim = org_module.out_features
            self.new_dim = int(math.sqrt((self.in_dim + self.out_dim)*self.dim)+0.5)
            self.new_dim  = self.new_dim//2*2 # require to be even for RoPE
            
            self.mora = nn.Linear(self.new_dim, self.new_dim, bias=False)
            # self.prune_mask = torch.ones(list(self.mora.weight.shape))    
            # self.prune_rank = torch.zeros(list(self.mora.weight.shape))    
            
            self.register_buffer("prune_mask", torch.ones(list(self.mora.weight.shape)))
            self.register_buffer("prune_rank", 1e3*torch.ones(list(self.mora.weight.shape)))
            
                        
            
            
        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{lora_name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)
                        
            # self.prune_mask_down = torch.ones(list(self.lora_down.weight.shape))  
            # self.prune_rank_down = torch.zeros(list(self.lora_down.weight.shape))    
            
            # self.prune_mask_up = torch.ones(list(self.lora_up.weight.shape))            
            # self.prune_rank_up = torch.zeros(list(self.lora_up.weight.shape))    

            self.register_buffer("prune_mask_down", torch.ones(list(self.lora_down.weight.shape)))
            self.register_buffer("prune_rank_down", 1e3*torch.ones(list(self.lora_down.weight.shape)))

            self.register_buffer("prune_mask_up", torch.ones(list(self.lora_up.weight.shape)))
            self.register_buffer("prune_rank_up", 1e3*torch.ones(list(self.lora_up.weight.shape)))

            
                        
            if type(alpha) == torch.Tensor:
                alpha = alpha.detach().numpy()
            alpha = dim if alpha is None or alpha == 0 else alpha
            self.scale = alpha / self.dim
            self.register_buffer("alpha", torch.tensor(alpha))

            # same as microsoft's
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def set_prune_flag(self, flag):
        self.prune_flag = flag        


    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        x_org = x
        
        if self.module_type == "LayerNorm":
            if self.prune_flag:
                weight = self.layernorm_update.weight * self.prune_mask
                return F.layer_norm(x, self.layernorm_update.normalized_shape, weight, self.layernorm_update.bias, self.layernorm_update.eps)

            else:
                return self.layernorm_update(x)  
                
        elif self.module_type == 'Linear':
            sum_inter = self.in_dim // self.new_dim
            rb1 = self.in_dim//self.new_dim if self.in_dim % self.new_dim == 0 else self.in_dim//self.new_dim + 1
            if self.in_dim % self.new_dim != 0:
                pad_size = self.new_dim - self.in_dim % self.new_dim
                x = torch.cat([x, x[..., :pad_size]], dim=-1)
                sum_inter += 1
            in_x = x.view(*x.shape[:-1], sum_inter, self.new_dim)
            if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
                inv_freq = 1.0 / (10000 ** (torch.arange(0, self.new_dim, 2).float() / self.new_dim))
                t = torch.arange(rb1)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
                self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
            rh_in_x = torch.cat((-in_x[..., self.new_dim//2:], in_x[..., :self.new_dim//2]), dim=-1)
            in_x = in_x*self.cos + rh_in_x*self.sin                

            #######################################################
            ###################### Pruning ########################
            #######################################################
            if self.prune_flag:
                weight = self.mora.weight * self.prune_mask
                out_x = F.linear(in_x, weight)
            else:
                out_x = self.mora(in_x)
            #######################################################
            ###################### Pruning ########################
            #######################################################

            out_x = out_x.view(*x.shape[:-1], -1)[..., :self.out_dim]
            if out_x.shape[-1] < self.out_dim:
                repeat_time = self.out_dim // out_x.shape[-1]
                if self.out_dim % out_x.shape[-1] != 0:
                    repeat_time += 1
                out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :self.out_dim]
                            
            return  out_x            

        else:
            if self.prune_flag:
                # breakpoint()
                down_weight = self.lora_down.weight * self.prune_mask_down
                out_lora_down = self.lora_down._conv_forward(x, down_weight)
            
                up_weight = self.lora_up.weight * self.prune_mask_up
                out_lora_up = self.lora_up._conv_forward(out_lora_down, up_weight)         
                return out_lora_up * self.scale
            
            else:
                return self.lora_up(self.lora_down(x)) * self.scale
        


  
class Damaging_LoRAModules_w_LayerNorm(LoRAModules_w_LayerNorm):
    def set_rank_for_pruning(self):
        for lora_layer in self.lora_layers:
            if lora_layer.module_type == "Linear":
                _, indices = torch.sort(lora_layer.mora.weight.view(-1))
                rank_matrix = torch.zeros_like(lora_layer.mora.weight.view(-1)).to(lora_layer.mora.weight.device)
                rank_matrix[indices] = torch.arange(1, len(indices)+1).float().to(lora_layer.mora.weight.device)
                rank_matrix = rank_matrix.view_as(lora_layer.mora.weight)
                
                lora_layer.prune_rank = rank_matrix
            
            elif lora_layer.module_type == "LayerNorm":
                weight = lora_layer.layernorm_update.weight.data

                _, indices = torch.sort(weight.view(-1).view(-1))
                rank_matrix = torch.zeros_like(weight).to(lora_layer.layernorm_update.weight.device)
                rank_matrix[indices] = torch.arange(1, len(indices) + 1).float().to(lora_layer.layernorm_update.weight.device)
                
                lora_layer.prune_rank = rank_matrix
                
            elif lora_layer.module_type == "Conv2d":
                _, indices = torch.sort(lora_layer.lora_down.weight.view(-1))
                rank_matrix = torch.zeros_like(lora_layer.lora_down.weight.view(-1)).to(lora_layer.lora_down.weight.device)
                rank_matrix[indices] = torch.arange(1, len(indices)+1).float().to(lora_layer.lora_down.weight.device)
                rank_matrix = rank_matrix.view_as(lora_layer.lora_down.weight)
                lora_layer.prune_rank_down = rank_matrix

                _, indices = torch.sort(lora_layer.lora_up.weight.view(-1))
                rank_matrix = torch.zeros_like(lora_layer.lora_up.weight.view(-1)).to(lora_layer.lora_up.weight.device)
                rank_matrix[indices] = torch.arange(1, len(indices)+1).float().to(lora_layer.lora_up.weight.device)
                rank_matrix = rank_matrix.view_as(lora_layer.lora_up.weight)
                lora_layer.prune_rank_up = rank_matrix

    def set_magnitude_prune(self, magnitudePruneFraction):   
        self.reset()
                 
        for lora_layer in self.lora_layers:
            if lora_layer.module_type == "Linear":
                num_rank =  torch.ceil( magnitudePruneFraction*len(lora_layer.mora.weight.view(-1)))
                lora_layer.prune_mask[lora_layer.prune_rank <= num_rank] = 0
   
            elif lora_layer.module_type == "LayerNorm":
                num_rank =  torch.ceil( magnitudePruneFraction*len(lora_layer.layernorm_update.weight.view(-1)))
                lora_layer.prune_mask[lora_layer.prune_rank <= num_rank] = 0
                   
            elif lora_layer.module_type == "Conv2d":
                num_rank_down =  torch.ceil( magnitudePruneFraction*len(lora_layer.lora_down.weight.view(-1)))
                lora_layer.prune_mask_down[lora_layer.prune_rank_down <= num_rank_down] = 0

                num_rank_up =  torch.ceil( magnitudePruneFraction*len(lora_layer.lora_up.weight.view(-1)))
                lora_layer.prune_mask_up[lora_layer.prune_rank_up <= num_rank_up] = 0
                
    def reset(self):
        for lora_layer in self.lora_layers:
            if lora_layer.module_type == "Linear" or lora_layer.module_type == "LayerNorm":
                lora_layer.prune_mask[True] = 1.
   
            elif lora_layer.module_type == "Conv2d":
                lora_layer.prune_mask_down[True] = 1.
                lora_layer.prune_mask_up[True] = 1.
                
    def set_prune_flag(self, flag):
        for lora_layer in self.lora_layers:
            lora_layer.set_prune_flag(flag)