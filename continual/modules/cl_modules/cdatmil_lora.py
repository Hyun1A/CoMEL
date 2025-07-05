
from functools import partial
from torch import nn
from .emb_position import *
from .datten import *
from .rmsa import *
from .nystrom_attention import NystromAttention
from .datten import DAttention
from timm.models.layers import DropPath
from .lora import LoRALinearLayer, LoRACompatibleIncrementalLinear



def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





class CDATAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.region_size = agent_num
        self.region_num = None

        self.min_region_num = 10
        self.min_region_ratio = 0.0001


        self.an_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.na_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)

        pool_size = int(agent_num )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        self.apply(initialize_weights)





    def padding(self,x):
        B, L, C = x.shape
        
        # breakpoint()
        
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H+_n, W+_n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H+_n, W+_n
            region_size = int(H // self.region_num)
            region_num = self.region_num
        
        add_length = H * W - L
        
        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H

        
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        
        return x,H,W,add_length,region_num,region_size


    def region_partition(self, x, region_size):
        """
        Args:
            x: (B, H, W, C)
            region_size (int): region size
        Returns:
            regions: (num_regions*B, region_size, region_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
        regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
        return regions


    def forward(self, x, cluster_idx=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        
        b, n, c = x.shape

        x,H,W,add_length,region_num,region_size = self.padding(x)
        x = x.view(b, H, W, c)
        x_regions = self.region_partition(x, region_size)  # nW*B, region_size, region_size, C
        x_regions = x_regions.view(-1, region_size * region_size, c).mean(dim=0,keepdim=True)  # nW*B, region_size*region_size, C

        x = x.reshape(b,H*W,c)
        b, n, c = x.shape

        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        agent_num = region_size**2
        qkv_regions = self.qkv(x_regions).reshape(b, agent_num, 3, c).permute(2, 0, 1, 3)
        agent_tokens, agent_keys = qkv_regions[0], qkv_regions[1]

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        agent_tokens = agent_tokens.reshape(b, agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_keys = agent_keys.reshape(b, agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias = self.an_bias.repeat(b,1,agent_num,x.size(1))
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias = self.na_bias.repeat(b,1,x.size(1),agent_num)
        q_attn = self.softmax((q * self.scale) @ agent_keys.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v.transpose(1, 2).reshape(b, n, c)
        x[:, :, :] = x[:, :, :] + 0.1*v_

        if add_length >0:
            x = x[:,:-add_length]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CDATBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 agent_num=49, window=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CDATAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                   agent_num=agent_num)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(initialize_weights)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class CDATMIL(nn.Module):
    def __init__(self, input_dim=1024,mlp_dim=512,act='relu',n_classes=2,dropout=0.25,pos_pos=0,pos='none',\
                peg_k=7,attn='rmsa',pool='attn',region_num=8,n_layers=2,n_heads=8,agent_num=128,drop_path=0.,da_act='relu',\
                trans_dropout=0.1,ffn=False,ffn_act='gelu',mlp_ratio=4.,da_gated=False,da_bias=False,da_dropout=False,\
                trans_dim=64,epeg=True,min_region_num=0,qkv_bias=True, norm_layer=None, act_layer=None,**kwargs):
        
        super(CDATMIL, self).__init__()

        self.mlp_dim = mlp_dim
        self.patch_to_emb = [nn.Linear(input_dim, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)


        self.region_size = 16
        self.region_num = None
        self.rank = 16
        self.eps_singular = 0.99


        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.agent_blocks = nn.ModuleList([
                CDATBlock(
                    dim=self.mlp_dim, num_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=trans_dropout,
                    attn_drop=0.0, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
                    agent_num=self.region_size) for i in range(n_layers)]
                )


        self.pool_fn = DAttention(self.mlp_dim,da_act,gated=da_gated,bias=da_bias,dropout=da_dropout) if pool == 'attn' else nn.AdaptiveAvgPool1d(1)
        
        
        self.classifier = nn.ModuleList([
            nn.Linear(self.mlp_dim, n_classes),
        ])

        self.apply(initialize_weights)


        self.lora_dim = self.rank
        self.num_expanded = 0

        self.attach_pairs = dict()


        self.update_modules = dict()
        

        for name, module in self.named_modules():
            for child_name, child_module in module.named_children():
                if ("classifier" not in name) and (child_module.__class__.__name__ == "Linear") and (child_module.out_features > 1):
                    full_name=f"{name}.{child_name}"

                    self.update_modules[full_name] = child_module

                    in_features = child_module.in_features
                    out_features = child_module.out_features
                    is_bias = (child_module.bias is not None)
                    
                    org_weight = child_module.weight.data
                    if is_bias:
                        org_bias = child_module.bias.data

                    lora_attach_linear = LoRACompatibleIncrementalLinear(
                                                            in_features=in_features,\
                                                            out_features=out_features,\
                                                            bias=is_bias)

                    lora_attach_linear.weight.data = org_weight
                    if is_bias:
                        lora_attach_linear.bias.data = org_bias

                    self.attach_pairs[full_name] = (module, child_name, lora_attach_linear)


        self.update_modules = dict()
        for idx, (full_name, pair) in enumerate(self.attach_pairs.items()):
            setattr(*pair)
            self.update_modules[full_name] = pair[-1]



    def expand_lora(self):
        for idx, (full_name, module) in enumerate(self.update_modules.items()):
            in_features = module.in_features
            out_features = module.out_features

            rank = 3*self.lora_dim if "qkv" in f"{full_name}" else self.lora_dim

            lora_module = LoRALinearLayer(in_features=in_features,\
                                        out_features=out_features,\
                                        rank=rank).to(module.weight.device)

            module.add_lora(lora_module)

        self.num_expanded += 1




    def forward_aggregator(self, x, return_attn=False,no_norm=False):
        x = self.patch_to_emb(x) # n*512
        x = self.dp(x)
        

        for block in self.agent_blocks:
            x = block(x)


        # feature aggregation
        a = None
        if return_attn:
            x_rep,a = self.pool_fn(x,return_attn=True,no_norm=no_norm)
        else:
            x_rep = self.pool_fn(x)

        return x, x_rep, a


    def return_results(self, return_attn, return_patch_logits, logits, patch_logits, a):
        if return_attn:
            if return_patch_logits:
                return logits, a, patch_logits.squeeze(0)
            else:
                return logits, a
            
        else:
            if return_patch_logits:
                return logits,patch_logits.squeeze(0)
            else:
                return logits


    def forward(self, x, return_attn=False,no_norm=False, return_inst=False):
        x_inst, x, a = self.forward_aggregator(x=x, 
                                              return_attn=return_attn,
                                              no_norm=no_norm)


        Y_prob = []
        for linear_m in self.classifier:
            logits = linear_m(x)
            Y_prob.append(logits)
        logits = torch.cat(Y_prob, dim=-1)
        
    
        if not return_inst:
            if return_attn:
                return logits,a
            else:
                return logits
        else:
            if return_attn:
                return logits,a,x,x_inst
            else:
                return logits,x,x_inst


