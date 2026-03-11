# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn import functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


################### mask=True or 1 indicates masking (unconditional).
class SAComEncoderCLS(nn.Module):
    def __init__(self, element_names, embed_dim=256, num_heads=4, init_temp=0.5):
        super().__init__()
        self.element_num = len(element_names)
        self.embed_dim = embed_dim
        
        self.element_embed = nn.Embedding(self.element_num, embed_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Temperature parameter, shared across all compositions. Initialized with a specified value. Uses natural log (base e) for better optimization stability.
        self.log_temp = nn.Parameter(torch.tensor([init_temp]).log())
        
        self.com_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.enhance = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        
    
    def forward(self, percentages, mask, return_attn_weights=False):
        batch_size = percentages.size(0)
        elem_indices = torch.arange(self.element_num, device=percentages.device)
        
        elem_embeds = self.element_embed(elem_indices)  # (E, D)
       
        temp = self.log_temp.exp()
        scaled_percent = F.softmax(percentages / temp, dim=-1)  # (B, E)
        weighted_embeds = elem_embeds.unsqueeze(0) * scaled_percent.unsqueeze(-1)  # (1,E,D)*(B, E, 1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        extended_embeds = torch.cat([cls_tokens, weighted_embeds], dim=1)  # (B, E+1, D)
        
        # # Expanded mask (B, E+1); [CLS] position is always valid/active.
        if mask.dtype == torch.bool:
            # Boolean: True to mask, False to keep.
            valid_mask = mask  
        else:
            # Integer: 1 to mask, 0 to keep.
            valid_mask = (mask == 1)  
        key_padding_mask = torch.cat([
            # First part is all False (no masking).
            torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), 
            # Mask when mask == 1 or True.
            valid_mask.expand(-1, self.element_num)
        ], dim=1)  # (B, E+1)
        # Pre-LN 
        normed_embeds = self.norm1(extended_embeds)
        
        attn_output, attn_weights = self.com_attn(
            query=normed_embeds,  # Query (B, E+1, D)
            key=normed_embeds,              # Key (B, E+1, D)
            value=normed_embeds,             # Value (B, E+1, D)
            key_padding_mask=key_padding_mask,
            average_attn_weights=True
        )
        
        cls_attn_output = attn_output[:, :1, :]
        # Residual connection: CLS attention output + original CLS embedding.
        cls_embeds = cls_attn_output + normed_embeds[:, :1, :]  # (B, 1, D)
        #######
        # Feed-forward network (FFN) processing.
        cls_embeds = self.enhance(cls_embeds) + cls_embeds
        ########
        # Final output (B, D)
        final_embeds = cls_embeds.squeeze(1) # (B, D)
        
        if return_attn_weights:
            return final_embeds, attn_weights
        else:
            return final_embeds


class RepeatComEncoder(nn.Module):
    """Minimalist baseline composition encoder: simple percentage repetition"""
    def __init__(self, element_names, embed_dim=256, init_temp=0.5):
        super().__init__()
        self.element_num = len(element_names)
        self.embed_dim = embed_dim
        
        self.log_temp = nn.Parameter(torch.tensor([init_temp]).log())
        
        self.base_repeat = embed_dim // self.element_num
        self.remainder = embed_dim % self.element_num
        
        self.repeats = [self.base_repeat] * (self.element_num - 1)
        self.repeats.append(self.base_repeat + self.remainder)
       
        self.null_embed = nn.Parameter(torch.randn(1, embed_dim))
        

    def forward(self, percentages, mask=None, return_attn_weights=False):
        """
        Args:
            percentages: atomic fractions of elements, (B, E)
            mask: Masking Tensor, (B, E)
            return_attn_weights: Kept for backward compatibility; return value is redundant.
        """
        batch_size = percentages.size(0)
        
        temp = self.log_temp.exp()
        scaled_percent = F.softmax(percentages / temp, dim=-1)  # (B, E)
        
        expanded_percent = torch.repeat_interleave(
            scaled_percent, 
            torch.tensor(self.repeats, device=scaled_percent.device), 
            dim=1
        )
        
        null_embeds = self.null_embed.expand(batch_size, -1)  # (B, D)
        
        if mask is None:
            final_embeds = expanded_percent
        else:
            mask_expanded = mask.expand(-1, self.embed_dim)
            
            if mask.dtype == torch.bool:
                final_embeds = torch.where(mask_expanded, null_embeds, expanded_percent)
            else:
                final_embeds = torch.where(mask_expanded == 1, null_embeds, expanded_percent)
        if return_attn_weights:
            pseudo_weights = torch.ones(batch_size, self.element_num, self.element_num, 
                                       device=final_embeds.device)
            return final_embeds, pseudo_weights
        else:
            return final_embeds
        


class EmbedFC(nn.Module):
    """Fully Connected Embedding Layer"""
    def __init__(self, input_dim, hidden_dim,com_encoder=None,to_t_embed=True,t_embed_dim=256,c_embed_dim=256, t_use_pos_encoding=True,n_steps=1000):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim=hidden_dim
        self.to_t_embed=to_t_embed
        self.t_use_pos_encoding=t_use_pos_encoding
        self.n_steps=n_steps
        
        if not self.to_t_embed:
            self.label_emb=com_encoder
            layers = [
                nn.Linear(c_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)  
                ]
        else:
            self.t_embed_dim=t_embed_dim
            layers = [
                nn.Linear(self.t_embed_dim if self.t_use_pos_encoding else input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ]
        
        self.model = nn.Sequential(*layers)

    def pos_encoding(self, t):
        half = self.t_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.t_embed_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, mask=None):
        x = x.reshape(-1, self.input_dim)
        
        if self.to_t_embed:
            x = x.float()
            if self.t_use_pos_encoding:
                x=self.pos_encoding(x)
            else:
                x=x/(self.n_steps-1)
            trans_embeds=self.model(x)
            assert trans_embeds.dim() == 2, f"时间编码应返回2D，但得到{trans_embeds.dim()}D: {trans_embeds.shape}"
        else:
            x=self.label_emb(percentages=x, mask=mask).float()
            trans_embeds = self.model(x)
        return trans_embeds

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        t_embed_dim=256,
        t_use_pos_encoding=False, ###Option
        y_embed_dim=256,
        learn_sigma=True,
        com_encoder='SAComEncoderCLS',         ###Option:'RepeatComEncoder' or 'SAComEncoderCLS'
        DiT_block_shared_FFN=True    ###Option
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        if com_encoder=='SAComEncoderCLS':
            self.com_encoder=SAComEncoderCLS(["Ni", "Al", "Mo"], embed_dim=y_embed_dim)
        else:
            self.com_encoder=RepeatComEncoder(["Ni", "Al", "Mo"], embed_dim=y_embed_dim)
        
        self.t_use_pos_encoding=t_use_pos_encoding
        self.DiT_block_shared_FFN=DiT_block_shared_FFN
        if DiT_block_shared_FFN:
            self.shared_time_emb = EmbedFC(
                1, hidden_size, 
                to_t_embed=True,
                t_embed_dim=t_embed_dim,
                t_use_pos_encoding=t_use_pos_encoding
            )
            self.shared_context_emb = EmbedFC(
                3, hidden_size, 
                com_encoder=self.com_encoder,
                c_embed_dim=y_embed_dim,
                to_t_embed=False
            )
        else:
            self.timeembs_depths = nn.ModuleList([EmbedFC(1, hidden_size, to_t_embed=True,t_embed_dim=t_embed_dim,t_use_pos_encoding=self.t_use_pos_encoding) for _ in range(depth)])  
            self.contextembs_depths = nn.ModuleList([EmbedFC(3, hidden_size, com_encoder=self.com_encoder,c_embed_dim=y_embed_dim, to_t_embed=False) for _ in range(depth)])  
        self.depth=depth

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # print(f"Initializing: {module}")
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if isinstance(self.com_encoder, SAComEncoderCLS):
            print("正在初始化 SAComEncoderCLS 的权重...")
            nn.init.normal_(self.com_encoder.element_embed.weight, std=0.02)
            ########enhance initialize
            nn.init.normal_(self.com_encoder.enhance[-1].weight, std=0.02)

        
        for i in range(self.depth):
            nn.init.constant_(self.blocks[i].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.blocks[i].adaLN_modulation[-1].bias, 0)
            if not self.DiT_block_shared_FFN:
                nn.init.normal_(self.timeembs_depths[i].model[0].weight, std=0.02)
                nn.init.normal_(self.timeembs_depths[i].model[2].weight, std=0.02)
                nn.init.normal_(self.contextembs_depths[i].model[0].weight, std=0.02)
                nn.init.normal_(self.contextembs_depths[i].model[2].weight, std=0.02)
        if self.DiT_block_shared_FFN:
            nn.init.normal_(self.shared_time_emb.model[0].weight, std=0.02)
            nn.init.normal_(self.shared_time_emb.model[2].weight, std=0.02)
            nn.init.normal_(self.shared_context_emb.model[0].weight, std=0.02)
            nn.init.normal_(self.shared_context_emb.model[2].weight, std=0.02)
            
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, mask):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if not self.DiT_block_shared_FFN:
            # indenpendent FFN:calc c for each DiT block
            for i, block in enumerate(self.blocks):
                c = self.contextembs_depths[i](y, mask) + self.timeembs_depths[i](t, mask=None)
                x = block(x, c)   # (N, T, D)
        else:
            # shared-FFN
            c = self.shared_context_emb(y, mask) + self.shared_time_emb(t, mask=None)
            for i, block in enumerate(self.blocks):
                x = block(x, c)
        ###########
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    
    #################添加
    def get_masked_context(self, context, p=0.1):
        "Randomly mask out context"
        mask = torch.rand((len(context), 1), device=context.device) < p
        return mask
    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """CFG implementation. cfg_scale values >=1 are recommended."""
        if cfg_scale < 1 and cfg_scale != 0:
            print(f"Warning: cfg_scale={cfg_scale} is less than 1. This will result in interpolation (weakened condition) instead of extrapolation (strengthened condition).")
        if cfg_scale == 1:
            cond_mask = self.get_masked_context(y, p=0)
            cond_pred_out = self.forward(x, t, y, cond_mask) 
            return cond_pred_out
        cond_mask = self.get_masked_context(y, p=0) 
        uncond_mask = self.get_masked_context(y, p=1) 
        
        x_in = torch.cat([x, x], dim=0) 
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y, y], dim=0)
        mask_in = torch.cat([uncond_mask, cond_mask], dim=0) 
    
        pred_out = self.forward(x_in, t_in, y_in, mask_in) 
        
        uncond_pred_out, cond_pred_out = pred_out.chunk(2, dim=0)
        
        # Apply CFG formula (valid for all cfg_scale >= 0). Linear interpolation: uncond + w * (cond - uncond)
        out = uncond_pred_out + cfg_scale * (cond_pred_out - uncond_pred_out)
        
        return out

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

###########
def DiT_S_1(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


