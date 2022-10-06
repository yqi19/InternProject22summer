# from statistics import linear_regression
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

# window size for features
window_size_1 = [(7,7),(7,7),(7,7),(7,7)]
# window size for cluster centers
window_size_2 = [(2,2),(2,2),(2,2),(2,2)]


# input:  
#   padded_img: N, H, W, C
#   ksize_h, ksize_w
# output: 
#   col: N*out_h*out_w, ksize_h*ksize_w, C
def im2col(img, ksize_h, ksize_w, stride=1):
    N, H, W, C = img.shape
    out_h = (H - ksize_h) // stride + 1
    out_w = (W - ksize_w) // stride + 1
    col = torch.empty((N * out_h * out_w, ksize_h * ksize_w * C))

    outsize = out_w * out_h
    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + ksize_h
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + ksize_w
            col[y_start+x::outsize, :] = img[:, y_min:y_max, x_min:x_max, :].reshape(N, -1)

    col = col.reshape(N*out_h*out_w, ksize_h*ksize_w, C)
    return col

def window_partition_without_overlap(x, window_size):
    """
    Args:
        x: tensor (B,H,W,C)
        window_size(int): window size

    Returns:
        windows: tensor (num_windows*B, window_size, window_size, C)
    """
    B,H,W,C = x.shape
    x = x.view(B,H//window_size, window_size, W//window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows    

def window_reverse_without_overlap(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition_with_overlap(cluster_centers, window_size_h, window_size_w):
    """
    Args:
        cluster_centers: tensor (B, num_tensors, C)
        window_size_h 
        window_size_w 

    Returns:
        cluster_centers: tensor ((num_tensors*B, window_size_h * window_size_w, C)
    """
    B, num_tensors,C = cluster_centers.shape
    assert num_tensors == 7*7, "num_tensors are not correct numbers"

    cluster_centers = cluster_centers.view(B,7,7,C)
    # print(cluster_centers.shape)
    cluster_centers = cluster_centers.permute(0,3,1,2)
    # cluster_centers.shape = (B,C,H,W)

    # padding process
    cluster_centers = torch._C._nn.reflection_pad2d(cluster_centers, (0, 1, 0, 1))
    
    cluster_centers = cluster_centers.permute(0,2,3,1)
    # cluster_centers.shape = (B,H,W,C)
    new_cluster_centers = im2col(cluster_centers,2,2,1)
    # new_cluster_centers.shape = (B*49, ksize_h*ksize_w, C)

    return new_cluster_centers.cuda()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LocalCrossAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size_1 (tuple[int]): The height and width of the window_1.
        window_size_2 (tuple[int]): The height and width of the window_2
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim{dim}  should be divided by num_heads {num_heads}."

        self.dim = dim 
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        # self.sr_ratio = sr_ratio

        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # input:
    # x.size() = num_windows*B, window_1_w * window_1_h, C
    #
    # cluster.size() = num_windows*B, window_2_w * window_2_h, C
    #
    # output : B, N, C
    def forward(self, x, H, W, cluster_centers, stage):

        (window_1_h, window_1_w) = window_size_1[stage-1]
        (window_2_h, window_2_w) = window_size_2[stage-1]

        B1, N1, C = x.shape
        B2, N2, C = cluster_centers.shape

        # assert N1==window_1_h*window_1_w, f"input features x have wrong size"
        # assert N2==window_2_h*window_2_w, f"input cluster centers have wrong size"
        assert B1==B2, f"wrong number of input tensors"

        q = self.q(x).reshape(B1, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B1, self.num_heads, N1, C//self.num_heads]
    
        kv = self.kv(cluster_centers).reshape(B2,-1,2,self.num_heads, C //self.num_heads).permute(2,0,3,1,4)
        # [2, B2, self.num_heads, -1, C//self.num_heads]
        k, v = kv[0], kv[1]
   
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MultiHeadSelfAttention(nn.Module):
    """
    do multi-head-self-attention inside of poolinged features
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.linear = linear
        # self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        """
        self.pool1 = nn.AdaptiveAvgPool2d(7)
        self.pool2 = nn.AdaptiveAvgPool2d(7)
        self.pool3 = nn.AdaptiveAvgPool2d(7)
        self.pool4 = nn.AdaptiveAvgPool2d(7)
        
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, stage):
        B, N, C = x.shape
        assert N == H*W, "input feature has wrong size"

        """
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x
        """
        if stage == 1 or stage == 2 or stage == 3:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x

        qkv = self.qkv(x_).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q,k,v = qkv[0],qkv[1],qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        """
        x = self.proj(x)
        x = self.proj_drop(x)
        """
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.multiheadselfattn = MultiHeadSelfAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.crosswindowattn = LocalCrossAttention(            
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, stage):

        if stage == 1:
            window_size = 8
        elif stage == 2:
            window_size = 4
        elif stage == 3:
            window_size = 2
        elif stage == 4:
            window_size = 1

        (B,N,C) = x.shape

        assert N==H*W, "wrong size of input tensors"

        shortcut = x
        x = self.norm1(x)
        cluster_centers = self.multiheadselfattn(x,H,W,stage)
        # cluster_centers.size() == (B, num_tensors, C)

        y = x.reshape(-1, H, W,C)
        # print(y.shape)
        y_windows = window_partition_without_overlap(y,window_size)
        # print(y_windows.shape)
        x_windows = y_windows.reshape(-1,window_size*window_size,C)
        # print(x_windows.shape)
        # x_windows.size() == (num_windows*B, window_size_1[0]* window_size_1[1], C)

        new_cluster_centers = cluster_centers.reshape(-1,1,C)
        # old :new_cluster_centers = window_partition_with_overlap(cluster_centers,2,2)
        # print(new_cluster_centers.shape)
        # new_cluster_centers.size() == (num_windows*B, window_size_2[0]*window_size_2[1], C)

        new_x_windows = self.crosswindowattn(x_windows, H, W, new_cluster_centers, stage)
        # print(new_x_windows.shape)
        new_x_windows_ = new_x_windows.reshape(-1,window_size, window_size, C)
        x_ = window_reverse_without_overlap(new_x_windows_, window_size, H,W)
        x_ = x_.view(B,N,C)
        
        x_ = shortcut + self.drop_path(x_)
        # x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x_ = x_ + self.drop_path(self.mlp(self.norm2(x_), H, W))

        return x_

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class OurVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        # depth is the corrsponding
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W, i+1)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def our_window_new_b0(pretrained=False, **kwargs):
    model = OurVisionTransformer(
        patch_size=4, embed_dims=[32,64,160,256], num_heads=[1,2,5,8], mlp_ratios=[8,8,4,4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths = [2,2,2,2], sr_ratios=[8,4,2,1],
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def our_window_new_b1(pretrained=False, **kwargs):
    model = OurVisionTransformer(
        patch_size=4, embed_dims=[64,128,320,512], num_heads=[1,2,5,8], mlp_ratios=[8,8,4,4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths = [2,2,2,2], sr_ratios=[8,4,2,1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def our_window_new_b2(pretrained=False, **kwargs):
    model = OurVisionTransformer(
        patch_size=4, embed_dims=[64,128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def our_window_new_b3(pretrained=False, **kwargs):
    model = OurVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def our_window_new_b4(pretrained=False, **kwargs):
    model = OurVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def our_window_new_b5(pretrained=False, **kwargs):
    model = OurVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model
