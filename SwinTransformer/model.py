import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_c=3,
                 embed_dim=96,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        self.patch_size = patch_size
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 如果图片的H,W不是patch_size的整数倍，需要padding
        _, _, h, w = x.shape
        if (h % self.patch_size != 0) or (w % self.patch_size != 0):
            x = F.pad(x, (0, self.patch_size - w % self.patch_size,
                          0, self.patch_size - h % self.patch_size,
                          0, 0))

        x = self.proj(x)
        _, _, h, w = x.shape

        # (b,c,h,w) -> (b,c,hw) -> (b,hw,c)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, h, w



class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x, h, w):
        # (b,hw,c)
        b, l, c = x.shape
        # (b,hw,c) -> (b,h,w,c)
        x = x.view(b, h, w, c)

        # 如果h,w不是2的整数倍，需要padding
        if (h % 2 == 1) or (w % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

        # (b,h/2,w/2,c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        # (b,h/2,w/2,c)*4 -> (b,h/2,w/2,4c)
        x = torch.cat([x0, x1, x2, x3], -1)
        # (b,hw/4,4c)
        x = x.view(b, -1, 4*c)

        x = self.norm(x)
        # (b,hw/4,4c) -> (b,hw/4,2c)
        x = self.reduction(x)

        return x


def window_partition(x, window_size):
    """
    将feature map按照window_size分割成windows
    """
    b, h, w, c = x.shape
    # (b,h,w,c) -> (b,h//m,m,w//m,m,c)
    x = x.view(b, h//window_size, window_size, w//window_size, window_size, c)
    # (b,h//m,m,w//m,m,c) -> (b,h//m,w//m,m,m,c)
    # -> (b,h//m*w//m,m,m,c) -> (b*n_windows,m,m,c)
    windows = (x
               .permute(0, 1, 3, 2, 4, 5)
               .contiguous()
               .view(-1, window_size, window_size, c))
    return windows

def window_reverse(x,window_size,h,w):
    """
    将分割后的windows还原成feature map
    """

    b = int(x.shape[0] / (h*w/window_size/window_size))
    # (b,h//m,w//m,m,m,c)
    x = x.view(b,h//window_size,w//window_size,window_size,window_size,-1)
    # (b,h//m,w//m,m,m,c) -> (b,h//m,m,w//m,m,c)
    # -> (b,h,w,c)
    x = x.permute(0,1,3,2,4,5).contiguous().view(b,h,w,-1)

    return x


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hid_features=None,
                 out_features=None,
                 dropout=0.):
        super(MLP, self).__init__()

        out_features = out_features or in_features
        hid_features = hid_features or in_features

        self.fc1 = nn.Linear(in_features, hid_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_features, out_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class WindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 window_size,
                 n_heads,
                 qkv_bias=True,
                 attn_dropout=0.,
                 proj_dropout=0.):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -.5

        # ((2m-1)*(2m-1),n_heads)
        # 相对位置参数表长为(2m-1)*(2m-1)
        # 行索引和列索引各有2m-1种可能，故其排列组合有(2m-1)*(2m-1)种可能
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size - 1) * (2*window_size - 1),
                        n_heads))

        # 构建窗口的绝对位置索引
        # 以window_size=2为例
        # coord_h = coord_w = [0,1]
        # meshgrid([0,1], [0,1])
        # -> [[0,0], [[0,1]
        #     [1,1]], [0,1]]
        # -> [[0,0,1,1],
        #     [0,1,0,1]]
        # (m,)
        coord_h = torch.arange(self.window_size)
        coord_w = torch.arange(self.window_size)
        # (m,)*2 -> (m,m)*2 -> (2,m,m)
        coords = torch.stack(torch.meshgrid([coord_h, coord_w],indexing='ij'))
        # (2,m*m)
        coord_flatten = torch.flatten(coords, 1)

        # 构建窗口的相对位置索引
        # (2,m*m,1) - (2,1,m*m) -> (2,m*m,m*m)
        # 以coord_flatten为
        # [[0,0,1,1]
        #  [0,1,0,1]]为例
        # 对于第一个元素[0,0,1,1]
        # [[0],[0],[1],[1]] - [[0,0,1,1]]
        # -> [[0,0,0,0] - [[0,0,1,1] = [[0,0,-1,-1]
        #     [0,0,0,0]    [0,0,1,1]    [0,0,-1,-1]
        #     [1,1,1,1]    [0,0,1,1]    [1,1, 0, 0]
        #     [1,1,1,1]]   [0,0,1,1]]   [1,1, 0, 0]]
        # 相当于每个元素的h减去每个元素的h
        # 例如，第一行[0,0,0,0] - [0,0,1,1] -> [0,0,-1,-1]
        # 即为元素(0,0)相对(0,0)(0,1)(1,0)(1,1)为列(h)方向的差
        # 第二个元素即为每个元素的w减去每个元素的w
        # 于是得到窗口内每个元素相对每个元素高和宽的差
        # 例如relative_coords[0,1,2]
        # 即为窗口的第1个像素(0,1)和第2个像素(1,0)在列(h)方向的差
        relative_coords = coord_flatten[:, :, None] - coord_flatten[:, None, :]
        # (m*m,m*m,2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # 论文中提到的，将二维相对位置索引转为一维的过程
        # 1. 行列都加上m-1
        # 2. 行乘以2m-1
        # 3. 行列相加
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        # (m*m,m*m,2) -> (m*m,m*m)
        relative_pos_idx = relative_coords.sum(-1)
        self.register_buffer('relative_pos_idx', relative_pos_idx)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        # (b*n_windows,m*m,total_embed_dim)
        # -> (b*n_windows,m*m,3*total_embed_dim)
        # -> (b*n_windows,m*m,3,n_heads,embed_dim_per_head)
        # -> (3,b*n_windows,n_heads,m*m,embed_dim_per_head)
        qkv = (self.qkv(x)
               .reshape(b, n, 3, self.n_heads, c//self.n_heads)
               .permute(2, 0, 3, 1, 4))
        # (b*n_windows,n_heads,m*m,embed_dim_per_head)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        # (b*n_windows,n_heads,m*m,m*m)
        attn = (q @ k.transpose(-2, -1))

        # (m*m*m*m,n_heads)
        # -> (m*m,m*m,n_heads)
        # -> (n_heads,m*m,m*m)
        # -> (b*n_windows,n_heads,m*m,m*m) + (1,n_heads,m*m,m*m)
        # -> (b*n_windows,n_heads,m*m,m*m)
        relative_pos_bias = (self.relative_position_bias_table[self.relative_pos_idx.view(-1)]
                             .view(self.window_size*self.window_size, self.window_size*self.window_size, -1))
        relative_pos_bias = relative_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_pos_bias.unsqueeze(0)

        if mask is not None:
            # mask: (n_windows,m*m,m*m)
            nw = mask.shape[0]
            # (b*n_windows,n_heads,m*m,m*m)
            # -> (b,n_windows,n_heads,m*m,m*m)
            # + (1,n_windows,1,m*m,m*m)
            # -> (b,n_windows,n_heads,m*m,m*m)
            attn = (attn.view(b//nw, nw, self.n_heads, n, n)
                    + mask.unsqueeze(1).unsqueeze(0))
            # (b,n_windows,n_heads,m*m,m*m)
            # -> (b*n_windows,n_heads,m*m,m*m)
            attn = attn.view(-1, self.n_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)

        # (b*n_windows,n_heads,m*m,embed_dim_per_head)
        # -> (b*n_windows,m*m,n_heads,embed_dim_per_head)
        # -> (b*n_windows,m*m,total_embed_dim)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x



class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 n_heads,
                 window_size,
                 mlp_ratio=4,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None):
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        # 窗口向右和向下的移动数为窗口宽度除以2向下取整
        self.shift_size = window_size // 2

        # 按照每个Stage的深度堆叠若干Block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, n_heads, window_size,
                                 0 if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio, qkv_bias, proj_dropout, attn_dropout,
                                 dropout[i] if isinstance(
                                     dropout, list) else dropout,
                                 norm_layer)
            for i in range(depth)])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, h, w):
        attn_mask = self.create_mask(x, h, w)
        for blk in self.blocks:
            blk.h, blk.w = h, w
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x, h, w)
            # 如果是奇数，相当于做padding后除以2
            # 如果是偶数，相当于直接除以2
            h, w = (h+1) // 2, (w+1) // 2

        return x, h, w

    def create_mask(self, x, h, w):
        # 保证hp,wp是window_size的整数倍
        hp = int(np.ceil(h/self.window_size)) * self.window_size
        wp = int(np.ceil(w/self.window_size)) * self.window_size

        # (1,hp,wp,1)
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)

        # 将feature map分割成9个区域
        # 例如，对于9x9图片
        # 若window_size=3, shift_size=3//2=1
        # 得到slices为([0,-3],[-3,-1],[-1,])
        # 于是h从0至-4(不到-3)，w从0至-4
        # 即(0,0)(-4,-4)围成的矩形为第1个区域
        # h从0至-4，w从-3至-2
        # 即(0,-3)(-4,-2)围成的矩形为第2个区域...
        # h\w 0 1 2 3 4 5 6 7 8
        # --+--------------------
        # 0 | 0 0 0 0 0 0 1 1 2
        # 1 | 0 0 0 0 0 0 1 1 2
        # 2 | 0 0 0 0 0 0 1 1 2
        # 3 | 0 0 0 0 0 0 1 1 2
        # 4 | 0 0 0 0 0 0 1 1 2
        # 5 | 0 0 0 0 0 0 1 1 2
        # 6 | 3 3 3 3 3 3 4 4 5
        # 7 | 3 3 3 3 3 3 4 4 5
        # 8 | 6 6 6 6 6 6 7 7 8
        # 这样在每个窗口内，相同数字的区域都是连续的
        slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # (1,hp,wp,1) -> (n_windows,m,m,1) m表示window_size
        mask_windows = window_partition(img_mask, self.window_size)
        # (n_windows,m,m,1) -> (n_windows,m*m)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)

        # (n_windows,1,m*m) - (n_windows,m*m,1)
        # -> (n_windows,m*m,m*m)
        # 以window
        # [[4 4 5]
        #  [4 4 5]
        #  [7 7 8]]
        # 为例
        # 展平后为 [4,4,5,4,4,5,7,7,8]
        # [[4,4,5,4,4,5,7,7,8]] - [[4]
        #                          [4]
        #                          [5]
        #                          [4]
        #                          [4]
        #                          [5]
        #                          [7]
        #                          [7]
        #                          [8]]
        # -> [[0,0,-,0,0,-,-,-,-]
        #     [0,0,-,0,0,-,-,-,-]
        #     [...]]
        # 于是有同样数字的区域为0，不同数字的区域为非0
        # attn_mask[1,3]即为窗口的第3个元素(1,0)和第1个元素(0,1)是否相同
        # 若相同，则值为0，否则为非0
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 将非0的元素设为-100
        attn_mask = (attn_mask
                     .masked_fill(attn_mask != 0, float(-100.))
                     .masked_fill(attn_mask == 0, float(0.)))

        return attn_mask


class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size, n_heads,
                                    qkv_bias, attn_dropout, proj_dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim,
                       hid_features=dim*mlp_ratio,
                       dropout=proj_dropout)

    def forward(self, x, attn_mask):
        h, w = self.h, self.w
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # (n_windows*b,m,m,c)
        x_windows = window_partition(shifted_x, self.window_size)
        # (n_windows*b,m*m,c)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, c)

        attn_windows = self.attn(x_windows, attn_mask)

        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h*w, c)

        x = shortcut + self.dropout(x)
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x


class SwinTransformer(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_c=3,
                 n_classes=1000,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 n_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True):
        super(SwinTransformer, self).__init__()

        self.n_classes = n_classes
        self.n_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # Stage4输出的channels，即embed_dim*8
        self.n_features = int(embed_dim * 2**(self.n_layers-1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(patch_size, in_c, embed_dim,
                                      norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(proj_dropout)

        # 根据深度递增dropout
        dpr = [x.item() for x in torch.linspace(0, dropout, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            layers = BasicLayer(int(embed_dim*2**i), depths[i], n_heads[i],
                                window_size, mlp_ratio, qkv_bias,
                                proj_dropout, attn_dropout, 
                                dpr[sum(depths[:i]):sum(depths[:i+1])],
                                norm_layer, PatchMerging if i < self.n_layers-1 else None)
            self.layers.append(layers)

        self.norm = norm_layer(self.n_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.n_features, n_classes) if n_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, h, w = layer(x, h, w)

        # (b,l,c)
        x = self.norm(x)
        # (b,l,c) -> (b,c,l) -> (b,c,1)
        x = self.avgpool(x.transpose(1, 2))
        # (b,c,1) -> (b,c)
        x = torch.flatten(x, 1)
        # (b,c) -> (b,n_classes)
        x = self.head(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)






def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(embed_dim=hidden_dim, depths=layers, n_heads=heads, **kwargs)

def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(embed_dim=hidden_dim, depths=layers, n_heads=heads, **kwargs)

def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(embed_dim=hidden_dim, depths=layers, n_heads=heads, **kwargs)

def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(embed_dim=hidden_dim, depths=layers, n_heads=heads, **kwargs)

def swin_cifar(patch_size=4, in_c=3, n_classes=10, embed_dim=96, 
               depths=(2, 6, 4), n_heads=(3, 6, 12), window_size=4, mlp_ratio=4):
    """
    专为CIFAR-10/100设计的SwinTransformer变体
    Args:
        patch_size: 每个patch的大小
        in_c: 输入通道数
        n_classes: 类别数量
        embed_dim: 嵌入维度
        depths: 每个阶段的block层数
        n_heads: 每个阶段的注意力头数
        window_size: 窗口大小
        mlp_ratio: MLP的扩展比例
    """
    model = SwinTransformer(
        patch_size=patch_size,
        in_c=in_c,
        n_classes=n_classes, 
        embed_dim=embed_dim,
        depths=depths,
        n_heads=n_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        proj_dropout=0.0,
        attn_dropout=0.0,
        dropout=0.1,
        patch_norm=True
    )
    return model