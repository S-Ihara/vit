import collections
import torch
import torch.nn as nn
import numpy as np

class ImagePatchEmbedding(nn.Module):
    """
    input: image(torch.Tensor) shape=[n,c,h,w]
    output: embedding_vector(torch.Tensor) shpae=[n,p,dim]
    
    nはバッチサイズ、chwは画像のカラーチャネル、高さ、幅、pはトークン数（＝パッチの個数）
    dimは埋め込みベクトルの長さ（ハイパーパラメータ）
    """
    def __init__(self,image_size=224,patch_size=16,in_channel=3,embedding_dim=768):
        """
        Args:
            image_size (Union[int, tuple]): 画像の高さと幅
            patch_size (int): 1画像パッチのピクセル幅
            in_channel (int): 入力画像のチャネル数
            embeedding_dim (int): トークン（埋め込みベクトル）の長さ
        """
        super().__init__()
        image_size = self._pair(image_size) # if int -> tuple
        patch_size = self._pair(patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0]//patch_size[0], image_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0]*self.grid_size[1]

        self.proj = nn.Conv2d(in_channel,embedding_dim,patch_size,patch_size)
        #self.normalize = nn.LayerNorm(embedding_dim)
    
    def forward(self,x):
        """
        Args:
            x (torch.Tensor): shape=[b,c,h,w]
        Returns:
            torch.Tensor: shape[b,p,dim]
        """
        n,c,h,w = x.shape
        assert h == self.image_size[0] and w == self.image_size[1], f'Input image size ({h}*{w}) doesn\'t match model ({self.image_size[0]}*{self.image_size[1]}).'

        x = self.proj(x)
        x = x.flatten(2).transpose(1,2) # (N,C,H,W) -> (B,P,C)
        #x = self.normalize(x)
        return x

    def _pair(self,x):
        """
        util function
        return a tuple if x is int 
        """
        return x if isinstance(x, collections.abc.Iterable) else (x, x)
        #return x if isinstance(x, tuple) else (x, x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bias=True,dropout=0.,quiet_attention=False):
        """
        Args:
            dim (int): 埋め込み次元数
            num_heads (int): MultiHeadAttentionのHead数
            qkv_bias (bool): MultiHeadAttentionの埋め込み全結合層にbiasを付けるかどうか
            dropout (float): ドロップアウト確率
            quiet_attention (bool): Trueの場合、softmaxの分母に1を足す
        Note:
            quiet attentionのreference
            https://www.evanmiller.org/attention-is-off-by-one.html
        """
        super().__init__()
        
        self.quiet_attention = quiet_attention
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"The hidden size {dim} is not a multiple of the number of head attention"
        self.hidden_dim = dim
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim,dim,bias=qkv_bias)
        self.key = nn.Linear(dim,dim,bias=qkv_bias)
        self.value = nn.Linear(dim,dim,bias=qkv_bias)
        
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Dropout(p=dropout),
        )
    
    def forward(self,x):
        batch_size,num_patches,_ = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # マルチヘッドに分割
        #multihead_qkv_shape = q.size()[:-1] + (self.num_heads, self.head_dim)
        multihead_qkv_shape = torch.Size([batch_size, num_patches, self.num_heads, self.head_dim])
        qs = q.view(multihead_qkv_shape)
        qs = qs.permute(0, 2, 1, 3)
        ks = k.view(multihead_qkv_shape)
        ks = ks.permute(0, 2, 1, 3)
        ks_T = ks.transpose(2,3)
        vs = v.view(multihead_qkv_shape)
        vs = vs.permute(0, 2, 1, 3)
        
        scaled_dot_product = qs@ks_T / np.sqrt(self.head_dim) # qs @ ks_T で速度変わったりする？
        if self.quiet_attention:
            self_attention = _softmax_one(scaled_dot_product,dim=-1)
        else:
            self_attention = nn.functional.softmax(scaled_dot_product,dim=-1)
        self_attention = self.dropout(self_attention)
        
        context_layer = self_attention@vs
        #context_layer = context_layer.transpose(1,2).reshape(batch_size,num_patchs,self.hidden_dim)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().reshape(batch_size,num_patches,self.hidden_dim)
        out = self.projection(context_layer)
        
        return out

def _softmax_one(x,dim=-1):
    """ https://www.evanmiller.org/attention-is-off-by-one.html の実装
    Args:
        x (torch.Tensor):
        dim (int, optional): softmaxを取る次元. Defaults to -1.
    Returns:
        torch.Tensor: softmaxを取った後のテンソル
    """
    x = x - x.max(dim=dim, keepdim=True).values # subtract the max for stability
    exp_x = torch.exp(x)
    return exp_x / (1+exp_x.sum(dim=dim,keepdim=True))

class ViTFeedForward(nn.Module):
    def __init__(self,dim,hidden_dim=768*4,activation=nn.GELU(),dropout=0.):
        """
        Args:
            dim (int): 埋め込み次元数
            hidden_dim (int): FeedForward Networkの隠れ層次元数
            activation (torch.nn.modules.activation): pytorchの活性化関数
            dropout (float): ドロップアウト確率
        """
        super().__init__()
        self.linear1 = nn.Linear(dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,dim)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x
        

class ViTBlock(nn.Module):
    def __init__(
        self,
        dim=768,
        hidden_dim=768*4,
        num_heads=12,
        activation=nn.GELU(),
        qkv_bias=True,
        dropout=0.,
        quiet_attention=False,
    ):
        """ Transformer Encoder block
        Args:
            dim (int): 埋め込み次元数
            hidden_dim (int): FeedForward Networkの隠れ層次元数
            num_heads (int): MultiHeadAttentionのHead数
            activation (torch.nn.modules.activation): pytorchの活性化関数
            qkv_bias (bool): MultiHeadAttentionの埋め込み全結合層にbiasを付けるかどうか
            dropout (float): ドロップアウト確率
            quiet_attention (bool): Trueの場合、softmaxの分母に1を足す
        """
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim,num_heads,qkv_bias,dropout,quiet_attention)
        self.ff   = ViTFeedForward(dim,hidden_dim,activation,dropout)
        self.ln1 = nn.LayerNorm(dim,eps=1e-10)
        self.ln2 = nn.LayerNorm(dim,eps=1e-10)
        # self.ln = nn.LayerNorm(dim,eps=1e-10,elementwise_affine=False) # 非学習パラメータにする場合
    
    def forward(self,x):
        z = self.ln1(x)
        z = self.mhsa(z)
        x = x + z
        z = self.ln2(x)
        z = self.ff(x)
        out = x + z  
        
        return out

class ViTEncoder(nn.Module):
    """
    TODO:
        attentionを取り出せるように
    """
    def __init__(
        self,
        dim=768,
        hidden_dim=768*4,
        num_heads=12,
        activation=nn.GELU(),
        qkv_bias=True,
        dropout=0.,
        num_blocks=8,
        quiet_attention=False,
    ):
        """
        Args:
            dim (int): 埋め込み次元数
            hidden_dim (int): FeedForward Networkの隠れ層次元数
            num_heads (int): MultiHeadAttentionのHead数
            activation (torch.nn.modules.activation): pytorchの活性化関数
            qkv_bias (bool): MultiHeadAttentionの埋め込み全結合層にbiasを付けるかどうか
            dropout (float): ドロップアウト確率
            num_blocks (int): ブロックの数
            quiet_attention (bool): Trueの場合、softmaxの分母に1を足す
        """
        super().__init__()
        self.blocks = nn.ModuleList([ViTBlock(
            dim,hidden_dim,num_heads,activation,qkv_bias,dropout,quiet_attention
        ) for _ in range(num_blocks)])
    
    def forward(self,x):
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x
