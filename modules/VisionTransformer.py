import torch
import torch.nn as nn
import numpy as np

from .Transformer_modules import ImagePatchEmbedding, ViTEncoder, ViTBlock
from .Sinusoidal_positional_Encoding import get_2d_sincons_pos_embed

class VisionTransformer(nn.Module):
    def __init__(self,image_size=224,patch_size=16,in_channel=3,dim=768,hidden_dim=768*4,
                num_heads=12,activation=nn.GELU(),num_blocks=8,qkv_bias=True,dropout=0.,
                quiet_attention=False,num_classes=1000):
        """
        Args:
            image_size (Union[int, tuple]): 画像の高さと幅
            patch_size (int): 1画像パッチのピクセル幅
            in_channel (int): 入力画像のチャネル数
            dim (int): 埋め込み次元数
            hidden_dim (int): FeedForward Networkの隠れ層次元数
            num_heads (int): MultiHeadAttentionのHead数
            activation (torch.nn.modules.activation): pytorchの活性化関数
            num_blocks (int): ブロックの数
            qkv_bias (bool): MultiHeadAttentionの埋め込み全結合層にbiasを付けるかどうか
            dropout (float): ドロップアウト確率
            quiet_attention (bool): Trueの場合、softmaxの分母に1を足す
            num_classes (int): 分類ヘッドの数
        """
        super().__init__()
        
        # input layer
        self.patch_embedding = ImagePatchEmbedding(image_size,patch_size,in_channel,dim)
        num_patches = self.patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,dim)))
        self.positional_embedding = nn.Parameter(torch.randn(size=(1,num_patches+1,dim)))
        
        # vit encoder 
        self.encoder = ViTEncoder(dim,hidden_dim,num_heads,activation,qkv_bias,dropout,num_blocks,quiet_attention)
        
        # mlp head
        self.ln = nn.LayerNorm(dim,eps=1e-10)
        self.head = nn.Linear(dim,num_classes)
    
    def forward(self,x):
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat((cls_token,x),dim=1) # (B,num_patches+1,embedding_dim)
        x = x + self.positional_embedding
        
        x = self.encoder(x)
        
        x = torch.index_select(x,1,torch.tensor(0,device=x.device))
        x = x.squeeze(1)
        x = self.ln(x)
        out = self.head(x)
        
        return out

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoderの実装
    """
    def __init__(self, image_size=224, patch_size=16, in_channel=3,
                 dim=768, hidden_dim=768*4, num_heads=12, activation=nn.GELU(),
                 num_blocks=8, qkv_bias=True, dropout=0., decoder_dim=512,
                 decoder_num_blocks=8, decoder_num_heads=16, quiet_attention=False):
        """
        Args:
            image_size (int): 入力画像の解像度
            patch_size (int): パッチ分割の際の1パッチのピクセル幅
            in_channel (int): 入力画像のチャネル数
            dim (int) : トークン（埋め込みベクトル）の長さ
            hidden_dim (int): FeedForward層でのベクトルの長さ、次元数
            num_heads (int): マルチヘッドの数
            activation (torch.nn.modules.activation): 活性化関数
            num_blocks (int): エンコーダー部分のViTBlockの層数
            qkv_bias (bool): query,key,valueに埋め込む際の全結合層のバイアス
            dropout (float): dropoutの確率
            decoder_dim (int): デコーダー部分のトークン（埋め込みベクトル）の長さ
            decoder_num_blocks (int): デコーダー部分のViTBlockの層数
            decoder_num_heads (int): デコーダー部分でのマルチヘッドの数
            quiet_attention (bool): Trueの場合、softmaxの分母に1を足す
        """
        super().__init__()
        
        self.encoder_dim = dim
        # MAE encoder
        self.patch_embedding = ImagePatchEmbedding(image_size,patch_size,in_channel,dim)
        num_patches = self.patch_embedding.num_patches
        
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,dim)))
        #self.positional_embedding = nn.Parameter(torch.randn(size=(1,num_patches+1,dim))) #　位置埋め込み（学習可能なパラメータ）
        self.positional_embedding = nn.Parameter(torch.zeros(size=(1,1+num_patches,dim)),requires_grad=False) # fixed sin-cos embedding
        
        self.encoder = nn.ModuleList([
            ViTBlock(dim,hidden_dim,num_heads,activation,qkv_bias,dropout,quiet_attention)
            for _ in range(num_blocks)])
        self.encoder_norm = nn.LayerNorm(dim,eps=1e-10)
        
        # MAE decoder
        self.decoder_embedding = nn.Linear(dim,decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(size=(1,1,decoder_dim)))
        self.decoder_pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.decoder = nn.ModuleList([
            ViTBlock(decoder_dim,decoder_dim*4,decoder_num_heads,activation,qkv_bias,dropout,quiet_attention)
            for _ in range(decoder_num_blocks)])
        self.decoder_norm = nn.LayerNorm(decoder_dim,eps=1e-10)
    
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_channel)
        self.initialize_weights() # pos embeddingのinit
    
    def initialize_weights(self):
        """sin,cosによる位置埋め込みの初期化
        """
        # Positional Embedding
        pe = get_2d_sincons_pos_embed(self.positional_embedding.shape[-1] ,self.patch_embedding.grid_size[0], cls_token=True)
        self.positional_embedding.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        
        dec_pe = get_2d_sincons_pos_embed(self.decoder_pos_embedding.shape[-1] ,self.patch_embedding.grid_size[0], cls_token=True)
        self.decoder_pos_embedding.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))
        
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # これどれくらい効果あるんだろ
        w = self.patch_embedding.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_masking(self,x,mask_ratio):
        """トークン化された入力画像をランダムにマスクする
        Args:
            x (torch.Tensor): shape=(N,L,D) N:batch_size, L:length (=num_tokens), D: dim
            mask_ratio (float): マスクの割合
        Returns:
            x_masked (torch.Tensor): shape=(B,L*mask_ratio,D) マスクした後のトークン
            mask (torch.Tensor): shape=(B,L) binary mask: 0 is keep, 1 is remove
            ids_restore (torch.Tensor): shape=(B,L) マスクトークンとマスクされていないトークンの位置を保持したリスト  
        """
        N,L,D = x.shape
        num_keep = int(L * (1 - mask_ratio)) # マスクしないパッチの個数
        
        noise = torch.rand(N,L,device=x.device)
        ids_shuffle = torch.argsort(noise,axis=1) # small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :num_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self,x,mask_ratio):
        """
        Args:
            x (torch.Tensor): shape=(B,C,W,H) 入力画像 B: バッチサイズ
            mask_ratio (float): マスクの割合
        """
        x = self.patch_embedding(x)
        x = x + self.positional_embedding[:,1:,:] # エンコーダではクラストークンをつけない
        
        x, mask, ids_restore = self.random_masking(x,mask_ratio) # (B,L*mask_ratio,D)
        
        cls_token = self.cls_token + self.positional_embedding[:,:1,:]
        cls_tokens = cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_tokens,x],dim=1)
        
        for block in self.encoder:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self,x,ids_restore):
        """
        Args:
            x (torch.Tensor): エンコーダにより集約された特徴量
            ids_restore (torch.Tensor): トークンやマスクされた位置の情報
        """
        x = self.decoder_embedding(x)
        
        # append mask token
        # ここもっといい実装ありそうだけど、どうだろ
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:,1:,:],mask_tokens],dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1) # append cls token
        
        x = x + self.decoder_pos_embedding
        
        for block in self.decoder:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        x = x[:,1:,:] # remove cls token # てかもっと前でよくね ?
        
        return x
        
    def patchify(self,imgs):
        """
        Args: 
            imgs(torch.Tensor): shape=(B,C,H,W)
        Returns:
            torch.Tensor: shape=(B,L,C*P*P) P:1パッチのピクセル幅
        """
        p = self.patch_embedding.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0 # 入力画像はshape=(B,C,H,W)でH=WかつHはpatch_sizeで綺麗に分割しきれないといけない
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        Args:
            x(torch.Tensor): shape=(B,L,P*P*C) 
        Returns: 
            torch.Tensor: (B,C,H,W)
        """
        p = self.patch_embedding.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_loss(self,imgs,pred,mask):
        """
        Args:
            imgs (torch.Tensor): shape=(B,C,H,W) 入力画像 
            pred (torch.Tensor): shape=(B,L,C*P*P) 予測(再構成)画像 C:チャネル数, P:1パッチのピクセル幅
            mask (torch.Tensor): shape=(B,L) 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        
        # 正規化
        mean = target.mean(dim=-1,keepdims=True)
        var  = target.var(dim=-1,keepdims=True)
        target = (target - mean)/(var + 1e-8)**0.5
        
        # MSE Loss
        loss = (pred - target)**2
        loss = loss.mean(dim=-1) # (B,L), それぞれのパッチで平均
        loss = (loss * mask).sum() / mask.sum() # マスクされている部分のみについてlossを計算
        
        return loss
    
    def forward(self,imgs,mask_ratio=0.75):
        """
        モデルのフォワード
        ただしlossまで内部で計算する
        """
        latent,mask,ids_restore = self.forward_encoder(imgs,mask_ratio)
        pred = self.forward_decoder(latent,ids_restore)
        loss = self.forward_loss(imgs,pred,mask)
        
        return loss,pred,mask

    def to_classifier_model(self,num_classes=1000):
        """
        デコーダ部分を捨ててheadをつける
        Args:
            num_classes (int): 分類クラス数
        """
        self.head = nn.Linear(self.encoder_dim,num_classes)
        self.forward = self.forward_classifier

        # メモリを削るためいらないモデル部分を消す
        del self.decoder_embedding
        del self.mask_token
        del self.decoder_pos_embedding
        del self.decoder
        del self.decoder_norm
        del self.decoder_pred
        torch.cuda.empty_cache()

    def forward_classifier(self,x):
        """
        分類器モデルのフォワード
        """
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_tokens,x],dim=1)
        x = x + self.positional_embedding
        
        for block in self.encoder:
            x = block(x)
        x = self.encoder_norm(x)
        x = torch.index_select(x,1,torch.tensor(0,device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

        