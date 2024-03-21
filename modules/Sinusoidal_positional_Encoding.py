# mixed sin cos pos embedding
import numpy as np

def get_2d_sincons_pos_embed(embed_dim,grid_size,cls_token=False):
    """
    Args:
        embed_dim (int): トークンの長さ、次元数
        grid_size (int): パッチ分割の格子の縦、横の長さ 
    Returns:
        pos_embed (nd.array): [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size,dtype=np.float32)
    grid_w = np.arange(grid_size,dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0) 
    grid = grid.reshape([2, 1, grid_size, grid_size]) # (2,grid_size,grid_size) -> (2,1,grid_size,grid_size)
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Args:
        embed_dim (int): トークンの長さ、次元数
        pos (nd.array): 符号化する位置のリスト: shape=(M,)
    Returns:
        nd.array: 位置埋め込みベクトル: shape(M,embed_dim)
    """
    assert embed_dim%2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim /2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md",pos,omega) # shape=(m,embed_dim/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    #emb = np.concatenate([emb_sin,emb_cos],axis=1) # shape=(m,embed_dim)
    
    # 一応ちゃんと交互にsin,cosが来るようにする
    m = emb_sin.shape[0]
    emb = np.empty(shape=(m,embed_dim))
    emb[:,0::2] = emb_sin
    emb[:,1::2] = emb_cos
    
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Args:
        embed_dim (int): トークンの長さ、次元数
        grid (nd.array): 符号化する位置のリスト: shape=(H,W)
    Returns:
        nd.array: 位置埋め込みベクトル: shape(H*W,embed_dim)
    """
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb