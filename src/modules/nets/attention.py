import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Self-attention

    When ``gating`` is enabled, the query projection is widened to produce both Q and
    per-head gate scores. Gate scores modulate the SDPA output via sigmoid before the
    output projection.
    
    :param dim: The dimension of the input features.
    :param num_heads: The number of attention heads.
    :param qkv_bias: Whether to use bias in the query, key, and value projections.
    :param gating: Whether to use elementwise gating on the output of the attention mechanism.
        cf. https://arxiv.org/abs/2505.06708
    :param attn_dropout: The dropout probability for the attention weights.
    :param proj_dropout: The dropout probability for the output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        gating: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gating = gating

        if gating:
            self.wq = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.wq, self.wk, self.wv, self.proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        kv_input = x if kv is None else kv
        kv_len = kv_input.shape[1]

        if self.gating:
            q_raw = self.wq(x).view(batch_size, seq_len, self.num_heads, -1)
            q, gate_score = torch.split(
                q_raw, [self.head_dim, self.head_dim], dim=-1
            )
        else:
            q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        k = self.wk(kv_input).reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        v = self.wv(kv_input).reshape(batch_size, kv_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        if self.gating:
            out = out * torch.sigmoid(gate_score)

        out = out.reshape(batch_size, seq_len, self.dim)
        out = self.proj_drop(self.proj(out))
        return out
