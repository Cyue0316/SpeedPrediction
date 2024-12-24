import torch.nn as nn
import torch



class CrossAttentionLayer(nn.Module):
    """Perform cross attention between query and key-value pairs.

    Supports attention across the -2 dim (where -1 dim is `model_dim`).

    E.g.
    - Input shape for `query`: (batch_size, tgt_length, num_nodes, model_dim).
    - Input shape for `key` and `value`: (batch_size, src_length, num_nodes, model_dim).
    - Cross attention will be performed across `num_nodes`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        # Separate linear projections for query, key, and value
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        # Linear projections
        query = self.FC_Q(query)  # (batch_size, ..., tgt_length, model_dim)
        key = self.FC_K(key)      # (batch_size, ..., src_length, model_dim)
        value = self.FC_V(value)  # (batch_size, ..., src_length, model_dim)


        # Qhead (num_heads * batch_size, ..., tgt_length, head_dim)
        # Khead, Vhead (num_heads * batch_size, ..., src_length, head_dim)
        print(query.shape)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)

        # Attention score calculation
        attn_score = (query @ key) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)
        print(attn_score.shape)

        # Apply mask if required
        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        # Softmax and weighted sum of values
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        
        # Concatenate heads
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # (batch_size, ..., tgt_length, model_dim)

        # Final linear projection
        out = self.out_proj(out)

        return out


class CrossAttentionLayerWithFeedForward(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()

        self.cross_attn = CrossAttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value, dim=-2):
        query = query.transpose(dim, -2)  # Transpose for attention
        key = key.transpose(dim, -2)
        value = value.transpose(dim, -2)

        # Cross attention
        residual = query
        out = self.cross_attn(query, key, value)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        # Feed-forward network
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        # Transpose back to original dimension order
        out = out.transpose(dim, -2)
        return out
    
    
# class MLP(nn.Module):
#     def __init__(self, hidden_dim=64):
#         super(MLP, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1) 
#         )

#     def forward(self, x):
#         # x 的维度是 (B, T, N, 1)
#         x = self.mlp(x)
#         return x


class AttnMLPModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=8,
        output_dim=1,
        input_embedding_dim=7,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim

        self.model_dim = (
            input_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        # self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        
        self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_s = nn.ModuleList(
            [
                CrossAttentionLayerWithFeedForward(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim)
        x = x[..., : self.input_dim]
        x = self.input_proj(x)  # Project input to embedding dimension
        kv = x[:, :, :2, : self.input_dim]
        for attn in self.attn_layers_s:
            x = attn(x, kv, kv, dim=2)  # Here, we use x as query, key, and value

        out = self.output_proj(x)  # Project back to output dimension
        return out


