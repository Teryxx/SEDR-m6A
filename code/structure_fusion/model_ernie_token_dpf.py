import torch
import torch.nn as nn


TOKEN_LEN = 201
ATTN_PADDED_LEN = 203
CENTER_INDEX = TOKEN_LEN // 2


def crop_head6(ernie_attn: torch.Tensor) -> torch.Tensor:
    if ernie_attn.ndim != 4 or ernie_attn.shape[1:] != (1, ATTN_PADDED_LEN, ATTN_PADDED_LEN):
        raise ValueError(
            f"Expected ERNIE attention shape [B,1,{ATTN_PADDED_LEN},{ATTN_PADDED_LEN}], "
            f"got {tuple(ernie_attn.shape)}"
        )
    return ernie_attn[:, 0, 1:-1, 1:-1].float()


def symmetrize_head6(ernie_attn: torch.Tensor) -> torch.Tensor:
    attn = crop_head6(ernie_attn)
    return 0.5 * (attn + attn.transpose(-1, -2))


def row_normalize(attn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    attn = torch.relu(attn)
    denom = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attn / denom


def build_sparse_structure_adjacency(attn: torch.Tensor, topk: int = 4) -> torch.Tensor:
    if attn.ndim != 3 or attn.shape[1:] != (TOKEN_LEN, TOKEN_LEN):
        raise ValueError(f"Expected attention shape [B,{TOKEN_LEN},{TOKEN_LEN}], got {tuple(attn.shape)}")
    topk = max(1, min(int(topk), TOKEN_LEN))
    normalized = row_normalize(attn)
    _, topk_idx = torch.topk(normalized, k=topk, dim=-1)
    mask = torch.zeros_like(normalized, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_idx, value=True)

    idx = torch.arange(TOKEN_LEN, device=attn.device)
    sequence_mask = (idx.view(1, -1, 1) - idx.view(1, 1, -1)).abs() <= 1
    weighted = normalized * mask.float()
    weighted = weighted + sequence_mask.float()
    return row_normalize(weighted)


class LightweightTokenEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.out_norm(tokens + y)


class CenterMotifAttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_queries: int = 4,
        query_uses_global: bool = False,
        bias_scale: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.query_uses_global = bool(query_uses_global)
        self.bias_scale = float(bias_scale)
        self.query_seed = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.normal_(self.query_seed, mean=0.0, std=0.02)
        query_input_dim = hidden_dim * (3 if self.query_uses_global else 2)
        self.query_proj = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * num_queries, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        query_context: torch.Tensor = None,
        attn_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        center = tokens[:, CENTER_INDEX, :]
        local = tokens[:, max(0, CENTER_INDEX - 2): CENTER_INDEX + 3, :].mean(dim=1)
        query_features = [center, local]
        if self.query_uses_global:
            query_features.append(tokens.mean(dim=1))
        query_base = self.query_proj(torch.cat(query_features, dim=-1))
        if query_context is not None:
            if query_context.shape != query_base.shape:
                raise ValueError(
                    f"Expected query_context shape {tuple(query_base.shape)}, got {tuple(query_context.shape)}"
            )
            query_base = query_base + query_context
        queries = query_base.unsqueeze(1) + self.query_seed.unsqueeze(0)
        attn_mask = None
        if attn_bias is not None and self.bias_scale != 0.0:
            if attn_bias.shape != (tokens.shape[0], tokens.shape[1]):
                raise ValueError(
                    f"Expected attn_bias shape {(tokens.shape[0], tokens.shape[1])}, "
                    f"got {tuple(attn_bias.shape)}"
                )
            safe_bias = attn_bias.clamp_min(1e-6).log() * self.bias_scale
            attn_mask = safe_bias[:, None, :].expand(
                tokens.shape[0],
                self.num_queries,
                tokens.shape[1],
            )
            attn_mask = attn_mask.repeat_interleave(self.attn.num_heads, dim=0)
        attended, _ = self.attn(queries, tokens, tokens, attn_mask=attn_mask, need_weights=False)
        return self.out(attended.reshape(tokens.shape[0], self.hidden_dim * self.num_queries))


class LatentQueryReadout(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_queries: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = int(num_queries)
        self.latents = nn.Parameter(torch.zeros(self.num_queries, hidden_dim))
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * self.num_queries, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        queries = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.attn(queries, tokens, tokens, need_weights=False)
        return self.out(attended.reshape(tokens.shape[0], self.hidden_dim * self.num_queries))


class StructureQueryReadout(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_queries: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = int(num_queries)
        self.query_seed = nn.Parameter(torch.zeros(self.num_queries, hidden_dim))
        nn.init.normal_(self.query_seed, mean=0.0, std=0.02)
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * (self.num_queries + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, structure_context: torch.Tensor = None) -> torch.Tensor:
        if structure_context is None:
            structure_context = tokens[:, CENTER_INDEX, :]
        center = tokens[:, CENTER_INDEX, :]
        query_base = self.query_proj(torch.cat([center, structure_context], dim=-1))
        queries = query_base.unsqueeze(1) + self.query_seed.unsqueeze(0)
        attended, _ = self.attn(queries, tokens, tokens, need_weights=False)
        pooled = torch.cat([attended.reshape(tokens.shape[0], self.hidden_dim * self.num_queries), center], dim=-1)
        return self.out(pooled)


class DPFReadout(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        structure_context_scale: float = 1.0,
        task_pool_queries: int = 4,
        include_center_residual_feature: bool = False,
        local_radius: int = 2,
        structure_context_dropout: float = 0.0,
        duplicate_structure_feature: bool = False,
        task_query_uses_global: bool = False,
        include_global_max_feature: bool = False,
        feature_norm: bool = False,
        structure_task_bias_scale: float = 0.0,
        include_topk_structure_context: bool = False,
        gate_topk_structure_context: bool = False,
    ):
        super().__init__()
        self.structure_context_scale = float(structure_context_scale)
        self.task_pool_queries = int(task_pool_queries)
        self.include_center_residual_feature = bool(include_center_residual_feature)
        self.local_radius = int(local_radius)
        self.structure_context_dropout_p = float(structure_context_dropout)
        self.duplicate_structure_feature = bool(duplicate_structure_feature)
        self.task_query_uses_global = bool(task_query_uses_global)
        self.include_global_max_feature = bool(include_global_max_feature)
        self.feature_norm_enabled = bool(feature_norm)
        self.structure_task_bias_scale = float(structure_task_bias_scale)
        self.include_topk_structure_context = bool(include_topk_structure_context)
        self.gate_topk_structure_context = bool(gate_topk_structure_context)
        self.task_pool = CenterMotifAttentionPooling(
            hidden_dim,
            num_heads,
            dropout,
            num_queries=self.task_pool_queries,
            query_uses_global=task_query_uses_global,
            bias_scale=structure_task_bias_scale,
        )
        feature_count = 5
        if self.include_center_residual_feature:
            feature_count += 1
        if self.duplicate_structure_feature:
            feature_count += 1
        if self.include_global_max_feature:
            feature_count += 1
        if self.include_topk_structure_context:
            feature_count += 1
        if self.gate_topk_structure_context:
            self.topk_structure_gate = nn.Linear(hidden_dim * 4, hidden_dim)
            nn.init.zeros_(self.topk_structure_gate.weight)
            nn.init.zeros_(self.topk_structure_gate.bias)
        else:
            self.register_parameter("topk_structure_gate", None)
        self.feature_norm = nn.LayerNorm(hidden_dim) if self.feature_norm_enabled else None
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * feature_count, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        original_tokens: torch.Tensor,
        encoded_tokens: torch.Tensor,
        structure_context: torch.Tensor = None,
        task_query_context: torch.Tensor = None,
        center_local_gate: torch.Tensor = None,
        structure_task_bias: torch.Tensor = None,
        topk_structure_context: torch.Tensor = None,
    ) -> torch.Tensor:
        center = encoded_tokens[:, CENTER_INDEX, :]
        radius = max(0, self.local_radius)
        local = encoded_tokens[:, max(0, CENTER_INDEX - radius): CENTER_INDEX + radius + 1, :].mean(dim=1)
        if center_local_gate is not None:
            if center_local_gate.shape != center[:, :1].shape:
                raise ValueError(
                    f"Expected center_local_gate shape {tuple(center[:, :1].shape)}, got {tuple(center_local_gate.shape)}"
                )
            center = center_local_gate * center + (1.0 - center_local_gate) * local
        global_mean = encoded_tokens.mean(dim=1)
        global_max = encoded_tokens.max(dim=1).values
        task_context = self.task_pool(
            encoded_tokens,
            query_context=task_query_context,
            attn_bias=structure_task_bias,
        )
        if structure_context is None:
            structure_context = original_tokens.mean(dim=1)
        structure_context = self.structure_context_scale * structure_context
        if self.training and self.structure_context_dropout_p > 0.0:
            structure_context = torch.nn.functional.dropout(
                structure_context,
                p=self.structure_context_dropout_p,
                training=True,
            )
        if self.gate_topk_structure_context:
            if topk_structure_context is None:
                topk_structure_context = structure_context
            gate_input = torch.cat(
                [
                    structure_context,
                    topk_structure_context,
                    structure_context - topk_structure_context,
                    structure_context * topk_structure_context,
                ],
                dim=-1,
            )
            gate = torch.sigmoid(self.topk_structure_gate(gate_input))
            structure_context = gate * topk_structure_context + (1.0 - gate) * structure_context
        features = [center, local, global_mean, task_context, structure_context]
        if self.include_global_max_feature:
            features.append(global_max)
        if self.include_topk_structure_context:
            if topk_structure_context is None:
                topk_structure_context = structure_context
            features.append(topk_structure_context)
        if self.duplicate_structure_feature:
            features.append(structure_context)
        if self.include_center_residual_feature:
            features.append(original_tokens[:, CENTER_INDEX, :])
        if self.feature_norm is not None:
            features = [self.feature_norm(feature) for feature in features]
        return self.proj(torch.cat(features, dim=-1))


class GatedStructurePropagationBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        gate_temperature: float = 1.0,
        gate_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.gate_temperature = float(gate_temperature)
        self.gate_residual_scale = float(gate_residual_scale)
        self.candidate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(hidden_dim * 4, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, attn: torch.Tensor, return_gate: bool = False):
        normalized = row_normalize(attn)
        propagated = torch.matmul(normalized, tokens)
        features = torch.cat([tokens, propagated, tokens - propagated, tokens * propagated], dim=-1)
        candidate = self.candidate(features)
        gate_logits = self.gate(features) / max(self.gate_temperature, 1e-6)
        gate = torch.sigmoid(gate_logits)
        out = self.out_norm(tokens + self.gate_residual_scale * gate * candidate)
        if return_gate:
            return out, gate
        return out


class StructureBiasedAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        bias_scale: float,
        bias_mode: str = "log",
    ):
        super().__init__()
        if bias_mode not in {"log", "soft"}:
            raise ValueError(f"Unsupported bias_mode: {bias_mode}")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.bias_scale = float(bias_scale)
        self.bias_mode = bias_mode
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        normalized = row_normalize(attn)
        if self.bias_mode == "soft":
            bias = normalized * self.bias_scale
        else:
            bias = normalized.clamp_min(1e-6).log() * self.bias_scale
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        batch_size, seq_len, _ = tokens.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores + bias.unsqueeze(1)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.norm(tokens + self.dropout(self.out(context)))


class StructureBiasedAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        layers: int = 1,
        bias_scale: float = 0.1,
        center_band_only: bool = False,
        center_band_radius: int = 12,
        center_query_only: bool = False,
        topk: int | None = None,
        local_band_bias: bool = False,
        local_band_radius: int = 2,
        bias_mode: str = "log",
    ):
        super().__init__()
        self.bias_scale = float(bias_scale)
        self.center_band_only = bool(center_band_only)
        self.center_band_radius = int(center_band_radius)
        self.center_query_only = bool(center_query_only)
        self.topk = None if topk is None else int(topk)
        self.local_band_bias = bool(local_band_bias)
        self.local_band_radius = int(local_band_radius)
        self.bias_mode = bias_mode
        self.layers = nn.ModuleList(
            StructureBiasedAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias_scale=bias_scale,
                bias_mode=bias_mode,
            )
            for _ in range(max(1, int(layers)))
        )

    def forward(self, tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if self.topk is not None:
            normalized = row_normalize(attn)
            topk = max(1, min(self.topk, normalized.shape[-1]))
            _, topk_idx = torch.topk(normalized, k=topk, dim=-1)
            mask = torch.zeros_like(normalized, dtype=torch.bool)
            mask.scatter_(dim=-1, index=topk_idx, value=True)
            attn = row_normalize(normalized * mask.float())
        if self.center_band_only:
            idx = torch.arange(TOKEN_LEN, device=attn.device)
            center_mask = (idx - CENTER_INDEX).abs() <= self.center_band_radius
            band_mask = center_mask.view(1, -1, 1) | center_mask.view(1, 1, -1)
            attn = row_normalize(row_normalize(attn) * band_mask.float())
        if self.center_query_only:
            query_mask = torch.zeros((TOKEN_LEN, 1), device=attn.device, dtype=attn.dtype)
            query_mask[CENTER_INDEX, 0] = 1.0
            attn = row_normalize(row_normalize(attn) * query_mask.view(1, TOKEN_LEN, 1))
        if self.local_band_bias:
            idx = torch.arange(TOKEN_LEN, device=attn.device)
            local_mask = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= self.local_band_radius
            local = local_mask.float().unsqueeze(0).expand_as(attn)
            attn = row_normalize(row_normalize(attn) + local)
        for layer in self.layers:
            tokens = layer(tokens, attn)
        return tokens


class StructureFeatureCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        layers: int = 1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.feature_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(max(1, int(layers)))
        )
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in self.layers)

    def forward(self, tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        normalized = row_normalize(attn)
        center_affinity = normalized[:, CENTER_INDEX, :]
        column_mass = normalized.mean(dim=1)
        diagonal = torch.diagonal(normalized, dim1=-2, dim2=-1)
        structure_features = torch.stack([center_affinity, column_mass, diagonal], dim=-1)
        structure_tokens = self.feature_proj(structure_features)
        for attn_layer, norm in zip(self.layers, self.norms):
            context, _ = attn_layer(tokens, structure_tokens, structure_tokens, need_weights=False)
            tokens = norm(tokens + self.residual_scale * self.dropout(context))
        return tokens


class GraphTransformerBiasLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, bias_scale: float):
        super().__init__()
        self.attn = StructureBiasedAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias_scale=bias_scale,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        tokens = self.attn(tokens, attn)
        return self.out_norm(tokens + self.ffn(tokens))


class GraphTransformerBiasBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        layers: int = 1,
        bias_scale: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            GraphTransformerBiasLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias_scale=bias_scale,
            )
            for _ in range(max(1, int(layers)))
        )

    def forward(self, tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, attn)
        return tokens


class ResidualGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.matmul(adjacency, tokens)
        message = torch.nn.functional.gelu(self.proj(message))
        message = self.dropout(message)
        return self.norm(tokens + message)


class ResidualGCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, graph_topk: int, gcn_layers: int):
        super().__init__()
        self.graph_topk = graph_topk
        self.layers = nn.ModuleList(
            ResidualGCNLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(max(1, int(gcn_layers)))
        )

    def forward(self, tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        adjacency = build_sparse_structure_adjacency(attn, topk=self.graph_topk)
        for layer in self.layers:
            tokens = layer(tokens, adjacency)
        return tokens


class GatedResidualGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor, return_gate: bool = False):
        message = torch.matmul(adjacency, tokens)
        message = torch.nn.functional.gelu(self.proj(message))
        message = self.dropout(message)
        gate_features = torch.cat([tokens, message, tokens - message, tokens * message], dim=-1)
        gate = self.gate(gate_features)
        out = self.norm(tokens + self.residual_scale * gate * message)
        if return_gate:
            return out, gate
        return out


class GatedResidualGCNBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        graph_topk: int,
        gcn_layers: int,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.graph_topk = graph_topk
        self.layers = nn.ModuleList(
            GatedResidualGCNLayer(hidden_dim=hidden_dim, dropout=dropout, residual_scale=residual_scale)
            for _ in range(max(1, int(gcn_layers)))
        )

    def forward(self, tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        adjacency = build_sparse_structure_adjacency(attn, topk=self.graph_topk)
        for layer in self.layers:
            tokens = layer(tokens, adjacency)
        return tokens


class DPFInspiredTokenClassifier(nn.Module):
    def __init__(
        self,
        variant: str = "center_attn",
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        graph_topk: int = 4,
        gcn_layers: int = 2,
    ):
        super().__init__()
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.token_encoder = LightweightTokenEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.readout = DPFReadout(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            structure_context_scale=1.0,
        )
        self.classification = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        if variant == "center_attn":
            self.fusion_block = None
        elif variant == "gated_propagation":
            self.fusion_block = GatedStructurePropagationBlock(hidden_dim=hidden_dim, dropout=dropout)
        elif variant == "residual_gcn":
            self.fusion_block = ResidualGCNBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                graph_topk=graph_topk,
                gcn_layers=gcn_layers,
            )
        elif variant == "gated_residual_gcn":
            self.fusion_block = GatedResidualGCNBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                graph_topk=graph_topk,
                gcn_layers=gcn_layers,
            )
        else:
            raise ValueError(f"Unsupported DPFunc-inspired variant: {variant}")

    def _validate_tokens(self, tokens: torch.Tensor):
        if tokens.ndim != 3 or tokens.shape[1] != TOKEN_LEN or tokens.shape[2] != self.hidden_dim:
            raise ValueError(
                f"Expected ERNIE token shape [B,{TOKEN_LEN},{self.hidden_dim}], got {tuple(tokens.shape)}"
            )

    def forward(self, batch):
        tokens = batch["ernie_tokens"].float()
        self._validate_tokens(tokens)
        encoded = self.token_encoder(tokens)

        structure_context = None
        if self.fusion_block is not None:
            attn = symmetrize_head6(batch["ernie_attn"])
            encoded = self.fusion_block(encoded, attn)
            structure_context = torch.matmul(row_normalize(attn)[:, CENTER_INDEX:CENTER_INDEX + 1, :], encoded).squeeze(1)

        x = self.readout(tokens, encoded, structure_context=structure_context)
        for layer in self.classification[:-1]:
            x = layer(x)
        embedding = x
        logits = self.classification[-1](x)
        return logits, embedding
