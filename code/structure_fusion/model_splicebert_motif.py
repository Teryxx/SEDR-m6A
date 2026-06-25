import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from structure_fusion.model_ernie_token_dpf import (
    ATTN_PADDED_LEN,
    CENTER_INDEX,
    TOKEN_LEN,
    DPFReadout,
    GatedResidualGCNBlock,
    GatedStructurePropagationBlock,
    GraphTransformerBiasBlock,
    LatentQueryReadout,
    LightweightTokenEncoder,
    ResidualGCNBlock,
    StructureQueryReadout,
    StructureBiasedAttentionBlock,
    StructureFeatureCrossAttentionBlock,
    row_normalize,
    symmetrize_head6,
)


DRACH_MOTIFS = tuple(
    "".join(parts)
    for parts in itertools.product(("A", "G", "T"), ("A", "G"), ("A",), ("C",), ("A", "C", "T"))
)
DRACH_TO_ID = {motif: idx for idx, motif in enumerate(DRACH_MOTIFS)}


def center_drach(seq: str) -> str:
    seq = str(seq).strip().upper().replace("U", "T")
    if len(seq) != TOKEN_LEN:
        raise ValueError(f"Expected {TOKEN_LEN}nt sequence, got length {len(seq)}")
    return seq[CENTER_INDEX - 2: CENTER_INDEX + 3]


def center_drach_to_id(seq: str) -> int:
    motif = center_drach(seq)
    if motif not in DRACH_TO_ID:
        raise ValueError(f"Expected central DRACH motif, got {motif!r}")
    return DRACH_TO_ID[motif]


def validate_motif_ids(motif_ids: torch.Tensor) -> torch.Tensor:
    motif_ids = motif_ids.long()
    if motif_ids.ndim != 1:
        raise ValueError(f"Expected motif_ids shape [B], got {tuple(motif_ids.shape)}")
    if bool((motif_ids < 0).any()) or bool((motif_ids >= len(DRACH_MOTIFS)).any()):
        raise ValueError("motif_ids must be in [0, 17]")
    return motif_ids


class MotifOnlyClassifier(nn.Module):
    def __init__(self, motif_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.motif_embedding = nn.Embedding(len(DRACH_MOTIFS), motif_dim)
        self.representation = nn.Sequential(
            nn.Linear(motif_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, batch):
        motif_ids = validate_motif_ids(batch["motif_ids"])
        x = self.motif_embedding(motif_ids)
        embedding = self.representation(x)
        logits = self.classifier(embedding)
        return logits, embedding


class MotifAwareDPFClassifier(nn.Module):
    def __init__(
        self,
        variant: str = "residual_gcn",
        hidden_dim: int = 512,
        motif_dim: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        graph_topk: int = 4,
        gcn_layers: int = 2,
        motif_fusion_mode: str = "concat",
        motif_dropout: float = 0.0,
        motif_delta_scale: float = 0.1,
        structure_fusion_mode: str = "single_readout",
        classifier_hidden_dim: int = 256,
        gate_temperature: float = 1.0,
        gate_residual_scale: float = 1.0,
        gated_residual_scale: float | None = None,
        readout_scale: float = 1.0,
        structure_context_scale: float = 1.0,
        motif_context_scale: float = 1.0,
        fusion_norm: str = "none",
        concat_residual_scale: float | None = None,
        classifier_activation: str = "relu",
        sequence_residual_scale: float = 0.0,
        motif_norm: str = "none",
        concat_input_norm: str = "none",
        fused_feature_norm: str = "none",
        logit_residual_scale: float = 0.0,
        center_residual_scale: float = 0.0,
        track_gate_regularization: bool = False,
        fusion_input_dropout: float = 0.0,
        center_token_bias_scale: float = 0.0,
        task_pool_queries: int = 4,
        readout_affine_calibration: bool = False,
        motif_affine_calibration: bool = False,
        fused_affine_calibration: bool = False,
        motif_query_bias: bool = False,
        structure_context_residual: bool = False,
        structure_context_residual_norm: bool = False,
        readout_center_feature: bool = False,
        readout_local_radius: int = 2,
        classifier_input_dropout: float = 0.0,
        structure_context_dropout: float = 0.0,
        fused_feature_noise_std: float = 0.0,
        classifier_post_activation_norm: bool = False,
        motif_center_local_gate: bool = False,
        classifier_residual_blend: bool = False,
        fused_bias_calibration: bool = False,
        duplicate_structure_feature: bool = False,
        task_query_uses_global: bool = False,
        structure_correction_head: bool = False,
        motif_product_interaction: bool = False,
        readout_global_max_feature: bool = False,
        classifier_weight_norm: bool = False,
        zero_init_classifier_output: bool = False,
        zero_init_classifier_bias: bool = False,
        fused_correction_head: bool = False,
        readout_feature_norm: bool = False,
        structure_stats_feature: bool = False,
        motif_structure_residual_scale: bool = False,
        structure_interaction_feature: bool = False,
        center_readout_interaction_feature: bool = False,
        motif_readout_mode: str = "embedding",
        motif_residual_fusion: bool = False,
        fusion_family: str = "none",
        readout_family: str = "dpf",
        arch_layers: int = 1,
        arch_bias_scale: float = 0.1,
        arch_latent_queries: int = 4,
        arch_num_experts: int = 3,
        structure_reliability_gate: bool = False,
    ):
        super().__init__()
        if motif_fusion_mode not in {"concat", "film", "logit_adapter"}:
            raise ValueError(f"Unsupported motif_fusion_mode: {motif_fusion_mode}")
        if structure_fusion_mode not in {"single_readout", "dual_readout"}:
            raise ValueError(f"Unsupported structure_fusion_mode: {structure_fusion_mode}")
        if fusion_norm not in {"none", "readout_layernorm", "readout_rmsnorm"}:
            raise ValueError(f"Unsupported fusion_norm: {fusion_norm}")
        if classifier_activation not in {"relu", "gelu", "silu"}:
            raise ValueError(f"Unsupported classifier_activation: {classifier_activation}")
        if motif_norm not in {"none", "layernorm"}:
            raise ValueError(f"Unsupported motif_norm: {motif_norm}")
        if concat_input_norm not in {"none", "l2"}:
            raise ValueError(f"Unsupported concat_input_norm: {concat_input_norm}")
        if fused_feature_norm not in {"none", "layernorm", "rmsnorm"}:
            raise ValueError(f"Unsupported fused_feature_norm: {fused_feature_norm}")
        if motif_readout_mode not in {"embedding", "mean_max"}:
            raise ValueError(f"Unsupported motif_readout_mode: {motif_readout_mode}")
        if fusion_family not in {"none", "motif_local_global", "moe", "reliability_gate", "motif_residual_gate", "token_moe"}:
            raise ValueError(f"Unsupported fusion_family: {fusion_family}")
        if readout_family not in {
            "dpf",
            "latent_query",
            "structure_biased_dpf",
            "structure_query",
            "topk_context_dpf",
            "gated_topk_context_dpf",
        }:
            raise ValueError(f"Unsupported readout_family: {readout_family}")
        if classifier_weight_norm and zero_init_classifier_output:
            raise ValueError("classifier_weight_norm and zero_init_classifier_output should be tested separately")
        if zero_init_classifier_output and zero_init_classifier_bias:
            raise ValueError("zero_init_classifier_output already zeros the classifier bias")
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.motif_fusion_mode = motif_fusion_mode
        self.structure_fusion_mode = structure_fusion_mode
        self.motif_dropout = float(motif_dropout)
        self.motif_delta_scale = float(motif_delta_scale)
        self.readout_scale = float(readout_scale)
        self.motif_context_scale = float(motif_context_scale)
        self.fusion_norm_mode = fusion_norm
        self.concat_residual_scale = concat_residual_scale
        self.classifier_activation = classifier_activation
        self.sequence_residual_scale = float(sequence_residual_scale)
        self.motif_norm_mode = motif_norm
        self.concat_input_norm = concat_input_norm
        self.fused_feature_norm_mode = fused_feature_norm
        self.logit_residual_scale = float(logit_residual_scale)
        self.center_residual_scale = float(center_residual_scale)
        self.track_gate_regularization = bool(track_gate_regularization)
        self.fusion_input_dropout_p = float(fusion_input_dropout)
        self.center_token_bias_scale = float(center_token_bias_scale)
        self.task_pool_queries = int(task_pool_queries)
        self.readout_affine_calibration = bool(readout_affine_calibration)
        self.motif_affine_calibration = bool(motif_affine_calibration)
        self.fused_affine_calibration = bool(fused_affine_calibration)
        self.motif_query_bias = bool(motif_query_bias)
        self.structure_context_residual = bool(structure_context_residual)
        self.structure_context_residual_norm_enabled = bool(structure_context_residual_norm)
        self.readout_center_feature = bool(readout_center_feature)
        self.readout_local_radius = int(readout_local_radius)
        self.classifier_input_dropout_p = float(classifier_input_dropout)
        self.structure_context_dropout_p = float(structure_context_dropout)
        self.fused_feature_noise_std = float(fused_feature_noise_std)
        self.classifier_post_activation_norm = bool(classifier_post_activation_norm)
        self.motif_center_local_gate = bool(motif_center_local_gate)
        self.classifier_residual_blend = bool(classifier_residual_blend)
        self.fused_bias_calibration = bool(fused_bias_calibration)
        self.duplicate_structure_feature = bool(duplicate_structure_feature)
        self.task_query_uses_global = bool(task_query_uses_global)
        self.structure_correction_head = bool(structure_correction_head)
        self.motif_product_interaction = bool(motif_product_interaction)
        self.readout_global_max_feature = bool(readout_global_max_feature)
        self.classifier_weight_norm = bool(classifier_weight_norm)
        self.zero_init_classifier_output = bool(zero_init_classifier_output)
        self.zero_init_classifier_bias = bool(zero_init_classifier_bias)
        self.fused_correction_head = bool(fused_correction_head)
        self.readout_feature_norm = bool(readout_feature_norm)
        self.structure_stats_feature = bool(structure_stats_feature)
        self.motif_structure_residual_scale = bool(motif_structure_residual_scale)
        self.structure_interaction_feature = bool(structure_interaction_feature)
        self.center_readout_interaction_feature = bool(center_readout_interaction_feature)
        self.motif_readout_mode = motif_readout_mode
        self.motif_residual_fusion = bool(motif_residual_fusion)
        self.fusion_family = fusion_family
        self.readout_family = readout_family
        self.arch_layers = int(arch_layers)
        self.arch_bias_scale = float(arch_bias_scale)
        self.arch_latent_queries = int(arch_latent_queries)
        self.arch_num_experts = int(arch_num_experts)
        self.graph_topk = int(graph_topk)
        self.gated_residual_scale = float(gate_residual_scale if gated_residual_scale is None else gated_residual_scale)
        self.structure_reliability_gate_enabled = bool(structure_reliability_gate)
        self._last_motif_delta = None
        self._last_gate = None
        self.token_encoder = LightweightTokenEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.readout = DPFReadout(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            structure_context_scale=structure_context_scale,
            task_pool_queries=task_pool_queries,
            include_center_residual_feature=readout_center_feature,
            local_radius=readout_local_radius,
            structure_context_dropout=structure_context_dropout,
            duplicate_structure_feature=duplicate_structure_feature,
            task_query_uses_global=task_query_uses_global,
            include_global_max_feature=readout_global_max_feature,
            feature_norm=readout_feature_norm,
            structure_task_bias_scale=arch_bias_scale if readout_family == "structure_biased_dpf" else 0.0,
            include_topk_structure_context=readout_family == "topk_context_dpf",
            gate_topk_structure_context=readout_family == "gated_topk_context_dpf",
        )
        self.latent_readout = (
            LatentQueryReadout(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_queries=arch_latent_queries,
            )
            if self.readout_family == "latent_query"
            else None
        )
        self.structure_query_readout = (
            StructureQueryReadout(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_queries=arch_latent_queries,
            )
            if self.readout_family == "structure_query"
            else None
        )
        if structure_fusion_mode == "dual_readout":
            self.dual_readout_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.dual_readout_fusion = None
        if fusion_norm == "readout_layernorm":
            self.readout_norm = nn.LayerNorm(hidden_dim)
        elif fusion_norm == "readout_rmsnorm":
            self.readout_norm = nn.RMSNorm(hidden_dim)
        else:
            self.readout_norm = None
        if self.readout_affine_calibration:
            self.readout_calibration_weight = nn.Parameter(torch.ones(hidden_dim))
            self.readout_calibration_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter("readout_calibration_weight", None)
            self.register_parameter("readout_calibration_bias", None)
        if variant == "reliability_scaled_attn":
            self.structure_attention_reliability_gate = nn.Sequential(
                nn.Linear(4, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            nn.init.zeros_(self.structure_attention_reliability_gate[-1].weight)
            nn.init.zeros_(self.structure_attention_reliability_gate[-1].bias)
        else:
            self.structure_attention_reliability_gate = None
        self.structure_context_residual_norm = (
            nn.LayerNorm(hidden_dim) if self.structure_context_residual_norm_enabled else None
        )
        if self.structure_correction_head:
            self.structure_correction = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.structure_correction[-1].weight)
            nn.init.zeros_(self.structure_correction[-1].bias)
        else:
            self.structure_correction = None
        if self.structure_reliability_gate_enabled or self.fusion_family == "reliability_gate":
            self.structure_reliability_gate = nn.Sequential(
                nn.Linear(4, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            nn.init.zeros_(self.structure_reliability_gate[-1].weight)
            nn.init.zeros_(self.structure_reliability_gate[-1].bias)
        else:
            self.structure_reliability_gate = None
        if self.structure_interaction_feature:
            self.structure_interaction_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            nn.init.zeros_(self.structure_interaction_proj.weight)
            nn.init.zeros_(self.structure_interaction_proj.bias)
        else:
            self.structure_interaction_proj = None
        if self.center_readout_interaction_feature:
            self.center_readout_interaction_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            nn.init.zeros_(self.center_readout_interaction_proj.weight)
            nn.init.zeros_(self.center_readout_interaction_proj.bias)
        else:
            self.center_readout_interaction_proj = None
        self.fusion_input_dropout = nn.Dropout(fusion_input_dropout) if fusion_input_dropout > 0.0 else None
        self.motif_embedding = nn.Embedding(len(DRACH_MOTIFS), motif_dim)
        if self.motif_readout_mode == "mean_max":
            self.motif_readout = nn.Sequential(
                nn.Linear(motif_dim * 2, motif_dim),
                nn.LayerNorm(motif_dim),
                nn.GELU(),
            )
        else:
            self.motif_readout = None
        self.motif_norm = nn.LayerNorm(motif_dim) if motif_norm == "layernorm" else None
        if self.motif_query_bias:
            self.motif_query_proj = nn.Linear(motif_dim, hidden_dim)
            nn.init.zeros_(self.motif_query_proj.weight)
            nn.init.zeros_(self.motif_query_proj.bias)
        else:
            self.motif_query_proj = None
        if self.motif_structure_residual_scale:
            self.motif_structure_scale_proj = nn.Linear(motif_dim, hidden_dim)
            nn.init.zeros_(self.motif_structure_scale_proj.weight)
            nn.init.zeros_(self.motif_structure_scale_proj.bias)
        else:
            self.motif_structure_scale_proj = None
        if self.motif_center_local_gate:
            self.motif_center_local_gate_proj = nn.Linear(motif_dim, 1)
            nn.init.zeros_(self.motif_center_local_gate_proj.weight)
            nn.init.zeros_(self.motif_center_local_gate_proj.bias)
        else:
            self.motif_center_local_gate_proj = None
        if self.motif_product_interaction:
            self.motif_product_proj = nn.Linear(motif_dim, hidden_dim)
            self.motif_product_out = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.motif_product_out.weight)
            nn.init.zeros_(self.motif_product_out.bias)
        else:
            self.motif_product_proj = None
            self.motif_product_out = None
        if self.motif_affine_calibration:
            self.motif_calibration_weight = nn.Parameter(torch.ones(motif_dim))
            self.motif_calibration_bias = nn.Parameter(torch.zeros(motif_dim))
        else:
            self.register_parameter("motif_calibration_weight", None)
            self.register_parameter("motif_calibration_bias", None)
        if self.motif_residual_fusion:
            self.motif_residual_proj = nn.Linear(motif_dim, hidden_dim)
            nn.init.zeros_(self.motif_residual_proj.weight)
            nn.init.zeros_(self.motif_residual_proj.bias)
        else:
            self.motif_residual_proj = None
        if self.fusion_family == "motif_local_global":
            self.local_global_branch_mixer = nn.Sequential(
                nn.Linear(hidden_dim * 4 + motif_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 4),
            )
            self.local_global_branch_out = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.local_global_branch_mixer[-1].weight)
            nn.init.zeros_(self.local_global_branch_mixer[-1].bias)
            nn.init.zeros_(self.local_global_branch_out.weight)
            nn.init.zeros_(self.local_global_branch_out.bias)
        else:
            self.local_global_branch_mixer = None
            self.local_global_branch_out = None
        if self.fusion_family == "motif_residual_gate":
            self.motif_residual_gate = nn.Sequential(
                nn.Linear(hidden_dim + motif_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
            self.motif_residual_out = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.motif_residual_gate[-1].weight)
            nn.init.zeros_(self.motif_residual_gate[-1].bias)
            nn.init.zeros_(self.motif_residual_out.weight)
            nn.init.zeros_(self.motif_residual_out.bias)
        else:
            self.motif_residual_gate = None
            self.motif_residual_out = None
        if self.fusion_family == "token_moe":
            self.token_moe_gate = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.token_moe_gate[-1].weight)
            nn.init.zeros_(self.token_moe_gate[-1].bias)
        else:
            self.token_moe_gate = None
        if self.fusion_family == "moe":
            self.sequence_expert = nn.Linear(hidden_dim, hidden_dim)
            self.structure_expert = nn.Linear(hidden_dim, hidden_dim)
            self.fused_expert = nn.Linear(hidden_dim, hidden_dim)
            self.moe_gate = nn.Sequential(
                nn.Linear(hidden_dim + motif_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 3),
            )
            self.moe_out = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.moe_gate[-1].weight)
            nn.init.zeros_(self.moe_gate[-1].bias)
            nn.init.zeros_(self.moe_out.weight)
            nn.init.zeros_(self.moe_out.bias)
        else:
            self.sequence_expert = None
            self.structure_expert = None
            self.fused_expert = None
            self.moe_gate = None
            self.moe_out = None
        if motif_fusion_mode == "concat":
            self.motif_fusion = nn.Sequential(
                nn.Linear(hidden_dim + motif_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.motif_fusion = None
        if motif_fusion_mode == "film":
            self.motif_film = nn.Linear(motif_dim, hidden_dim * 2)
            nn.init.zeros_(self.motif_film.weight)
            nn.init.zeros_(self.motif_film.bias)
            self.film_norm = nn.LayerNorm(hidden_dim)
        else:
            self.motif_film = None
            self.film_norm = None
        if fused_feature_norm == "layernorm":
            self.fused_feature_norm = nn.LayerNorm(hidden_dim)
        elif fused_feature_norm == "rmsnorm":
            self.fused_feature_norm = nn.RMSNorm(hidden_dim)
        else:
            self.fused_feature_norm = None
        self.classifier_input_dropout = (
            nn.Dropout(classifier_input_dropout) if classifier_input_dropout > 0.0 else None
        )
        if self.fused_affine_calibration:
            self.fused_calibration_weight = nn.Parameter(torch.ones(hidden_dim))
            self.fused_calibration_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter("fused_calibration_weight", None)
            self.register_parameter("fused_calibration_bias", None)
        if self.fused_bias_calibration:
            self.fused_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter("fused_bias", None)
        if self.fused_correction_head:
            self.fused_correction = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.fused_correction[-1].weight)
            nn.init.zeros_(self.fused_correction[-1].bias)
        else:
            self.fused_correction = None
        if self.structure_stats_feature:
            self.structure_stats_proj = nn.Linear(4, hidden_dim)
            nn.init.zeros_(self.structure_stats_proj.weight)
            nn.init.zeros_(self.structure_stats_proj.bias)
        else:
            self.structure_stats_proj = None
        self.classifier_residual_proj = nn.Linear(hidden_dim, classifier_hidden_dim) if self.classifier_residual_blend else None
        activation_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }[classifier_activation]
        classifier_hidden_dim = int(classifier_hidden_dim)
        classifier_layers = [
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.Dropout(dropout),
            activation_layer(),
        ]
        if self.classifier_post_activation_norm:
            classifier_layers.append(nn.LayerNorm(classifier_hidden_dim))
        output_layer = nn.Linear(classifier_hidden_dim, 2)
        if self.zero_init_classifier_output:
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        elif self.zero_init_classifier_bias:
            nn.init.zeros_(output_layer.bias)
        if self.classifier_weight_norm:
            parametrizations = getattr(nn.utils, "parametrizations", None)
            if parametrizations is not None and hasattr(parametrizations, "weight_norm"):
                output_layer = parametrizations.weight_norm(output_layer)
            else:
                output_layer = nn.utils.weight_norm(output_layer)
        classifier_layers.append(output_layer)
        self.classification = nn.Sequential(*classifier_layers)
        if self.logit_residual_scale != 0.0:
            self.logit_residual = nn.Linear(classifier_hidden_dim, 2)
            nn.init.zeros_(self.logit_residual.weight)
            nn.init.zeros_(self.logit_residual.bias)
        else:
            self.logit_residual = None
        if motif_fusion_mode == "logit_adapter":
            adapter_hidden = max(8, hidden_dim // 2)
            self.logit_adapter = nn.Sequential(
                nn.Linear(hidden_dim + motif_dim, adapter_hidden),
                nn.LayerNorm(adapter_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_hidden, 2),
            )
            nn.init.zeros_(self.logit_adapter[-1].weight)
            nn.init.zeros_(self.logit_adapter[-1].bias)
        else:
            self.logit_adapter = None

        if variant == "center_attn":
            self.fusion_block = None
        elif variant == "gated_propagation":
            self.fusion_block = GatedStructurePropagationBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                gate_temperature=gate_temperature,
                gate_residual_scale=gate_residual_scale,
            )
        elif variant == "structure_biased_attn":
            self.fusion_block = StructureBiasedAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                bias_scale=arch_bias_scale,
            )
        elif variant == "reliability_scaled_attn":
            self.fusion_block = StructureBiasedAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                bias_scale=arch_bias_scale,
            )
        elif variant == "center_band_attn":
            self.fusion_block = StructureBiasedAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                bias_scale=arch_bias_scale,
                center_band_only=True,
            )
        elif variant == "center_query_attn":
            self.fusion_block = StructureBiasedAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                bias_scale=arch_bias_scale,
                center_query_only=True,
            )
        elif variant == "topk_structure_attn":
            self.fusion_block = StructureBiasedAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                bias_scale=arch_bias_scale,
                topk=graph_topk,
            )
        elif variant == "soft_structure_attn":
            self.fusion_block = StructureBiasedAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                bias_scale=arch_bias_scale,
                bias_mode="soft",
            )
        elif variant == "local_plus_structure_attn":
            self.fusion_block = StructureBiasedAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                bias_scale=arch_bias_scale,
                local_band_bias=True,
            )
        elif variant == "structure_cross_attn":
            self.fusion_block = StructureFeatureCrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                residual_scale=arch_bias_scale,
            )
        elif variant == "graph_transformer_bias":
            self.fusion_block = GraphTransformerBiasBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layers=arch_layers,
                bias_scale=arch_bias_scale,
            )
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
                residual_scale=self.gated_residual_scale,
            )
        else:
            raise ValueError(f"Unsupported motif-aware DPF variant: {variant}")

    def _validate_tokens(self, tokens: torch.Tensor):
        if tokens.ndim != 3 or tokens.shape[1] != TOKEN_LEN or tokens.shape[2] != self.hidden_dim:
            raise ValueError(
                f"Expected token shape [B,{TOKEN_LEN},{self.hidden_dim}], got {tuple(tokens.shape)}"
            )

    def _motif_context(self, motif_ids: torch.Tensor) -> torch.Tensor:
        context = self.motif_embedding(motif_ids)
        if self.motif_readout is not None:
            motif_mean = self.motif_embedding.weight.mean(dim=0, keepdim=True).expand_as(context)
            motif_max = torch.maximum(context, motif_mean)
            context = self.motif_readout(torch.cat([context, motif_max], dim=-1))
        if self.training and self.motif_dropout > 0.0:
            keep_prob = max(1.0 - self.motif_dropout, 1e-6)
            keep = torch.rand(context.shape[0], 1, device=context.device, dtype=context.dtype) < keep_prob
            context = context * keep / keep_prob
        return context

    def motif_regularization_loss(self) -> torch.Tensor:
        if self._last_motif_delta is None:
            return next(self.parameters()).new_zeros(())
        return self._last_motif_delta.pow(2).mean()

    def gate_balance_loss(self) -> torch.Tensor:
        if self._last_gate is None:
            return next(self.parameters()).new_zeros(())
        return (self._last_gate - 0.5).pow(2).mean()

    def _structure_reliability_features(self, attn: torch.Tensor) -> torch.Tensor:
        normalized = row_normalize(attn)
        entropy = -(normalized.clamp_min(1e-6).log() * normalized).sum(dim=-1).mean(dim=-1, keepdim=True)
        center_row = normalized[:, CENTER_INDEX, :]
        center_peak = center_row.max(dim=-1, keepdim=True).values
        local_start = max(0, CENTER_INDEX - 2)
        local_end = min(TOKEN_LEN, CENTER_INDEX + 3)
        local_mass = center_row[:, local_start:local_end].sum(dim=-1, keepdim=True)
        sparsity = (normalized > (1.0 / TOKEN_LEN)).float().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return torch.cat([entropy, center_peak, local_mass, sparsity], dim=-1)

    def forward(self, batch):
        tokens = batch["ernie_tokens"].float()
        self._validate_tokens(tokens)
        motif_ids = validate_motif_ids(batch["motif_ids"])
        self._last_motif_delta = None
        self._last_gate = None
        sequence_encoded = self.token_encoder(tokens)
        encoded = sequence_encoded
        motif_context = self._motif_context(motif_ids)

        structure_context = None
        attn = None
        if self.fusion_block is not None:
            ernie_attn = batch["ernie_attn"]
            if ernie_attn.ndim != 4 or ernie_attn.shape[1:] != (1, ATTN_PADDED_LEN, ATTN_PADDED_LEN):
                raise ValueError(f"Expected ernie_attn shape [B,1,203,203], got {tuple(ernie_attn.shape)}")
            attn = symmetrize_head6(ernie_attn)
            if self.track_gate_regularization:
                encoded, gate = self.fusion_block(encoded, attn, return_gate=True)
                self._last_gate = gate
            else:
                encoded = self.fusion_block(encoded, attn)
            if self.motif_structure_scale_proj is not None:
                structure_scale = 1.0 + 0.1 * torch.tanh(self.motif_structure_scale_proj(motif_context))
                encoded = sequence_encoded + structure_scale.unsqueeze(1) * (encoded - sequence_encoded)
            if self.structure_attention_reliability_gate is not None:
                reliability_features = self._structure_reliability_features(attn)
                reliability_gate = torch.sigmoid(
                    self.structure_attention_reliability_gate(reliability_features)
                )
                encoded = sequence_encoded + reliability_gate.unsqueeze(-1) * (encoded - sequence_encoded)
            structure_context = torch.matmul(
                row_normalize(attn)[:, CENTER_INDEX:CENTER_INDEX + 1, :],
                encoded,
            ).squeeze(1)
        if self.token_moe_gate is not None:
            token_gate_features = torch.cat(
                [sequence_encoded, encoded, sequence_encoded - encoded, sequence_encoded * encoded],
                dim=-1,
            )
            token_gate = torch.sigmoid(self.token_moe_gate(token_gate_features))
            encoded = token_gate * encoded + (1.0 - token_gate) * sequence_encoded
        if self.center_token_bias_scale != 0.0:
            center_bias = self.center_token_bias_scale * encoded[:, CENTER_INDEX:CENTER_INDEX + 1, :]
            encoded = encoded + center_bias

        motif_query_context = self.motif_query_proj(motif_context) if self.motif_query_proj is not None else None
        center_local_gate = (
            torch.sigmoid(self.motif_center_local_gate_proj(motif_context))
            if self.motif_center_local_gate_proj is not None
            else None
        )
        if self.latent_readout is not None:
            readout = self.latent_readout(encoded)
        elif self.structure_query_readout is not None:
            readout = self.structure_query_readout(encoded, structure_context=structure_context)
        else:
            structure_task_bias = row_normalize(attn)[:, CENTER_INDEX, :] if (
                self.readout_family == "structure_biased_dpf" and attn is not None
            ) else None
            topk_structure_context = None
            if self.readout_family in {"topk_context_dpf", "gated_topk_context_dpf"} and attn is not None:
                center_weights = row_normalize(attn)[:, CENTER_INDEX, :]
                topk = max(1, min(self.graph_topk, center_weights.shape[-1]))
                _, topk_idx = torch.topk(center_weights, k=topk, dim=-1)
                topk_mask = torch.zeros_like(center_weights)
                topk_mask.scatter_(dim=-1, index=topk_idx, value=1.0)
                topk_weights = row_normalize((center_weights * topk_mask).unsqueeze(1)).squeeze(1)
                topk_structure_context = torch.matmul(topk_weights.unsqueeze(1), encoded).squeeze(1)
            readout = self.readout(
                tokens,
                encoded,
                structure_context=structure_context,
                task_query_context=motif_query_context,
                center_local_gate=center_local_gate,
                structure_task_bias=structure_task_bias,
                topk_structure_context=topk_structure_context,
            )
        if self.structure_reliability_gate is not None and structure_context is not None and attn is not None:
            reliability_features = self._structure_reliability_features(attn)
            reliability_gate = torch.sigmoid(self.structure_reliability_gate(reliability_features))
            readout = reliability_gate * readout + (1.0 - reliability_gate) * sequence_encoded[:, CENTER_INDEX, :]
        if self.structure_fusion_mode == "dual_readout":
            sequence_readout = self.readout(tokens, sequence_encoded, structure_context=None)
            readout = self.dual_readout_fusion(torch.cat([sequence_readout, readout], dim=-1))
        elif self.sequence_residual_scale != 0.0:
            sequence_readout = self.readout(tokens, sequence_encoded, structure_context=None)
            readout = readout + self.sequence_residual_scale * sequence_readout
        if self.center_residual_scale != 0.0:
            readout = readout + self.center_residual_scale * encoded[:, CENTER_INDEX, :]
        if self.structure_context_residual and structure_context is not None:
            residual_context = structure_context
            if self.structure_context_residual_norm is not None:
                residual_context = self.structure_context_residual_norm(residual_context)
            readout = readout + residual_context
        if self.structure_correction is not None and structure_context is not None:
            readout = readout + self.structure_correction(structure_context)
        if self.structure_interaction_proj is not None and structure_context is not None:
            interaction = torch.cat([readout * structure_context, torch.abs(readout - structure_context)], dim=-1)
            readout = readout + self.structure_interaction_proj(interaction)
        if self.center_readout_interaction_proj is not None:
            center_token = encoded[:, CENTER_INDEX, :]
            interaction = torch.cat([readout * center_token, torch.abs(readout - center_token)], dim=-1)
            readout = readout + self.center_readout_interaction_proj(interaction)
        if self.local_global_branch_mixer is not None and structure_context is not None:
            center_branch = encoded[:, CENTER_INDEX, :]
            local_branch = encoded[:, max(0, CENTER_INDEX - 2): CENTER_INDEX + 3, :].mean(dim=1)
            global_branch = encoded.mean(dim=1)
            structure_branch = structure_context
            branch_logits = self.local_global_branch_mixer(
                torch.cat([center_branch, local_branch, global_branch, structure_branch, motif_context], dim=-1)
            )
            branch_weights = torch.softmax(branch_logits, dim=-1)
            stacked = torch.stack([center_branch, local_branch, global_branch, structure_branch], dim=1)
            branch_delta = torch.sum(branch_weights.unsqueeze(-1) * stacked, dim=1)
            readout = readout + self.local_global_branch_out(branch_delta)
        readout = self.readout_scale * readout
        if self.readout_norm is not None:
            readout = self.readout_norm(readout)
        if self.readout_affine_calibration:
            readout = readout * self.readout_calibration_weight + self.readout_calibration_bias
        motif_context = self.motif_context_scale * motif_context
        if self.motif_norm is not None:
            motif_context = self.motif_norm(motif_context)
        if self.motif_affine_calibration:
            motif_context = motif_context * self.motif_calibration_weight + self.motif_calibration_bias
        if self.motif_product_interaction:
            motif_product = readout * self.motif_product_proj(motif_context)
            readout = readout + self.motif_product_out(motif_product)
        if self.motif_residual_gate is not None:
            residual_gate = torch.sigmoid(self.motif_residual_gate(torch.cat([readout, motif_context], dim=-1)))
            readout = readout + self.motif_residual_out(residual_gate * readout)
        if self.motif_fusion_mode == "concat":
            fusion_readout = readout
            fusion_motif = motif_context
            if self.concat_input_norm == "l2":
                fusion_readout = F.normalize(fusion_readout, dim=-1)
                fusion_motif = F.normalize(fusion_motif, dim=-1)
            if self.fusion_input_dropout is not None:
                fusion_readout = self.fusion_input_dropout(fusion_readout)
                fusion_motif = self.fusion_input_dropout(fusion_motif)
            fused = self.motif_fusion(torch.cat([fusion_readout, fusion_motif], dim=-1))
            if self.concat_residual_scale is not None:
                fused = readout + float(self.concat_residual_scale) * fused
        elif self.motif_fusion_mode == "film":
            gamma_raw, beta_raw = self.motif_film(motif_context).chunk(2, dim=-1)
            gamma = 1.0 + self.motif_delta_scale * torch.tanh(gamma_raw)
            beta = self.motif_delta_scale * torch.tanh(beta_raw)
            motif_adjustment = torch.cat([gamma - 1.0, beta], dim=-1)
            self._last_motif_delta = motif_adjustment
            fused = self.film_norm(gamma * readout + beta)
        elif self.motif_fusion_mode == "logit_adapter":
            fused = readout
        else:
            raise ValueError(f"Unsupported motif_fusion_mode: {self.motif_fusion_mode}")

        if self.fused_feature_norm is not None:
            fused = self.fused_feature_norm(fused)
        if self.motif_residual_proj is not None:
            fused = fused + self.motif_residual_proj(motif_context)
        if self.moe_gate is not None and structure_context is not None:
            sequence_expert = self.sequence_expert(sequence_encoded[:, CENTER_INDEX, :])
            structure_expert = self.structure_expert(structure_context)
            fused_expert = self.fused_expert(fused)
            gate_input = torch.cat([fused, motif_context], dim=-1)
            expert_weights = torch.softmax(self.moe_gate(gate_input), dim=-1)
            experts = torch.stack([sequence_expert, structure_expert, fused_expert], dim=1)
            expert_delta = torch.sum(expert_weights.unsqueeze(-1) * experts, dim=1)
            fused = fused + self.moe_out(expert_delta)
        if self.fused_affine_calibration:
            fused = fused * self.fused_calibration_weight + self.fused_calibration_bias
        if self.fused_bias_calibration:
            fused = fused + self.fused_bias
        if self.fused_correction is not None:
            fused = fused + self.fused_correction(fused)
        if self.structure_stats_proj is not None and attn is not None:
            fused = fused + self.structure_stats_proj(self._structure_reliability_features(attn))
        if self.training and self.fused_feature_noise_std > 0.0:
            fused = fused + self.fused_feature_noise_std * torch.randn_like(fused)
        if self.classifier_input_dropout is not None:
            fused = self.classifier_input_dropout(fused)
        classifier_input = fused
        for layer in self.classification[:-1]:
            fused = layer(fused)
        if self.classifier_residual_proj is not None:
            fused = fused + self.classifier_residual_proj(classifier_input)
        embedding = fused
        logits = self.classification[-1](fused)
        if self.logit_residual is not None:
            logits = logits + self.logit_residual_scale * self.logit_residual(fused)
        if self.motif_fusion_mode == "logit_adapter":
            delta = self.motif_delta_scale * self.logit_adapter(torch.cat([readout, motif_context], dim=-1))
            self._last_motif_delta = delta
            logits = logits + delta
        return logits, embedding
