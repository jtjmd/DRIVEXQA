import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VisionTransformer
from transformers import CLIPVisionModel
from typing import Optional, Dict, List
import logging

from pointnet_util import PointNetPlusPlusEncoder


ScenePointEncoder = PointNetPlusPlusEncoder

logger = logging.getLogger(__name__)


class SelfQueryHub(nn.Module):
    
    def __init__(self, hidden_size: int, aux_hidden_size: int = None):
        super().__init__()
        if aux_hidden_size is None:
            aux_hidden_size = hidden_size
            
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(aux_hidden_size, hidden_size)
        self.value_proj = nn.Linear(aux_hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5
        
    def forward(self, rgb_features: torch.Tensor, aux_features: torch.Tensor, aux_mask: torch.Tensor = None) -> torch.Tensor:
        
        B, N_rgb, D = rgb_features.shape
        B_aux, N_aux, D_aux = aux_features.shape
        

        if B != B_aux:
            raise ValueError(f"SQ-Hub batch size mismatch: RGB batch={B} vs Aux batch={B_aux}")
        if D != D_aux:
            raise ValueError(f"SQ-Hub dimension mismatch: RGB dimension={D} vs Aux dimension={D_aux}")
        
        # calculate query, key, value
        query = self.query_proj(rgb_features)  # [B, N_rgb, D]
        key = self.key_proj(aux_features)      # [B, N_aux, D]
        value = self.value_proj(aux_features)  # [B, N_aux, D]
        
        # calculate attention weights
        attn_weights = torch.bmm(query, key.transpose(-2, -1)) * self.scale  # [B, N_rgb, N_aux]
        

        if aux_mask is not None:

            if aux_mask.shape != (B, N_aux):
                raise ValueError(f"Aux mask shape mismatch: expected [{B}, {N_aux}], actual {aux_mask.shape}")
            else:
                mask_expanded = aux_mask.unsqueeze(1).expand(B, N_rgb, N_aux)
                attn_weights = attn_weights.masked_fill(~mask_expanded, float('-inf'))
        
 
        if aux_mask is not None and not aux_mask.any():
            logger.debug("all auxiliary modalities are invalid, returning original RGB features")
            return rgb_features
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # check if attention weights contain nan
        if torch.isnan(attn_weights).any():
            raise ValueError("SQ-Hub attention weights contain NaN")
        
        # apply attention
        attended_features = torch.bmm(attn_weights, value)  # [B, N_rgb, D]
        
        # residual connection
        enhanced_features = rgb_features + attended_features
        
        return enhanced_features


class ParallelPoolingMixer(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:

        B, N, D = features.shape
        
        # convert to convolution format [B, D, N]
        features_conv = features.transpose(1, 2)
        
        # global pooling branch
        global_feat = self.global_pool(features_conv)  # [B, D, 1]
        global_feat = global_feat.expand(-1, -1, N)    # [B, D, N]
        
        # local convolution branch
        local_feat = self.local_conv(features_conv)    # [B, D, N]
        
        # concatenate and fuse
        concat_feat = torch.cat([global_feat, local_feat], dim=1)  # [B, 2*D, N]
        concat_feat = concat_feat.transpose(1, 2)  # [B, N, 2*D]
        
        mixed_features = self.fusion_proj(concat_feat)  # [B, N, D]
        
        return mixed_features


class CAbstractor(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, pooling_size: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.pooling_size = pooling_size
        

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        

        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_tokens)
        

        self.proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
           
        B, N, D = features.shape
        
        
        features_conv = features.transpose(1, 2)  # [B, 768, 49]
        
    
        conv_features = self.conv_layers(features_conv)  # [B, output_dim, 49]
        

        pooled_features = self.adaptive_pool(conv_features)  # [B, output_dim, num_tokens]
        
        
        output_features = pooled_features.transpose(1, 2)  # [B, 64, 4096]
        
       
        output_features = self.proj(output_features)
        
        return output_features


class HoneybeeProjector(nn.Module):

    def __init__(self, input_dim: int = 768, output_dim: int = 4096, num_tokens: int = 64, 
                 num_layers: int = 2, pooling_size: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        
        
        self.abstractors = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:

                abstractor = CAbstractor(current_dim, output_dim, num_tokens, pooling_size)
            else:

                next_dim = current_dim + (output_dim - input_dim) // num_layers
                abstractor = CAbstractor(current_dim, next_dim, num_tokens, pooling_size)
                current_dim = next_dim
            
            self.abstractors.append(abstractor)
        

        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
            

        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
       
        x = features
        

        for abstractor in self.abstractors:
            x = abstractor(x)
        

        if self.residual_proj is not None and features.shape[1] == x.shape[1]:

            residual = self.residual_proj(features)
            x = x + residual
        
        # Layer normalization
        x = self.norm(x)
        
        return x


class PartialEncoder(nn.Module):

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.partial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

        self.local_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
       
        encoded_features = self.partial_encoder(features)  # [B, 49, 768]
        

        attended_features, _ = self.local_attention(
            encoded_features, encoded_features, encoded_features
        )  # [B, 49, 768]
        

        partial_features = self.norm(encoded_features + attended_features)
        
        return partial_features


class GlobalEncoder(nn.Module):

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        

        self.global_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        

        self.global_to_local_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
       
        B, N, D = features.shape
        

        global_feature = features.mean(dim=1, keepdim=True)  # [B, 1, 768]
        

        global_context = self.global_mlp(global_feature)  # [B, 1, 768]
        

        global_broadcasted = global_context.expand(B, N, D)  # [B, 49, 768]
        

        attended_features, _ = self.global_to_local_attention(
            features, global_broadcasted, global_broadcasted
        )  # [B, 49, 768]
        
        global_features = self.norm(features + attended_features)
        
        return global_features


class ParGoProjector(nn.Module):

    def __init__(self, input_dim: int = 768, output_dim: int = 4096, num_tokens: int = 64,
                 partial_layers: int = 2, global_layers: int = 2, fusion_dim: int = 512, 
                 temperature: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.temperature = temperature
        
        self.partial_encoder = PartialEncoder(input_dim, fusion_dim, partial_layers)
        self.global_encoder = GlobalEncoder(input_dim, fusion_dim, global_layers)
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.feature_projector = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, output_dim)
        )
        
        if num_tokens != 49:  # 49是7x7 patches的默认数量
            self.adaptive_pool = nn.AdaptiveAvgPool1d(num_tokens)
        else:
            self.adaptive_pool = None
            
        self.final_norm = nn.LayerNorm(output_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:

        B, N, D = features.shape
        
        partial_features = self.partial_encoder(features)  # [B, 49, 768]
        global_features = self.global_encoder(features)    # [B, 49, 768]
        
        fused_features, attention_weights = self.fusion_attention(
            partial_features, 
            global_features,   
            global_features    
        )  # [B, 49, 768]
        
        fused_features = fused_features + partial_features
        
        projected_features = self.feature_projector(fused_features)  # [B, 49, output_dim]
        
        if self.adaptive_pool is not None:
            projected_features = projected_features.transpose(1, 2)
            projected_features = self.adaptive_pool(projected_features)
            projected_features = projected_features.transpose(1, 2)
        
        enhanced_features = self.final_norm(projected_features)
        
        return enhanced_features


class CrossAttentionModule(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        B, N_q, D = query.shape
        B, N_k, D = key.shape
        

        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_q, d]
        k = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, N_k, d]
        v = self.v_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_k, d]
        

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N_q, N_k]
        

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N_k]
            attention_scores.masked_fill_(~mask_expanded, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        

        context = torch.matmul(attention_probs, v)  # [B, H, N_q, d]
        context = context.transpose(1, 2).contiguous().view(B, N_q, D)  # [B, N_q, D]
        

        output = self.out_proj(context)
        return output


class MultiViewFusionEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.target_size = 224
        
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        self.modality_to_hidden = nn.Linear(768, config.multiview_hidden_dim)
        
        self.rgb_spatial_align = nn.Linear(49, 48)     
        self.depth_spatial_align = nn.Linear(49, 48)   
        self.event_spatial_align = nn.Linear(49, 48)   
        self.spatial_align = nn.Linear(96, 48)         
        
        self.spatial_attention = CrossAttentionModule(
            hidden_dim=config.multiview_hidden_dim,  # 512
            num_heads=config.multiview_spatial_heads,
            dropout=config.multiview_dropout
        )

        self.channel_attention = CrossAttentionModule(
            hidden_dim=48,  
            num_heads=4,    
            dropout=config.multiview_dropout
        )
        
        self.final_proj = nn.Linear(config.multiview_hidden_dim, config.multiview_hidden_dim)
        self.layer_norm = nn.LayerNorm(config.multiview_hidden_dim)
        
        self.fusion_variant = getattr(config, 'fusion_variant', 'gap')
        if self.fusion_variant in ['qattn', 'qattn_spectral', 'qattn_depthgate']:
            num_queries = 3 if getattr(config, 'fusion_token_layout', 'single') == 'triple' else 1
            self.pool_query = nn.Parameter(torch.randn(1, num_queries, config.multiview_hidden_dim))
            self.pool_attn = CrossAttentionModule(
                hidden_dim=config.multiview_hidden_dim,
                num_heads=config.multiview_spatial_heads,
                dropout=config.multiview_dropout
            )
        if self.fusion_variant == 'qattn_spectral':
            self.spectral_conv = nn.Conv1d(
                in_channels=config.multiview_hidden_dim,
                out_channels=config.multiview_hidden_dim,
                kernel_size=3,
                padding=1,
                groups=config.multiview_hidden_dim
            )
        if self.fusion_variant == 'qattn_depthgate':
            self.depth_gate = nn.Linear(config.multiview_hidden_dim, 1)
        
    def opencv_grid_fusion(self, views: torch.Tensor) -> torch.Tensor:
        if views.shape[1] != 4:
            raise ValueError(f"Expected 4 views, got {views.shape[1]}")
            
        B, N, C, H, W = views.shape
        
        #  [FRONT|BACK; LEFT|RIGHT] pattern
        front, back, left, right = views[:, 0], views[:, 1], views[:, 2], views[:, 3]
        

        top_row = torch.cat([front, back], dim=-1)  # [B, C, H, 2W]
        

        bottom_row = torch.cat([left, right], dim=-1)  # [B, C, H, 2W]
        

        fused = torch.cat([top_row, bottom_row], dim=-2)  # [B, C, 2H, 2W]
        
        return fused
        
    def forward(self, images):
        B = images.shape[0]
        

        rgb_views = images[:, 0:4]  # [B, 4, 3, 224, 224]
        depth_views = images[:, 4:8]  # [B, 4, 3, 224, 224]
        event_views = images[:, 8:12]  # [B, 4, 3, 224, 224]
        
        rgb_fused = self.opencv_grid_fusion(rgb_views)      # [B, 3, 448, 448]
        depth_fused = self.opencv_grid_fusion(depth_views)  # [B, 3, 448, 448]
        event_fused = self.opencv_grid_fusion(event_views)  # [B, 3, 448, 448]
        
        resized_rgb = F.interpolate(rgb_fused, size=(224, 224), mode='bilinear', align_corners=False)
        
        if torch.isnan(resized_rgb).any():
            logger.warning("found NaN in resized_rgb")
        if torch.isinf(resized_rgb).any():
            logger.warning("found Inf in resized_rgb")
        if (resized_rgb.abs() > 100).any():
            logger.warning(f"found large values in resized_rgb, max: {resized_rgb.max()}, min: {resized_rgb.min()}")
        
        clip_output_rgb = self.vision_model(pixel_values=resized_rgb)
        rgb_features = clip_output_rgb.last_hidden_state[:, 1:]  # [B, 49, 768]
        
        resized_depth = F.interpolate(depth_fused, size=(224, 224), mode='bilinear', align_corners=False)
        
        if torch.isnan(resized_depth).any():
            logger.warning("found NaN in resized_depth")
        if torch.isinf(resized_depth).any():
            logger.warning("found Inf in resized_depth")
        if (resized_depth.abs() > 100).any():
            logger.warning(f"found large values in resized_depth, max: {resized_depth.max()}, min: {resized_depth.min()}")
        
        clip_output_depth = self.vision_model(pixel_values=resized_depth)
        depth_features = clip_output_depth.last_hidden_state[:, 1:]  # [B, 49, 768]
        
        resized_event = F.interpolate(event_fused, size=(224, 224), mode='bilinear', align_corners=False)
        
        if torch.isnan(resized_event).any():
            logger.warning("found NaN in resized_event")
        if torch.isinf(resized_event).any():
            logger.warning("found Inf in resized_event")
        if (resized_event.abs() > 100).any():
            logger.warning(f"found large values in resized_event, max: {resized_event.max()}, min: {resized_event.min()}")
        
        clip_output_event = self.vision_model(pixel_values=resized_event)
        event_features = clip_output_event.last_hidden_state[:, 1:]  # [B, 49, 768]
        
        rgb_hidden = self.modality_to_hidden(rgb_features)      # [B, 49, 512]
        depth_hidden = self.modality_to_hidden(depth_features)  # [B, 49, 512]
        event_hidden = self.modality_to_hidden(event_features)  # [B, 49, 512]
        
        if torch.isnan(rgb_features).any():
            logger.warning("found NaN in RGB ViT features")
        if torch.isnan(depth_features).any():
            logger.warning("found NaN in Depth ViT features")
        if torch.isnan(event_features).any():
            logger.warning("found NaN in Event ViT features")
            
        if torch.isnan(rgb_hidden).any():
            logger.warning("found NaN in RGB projection")
        if torch.isnan(depth_hidden).any():
            logger.warning("found NaN in Depth projection")
        if torch.isnan(event_hidden).any():
            logger.warning("found NaN in Event projection")
        
        rgb_aligned = self.rgb_spatial_align(
            rgb_hidden.transpose(1, 2)  # [B, 512, 49] -> [B, 512, 48]
        ).transpose(1, 2)  # [B, 48, 512]
        
        depth_aligned = self.depth_spatial_align(  
            depth_hidden.transpose(1, 2)  # [B, 512, 49] -> [B, 512, 48]
        ).transpose(1, 2)  # [B, 48, 512]
        
        event_aligned = self.event_spatial_align(  
            event_hidden.transpose(1, 2)  # [B, 512, 49] -> [B, 512, 48]
        ).transpose(1, 2)  # [B, 48, 512]
        
        if torch.isnan(rgb_aligned).any():
            logger.warning("found NaN in RGB aligned")
        if torch.isnan(depth_aligned).any():
            logger.warning("found NaN in Depth aligned")
        if torch.isnan(event_aligned).any():
            logger.warning("found NaN in Event aligned")
        
        depth_event_kv = torch.cat([depth_aligned, event_aligned], dim=1)  # [B, 96, 512] (48+48)
        
        if torch.isnan(rgb_aligned).any() or torch.isnan(depth_event_kv).any():
            logger.warning("found NaN in spatial attention input")
            
        spatial_cross_output = self.spatial_attention(
            query=rgb_aligned,           
            key=depth_event_kv,         
            value=depth_event_kv        
        )  
        
        rgb_channel = rgb_aligned.transpose(1, 2)           # [B, 512, 48]
        depth_event_channel = depth_event_kv.transpose(1, 2)  # [B, 512, 96]
        
        depth_event_channel_aligned = self.spatial_align(depth_event_channel)  # [B, 512, 96] -> [B, 512, 48]
        
        if torch.isnan(rgb_channel).any() or torch.isnan(depth_event_channel_aligned).any():
            logger.warning("found NaN in channel attention input")
        
        channel_cross_output = self.channel_attention(
            query=rgb_channel,                    
            key=depth_event_channel_aligned,     
            value=depth_event_channel_aligned    
        )  
        
        channel_cross_output = channel_cross_output.transpose(1, 2)  # [B, 48, 512]
        
        alpha = 0.5  
        fused_tokens = alpha * spatial_cross_output + (1 - alpha) * channel_cross_output  # [B, 48, 512]

        if torch.isnan(fused_tokens).any():
            logger.warning("found NaN in fusion step")
            logger.warning(f"spatial_cross_output has nan: {torch.isnan(spatial_cross_output).any()}")
            logger.warning(f"channel_cross_output has nan: {torch.isnan(channel_cross_output).any()}")
        
        if self.fusion_variant == 'gap':
            pooled = fused_tokens.mean(dim=1)  # [B, 512]
            if getattr(self.config, 'fusion_token_layout', 'single') == 'triple':
                pooled = pooled.unsqueeze(1).expand(-1, 3, -1)  # [B, 3, 512]
        else:
            kv_tokens = fused_tokens  # [B, 48, 512]
            if self.fusion_variant == 'qattn_spectral':
                smoothed = self.spectral_conv(kv_tokens.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
                kv_tokens = torch.cat([kv_tokens, smoothed], dim=1)  # [B, 96, 512]
            if self.fusion_variant == 'qattn_depthgate':
                depth_score = torch.sigmoid(self.depth_gate(depth_aligned))  # [B, 48, 1]
                kv_tokens = kv_tokens * depth_score  
            query = self.pool_query.expand(kv_tokens.size(0), -1, -1)  # [B, Q, 512]
            pooled = self.pool_attn(query=query, key=kv_tokens, value=kv_tokens)  # [B, Q, 512]
            if pooled.dim() == 3 and pooled.shape[1] == 1:
                pooled = pooled[:, 0, :]  # [B, 512]
        
        fusion_output = self.final_proj(pooled)  # [..., 512]
        fusion_output = self.layer_norm(fusion_output)   # [..., 512]
        
        if getattr(self.config, 'fusion_token_layout', 'single') == 'triple':
            pass
        
        if torch.isnan(fusion_output).any():
            logger.warning("found NaN in final output")
        
        return fusion_output


class MultiViewAblationEncoder(nn.Module):
 
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.target_size = 224
        
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        self.modality_to_hidden = nn.Linear(768, config.multiview_hidden_dim)
        
        self.rgb_spatial_align = nn.Linear(49, 48)     # RGB: 49->48  
        self.depth_spatial_align = nn.Linear(49, 48)   
        self.event_spatial_align = nn.Linear(49, 48)   
        
        self.spatial_align_96to48 = nn.Linear(96, 48)  
        self.spatial_align_144to48 = nn.Linear(144, 48) 
        
        self.spatial_attention = CrossAttentionModule(
            hidden_dim=config.multiview_hidden_dim,  # 512
            num_heads=config.multiview_spatial_heads,
            dropout=config.multiview_dropout
        )
        
        self.channel_attention = CrossAttentionModule(
            hidden_dim=48,  
            num_heads=4,   
            dropout=config.multiview_dropout
        )
        
        self.final_proj = nn.Linear(config.multiview_hidden_dim, config.multiview_hidden_dim)
        self.layer_norm = nn.LayerNorm(config.multiview_hidden_dim)
        
    def opencv_grid_fusion(self, views: torch.Tensor) -> torch.Tensor:
        if views.shape[1] != 4:
            raise ValueError(f"Expected 4 views, got {views.shape[1]}")
            
        B, N, C, H, W = views.shape
        
        front, back, left, right = views[:, 0], views[:, 1], views[:, 2], views[:, 3]
        
        top_row = torch.cat([front, back], dim=-1)  # [B, C, H, 2W]
        
        bottom_row = torch.cat([left, right], dim=-1)  # [B, C, H, 2W]
        
        fused = torch.cat([top_row, bottom_row], dim=-2)  # [B, C, 2H, 2W]
        
        return fused
    
    def _determine_query_modality(self, available_modalities: List[str]) -> str:
        priority = self.config.ablation_query_priority
        for modality in priority:
            if modality in available_modalities:
                return modality
        raise ValueError(f"No valid query modality found in {available_modalities}")
    
    def forward(self, modality_images: Dict[str, torch.Tensor]):

        available_modalities = [k for k, v in modality_images.items() if v is not None]
        
        if len(available_modalities) == 0:
            raise ValueError("At least one modality must be provided")
        
        B = next(iter(modality_images.values())).shape[0]
        
        modality_features = {}
        
        for modality, images in modality_images.items():
            if images is None:
                continue
                
            fused_img = self.opencv_grid_fusion(images)  # [B, 3, 448, 448]
            
            resized_img = F.interpolate(fused_img, size=(224, 224), mode='bilinear', align_corners=False)
            
            vision_outputs = self.vision_model(pixel_values=resized_img)
            patch_embeddings = vision_outputs.last_hidden_state  # [B, 50, 768] 
            
            patch_features = patch_embeddings[:, 1:]  # [B, 49, 768]
            
            hidden_features = self.modality_to_hidden(patch_features)  # [B, 49, 512]
            
            if modality == 'rgb':
                aligned_features = self.rgb_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            elif modality == 'depth':
                aligned_features = self.depth_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            elif modality == 'event':
                aligned_features = self.event_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            else:
                raise ValueError(f"Unknown modality: {modality}")
                
            modality_features[modality] = aligned_features
        
        if len(available_modalities) == 1:
            single_modality = available_modalities[0]
            fusion_output = modality_features[single_modality]  # [B, 48, 512]
            
            fusion_output = fusion_output.mean(dim=1)  # [B, 512]
            
        else:
            query_modality = self._determine_query_modality(available_modalities)
            kv_modalities = [m for m in available_modalities if m != query_modality]
            
            query_features = modality_features[query_modality]  # [B, 48, 512]
            
            kv_features_list = [modality_features[m] for m in kv_modalities]
            kv_features = torch.cat(kv_features_list, dim=1)  # [B, 48*N, 512]
            
            spatial_cross_output = self.spatial_attention(
                query=query_features,  # [B, 48, 512]
                key=kv_features,      # [B, 48*N, 512]
                value=kv_features     # [B, 48*N, 512]
            )  # [B, 48, 512]
            
            query_channel = query_features.transpose(1, 2)  # [B, 512, 48]
            kv_channel = kv_features.transpose(1, 2)        # [B, 512, 48*N]
            
            if kv_channel.shape[2] == 96:  
                kv_channel_aligned = self.spatial_align_96to48(kv_channel)  # [B, 512, 48]
            elif kv_channel.shape[2] == 144:  
                kv_channel_aligned = self.spatial_align_144to48(kv_channel)  # [B, 512, 48]
            else:
                kv_channel_aligned = kv_channel  
            
            channel_cross_output = self.channel_attention(
                query=query_channel,           # [B, 512, 48]
                key=kv_channel_aligned,       # [B, 512, 48]
                value=kv_channel_aligned      # [B, 512, 48]
            )  # [B, 512, 48]
            
            channel_cross_output = channel_cross_output.transpose(1, 2)  # [B, 48, 512]
            
            alpha = 0.5
            fusion_output = alpha * spatial_cross_output + (1 - alpha) * channel_cross_output  # [B, 48, 512]
            
            fusion_output = fusion_output.mean(dim=1)  # [B, 512]
        
        fusion_output = self.final_proj(fusion_output)
        fusion_output = self.layer_norm(fusion_output)
        
        return fusion_output


class MultiViewFusionHoneybeeEncoder(nn.Module):
    
    def __init__(self, config, output_dim: int = 4096):
        super().__init__()
        self.config = config
        
        self.target_size = 224
        
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        self.modality_to_hidden = nn.Linear(768, config.multiview_hidden_dim)
        
        self.rgb_spatial_align = nn.Linear(49, 48)      
        self.depth_spatial_align = nn.Linear(49, 48)   
        self.event_spatial_align = nn.Linear(49, 48)   
        self.spatial_align_96to48 = nn.Linear(96, 48)  
        self.spatial_align_144to48 = nn.Linear(144, 48) 
        
        self.spatial_attention = CrossAttentionModule(
            hidden_dim=config.multiview_hidden_dim,  # 512
            num_heads=config.multiview_spatial_heads,
            dropout=config.multiview_dropout
        )

        self.channel_attention = CrossAttentionModule(
            hidden_dim=48,  
            num_heads=4,   
            dropout=config.multiview_dropout
        )
        
        self.honeybee_projector = HoneybeeProjector(
            input_dim=config.multiview_hidden_dim,  # 512
            output_dim=output_dim,                  
            num_tokens=config.honeybee_num_tokens,   
            num_layers=config.honeybee_c_abs_layers, 
            pooling_size=config.honeybee_pooling_size 
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def opencv_grid_fusion(self, views: torch.Tensor) -> torch.Tensor:
      
        B, N, C, H, W = views.shape
        assert N == 4, f"expected 4 views, got {N} views"
        
        front = views[:, 0]  
        back = views[:, 1]   
        left = views[:, 2]   
        right = views[:, 3]  
        
        top_row = torch.cat([front, back], dim=-1)    # [B, C, H, 2*W]
        bottom_row = torch.cat([left, right], dim=-1) # [B, C, H, 2*W]
        fused_image = torch.cat([top_row, bottom_row], dim=-2)  # [B, C, 2*H, 2*W]
        
        return fused_image
    
    def forward(self, modality_images: Dict[str, torch.Tensor]) -> torch.Tensor:
      
        modality_features = {}
        available_modalities = [k for k, v in modality_images.items() if v is not None]
        
        for modality, images in modality_images.items():
            if images is None:
                continue
                
            fused_img = self.opencv_grid_fusion(images)  # [B, 3, 448, 448]
            
            resized_img = F.interpolate(fused_img, size=(224, 224), mode='bilinear', align_corners=False)
            
            vision_outputs = self.vision_model(pixel_values=resized_img)
            patch_embeddings = vision_outputs.last_hidden_state  # [B, 50, 768] 
            
            patch_features = patch_embeddings[:, 1:]  # [B, 49, 768]
            
            hidden_features = self.modality_to_hidden(patch_features)  # [B, 49, 512]
            
            if modality == 'rgb':
                aligned_features = self.rgb_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            elif modality == 'depth':
                aligned_features = self.depth_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            elif modality == 'event':
                aligned_features = self.event_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            else:
                raise ValueError(f"Unknown modality: {modality}")
                
            modality_features[modality] = aligned_features
        
        if len(available_modalities) == 1:
            single_modality = available_modalities[0]
            fusion_output = modality_features[single_modality]  # [B, 48, 512]
            
        elif set(available_modalities) == {'rgb', 'depth', 'event'}:
            query_features = modality_features['rgb']  # [B, 48, 512]
            kv_features = torch.cat([modality_features['depth'], modality_features['event']], dim=1)  # [B, 96, 512]
            
            spatial_cross_output = self.spatial_attention(query_features, kv_features, kv_features)  # [B, 48, 512]
            
            query_transposed = query_features.transpose(1, 2)  # [B, 512, 48]
            kv_transposed = kv_features.transpose(1, 2)  # [B, 512, 96]
            kv_aligned = self.spatial_align_96to48(kv_transposed)  # [B, 512, 48]
            
            channel_cross_output = self.channel_attention(query_transposed, kv_aligned, kv_aligned)  # [B, 512, 48]
            channel_cross_output = channel_cross_output.transpose(1, 2)  # [B, 48, 512]
            
            alpha = 0.5
            fusion_output = alpha * spatial_cross_output + (1 - alpha) * channel_cross_output  # [B, 48, 512]
            
        elif set(available_modalities) == {'rgb', 'depth'}:
            query_features = modality_features['rgb']  # [B, 48, 512]
            kv_features = modality_features['depth']  # [B, 48, 512]
            
            spatial_cross_output = self.spatial_attention(query_features, kv_features, kv_features)  # [B, 48, 512]
            fusion_output = spatial_cross_output
            
        elif set(available_modalities) == {'rgb', 'event'}:
            query_features = modality_features['rgb']  # [B, 48, 512]
            kv_features = modality_features['event']  # [B, 48, 512]
            
            spatial_cross_output = self.spatial_attention(query_features, kv_features, kv_features)  # [B, 48, 512]
            fusion_output = spatial_cross_output
            
        elif set(available_modalities) == {'depth', 'event'}:
            query_features = modality_features['depth']  # [B, 48, 512]
            kv_features = modality_features['event']  # [B, 48, 512]
            
            spatial_cross_output = self.spatial_attention(query_features, kv_features, kv_features)  # [B, 48, 512]
            fusion_output = spatial_cross_output
            
        else:
            raise ValueError(f"Unsupported modality combination: {available_modalities}")
        
        pooled_features = fusion_output.mean(dim=1, keepdim=True)  # [B, 1, 512]
        
        expanded_features = pooled_features.expand(-1, 49, -1)  # [B, 49, 512]

        honeybee_output = self.honeybee_projector(expanded_features)  # [B, 64, 4096]
        
        final_output = honeybee_output.mean(dim=1)  # [B, 4096]
        
        final_output = self.layer_norm(final_output)
        
        return final_output


class MultiViewFusionPargoEncoder(nn.Module):
    
    def __init__(self, config, output_dim: int = 4096):
        super().__init__()
        self.config = config
        
        self.target_size = 224
        
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        self.modality_to_hidden = nn.Linear(768, config.multiview_hidden_dim)
        
        self.rgb_spatial_align = nn.Linear(49, 48)    
        self.depth_spatial_align = nn.Linear(49, 48)  
        self.event_spatial_align = nn.Linear(49, 48)  
        self.spatial_align_96to48 = nn.Linear(96, 48) 
        self.spatial_align_144to48 = nn.Linear(144, 48) 
        
        self.spatial_attention = CrossAttentionModule(
            hidden_dim=config.multiview_hidden_dim,  # 512
            num_heads=config.multiview_spatial_heads,
            dropout=config.multiview_dropout
        )
        
        self.channel_attention = CrossAttentionModule(
            hidden_dim=48,  
            num_heads=4,   
            dropout=config.multiview_dropout
        )
        
        self.pargo_projector = ParGoProjector(
            input_dim=config.multiview_hidden_dim,   # 512
            output_dim=output_dim,                  
            num_tokens=config.pargo_num_tokens,     
            partial_layers=config.pargo_partial_layers, 
            global_layers=config.pargo_global_layers,  
            fusion_dim=config.pargo_fusion_dim,     
            temperature=config.pargo_temperature    
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def opencv_grid_fusion(self, views: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = views.shape
        assert N == 4, f"expected 4 views, got {N} views"
        
        front = views[:, 0]  
        back = views[:, 1]   
        left = views[:, 2]   
        right = views[:, 3]  
        
        top_row = torch.cat([front, back], dim=-1)   
        bottom_row = torch.cat([left, right], dim=-1) 
        fused_image = torch.cat([top_row, bottom_row], dim=-2) 
        
        return fused_image
    
    def forward(self, modality_images: Dict[str, torch.Tensor]) -> torch.Tensor:
      
        modality_features = {}
        available_modalities = [k for k, v in modality_images.items() if v is not None]
        
        for modality, images in modality_images.items():
            if images is None:
                continue
                
            fused_img = self.opencv_grid_fusion(images)  # [B, 3, 448, 448]
            
            resized_img = F.interpolate(fused_img, size=(224, 224), mode='bilinear', align_corners=False)
            
            vision_outputs = self.vision_model(pixel_values=resized_img)
            patch_embeddings = vision_outputs.last_hidden_state  # [B, 50, 768] 
            
            patch_features = patch_embeddings[:, 1:]  # [B, 49, 768]
            
            hidden_features = self.modality_to_hidden(patch_features)  # [B, 49, 512]
            
            if modality == 'rgb':
                aligned_features = self.rgb_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            elif modality == 'depth':
                aligned_features = self.depth_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            elif modality == 'event':
                aligned_features = self.event_spatial_align(hidden_features.transpose(1, 2)).transpose(1, 2)  # [B, 48, 512]
            else:
                raise ValueError(f"Unknown modality: {modality}")
                
            modality_features[modality] = aligned_features
        
        if len(available_modalities) == 1:
            single_modality = available_modalities[0]
            fusion_output = modality_features[single_modality]  # [B, 48, 512]
            
        elif set(available_modalities) == {'rgb', 'depth', 'event'}:
            query_features = modality_features['rgb']  # [B, 48, 512]
            kv_features = torch.cat([modality_features['depth'], modality_features['event']], dim=1)  # [B, 96, 512]
            
            spatial_cross_output = self.spatial_attention(query_features, kv_features, kv_features)  # [B, 48, 512]
            
            query_transposed = query_features.transpose(1, 2)  # [B, 512, 48]
            kv_transposed = kv_features.transpose(1, 2)  # [B, 512, 96]
            kv_aligned = self.spatial_align_96to48(kv_transposed)  # [B, 512, 48]
            
            channel_cross_output = self.channel_attention(query_transposed, kv_aligned, kv_aligned)  # [B, 512, 48]
            channel_cross_output = channel_cross_output.transpose(1, 2)  # [B, 48, 512]
            
            alpha = 0.5
            fusion_output = alpha * spatial_cross_output + (1 - alpha) * channel_cross_output  # [B, 48, 512]
            
        elif set(available_modalities) == {'rgb', 'depth'}:
            query_features = modality_features['rgb']  # [B, 48, 512]
            kv_features = modality_features['depth']  # [B, 48, 512]
            
            spatial_cross_output = self.spatial_attention(query_features, kv_features, kv_features)  # [B, 48, 512]
            fusion_output = spatial_cross_output
            
        elif set(available_modalities) == {'rgb', 'event'}:
            query_features = modality_features['rgb']  # [B, 48, 512]
            kv_features = modality_features['event']  # [B, 48, 512]
            
            spatial_cross_output = self.spatial_attention(query_features, kv_features, kv_features)  # [B, 48, 512]
            fusion_output = spatial_cross_output
            
        elif set(available_modalities) == {'depth', 'event'}:
            query_features = modality_features['depth']  # [B, 48, 512]
            kv_features = modality_features['event']  # [B, 48, 512]
            
            spatial_cross_output = self.spatial_attention(query_features, kv_features, kv_features)  # [B, 48, 512]
            fusion_output = spatial_cross_output
            
        else:
            raise ValueError(f"Unsupported modality combination: {available_modalities}")
        
        pooled_features = fusion_output.mean(dim=1, keepdim=True)  # [B, 1, 512]
        
        expanded_features = pooled_features.expand(-1, 49, -1)  # [B, 49, 512]
        
        pargo_output = self.pargo_projector(expanded_features)  # [B, 64, 4096]
        
        final_output = pargo_output.mean(dim=1)  # [B, 4096]
        
        final_output = self.layer_norm(final_output)
        
        return final_output


class UnifiedVisionEncoder(nn.Module):
    
    def __init__(self, output_dim: int, config=None):
        super().__init__()
        self.config = config
        self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_proj = nn.Linear(self.clip_vision.config.hidden_size, output_dim)
       
        clip_hidden_size = self.clip_vision.config.hidden_size
        
        self.rgb_positional_embedding = nn.Parameter(torch.randn(4, clip_hidden_size))
        self.depth_positional_embedding = nn.Parameter(torch.randn(4, clip_hidden_size)) 
        self.event_positional_embedding = nn.Parameter(torch.randn(4, clip_hidden_size)) 
        
        self.scene_encoder = ScenePointEncoder(output_dim=output_dim)
        
        if config and config.model_type == 'cmnext':
            self.use_cmnext = True
            if config.use_sq_hub:
                self.depth_sq_hub = SelfQueryHub(output_dim, output_dim)
                self.event_sq_hub = SelfQueryHub(output_dim, output_dim)
            
            if config.use_ppx:
                self.rgb_ppx = ParallelPoolingMixer(output_dim)
                self.depth_ppx = ParallelPoolingMixer(output_dim)
                self.event_ppx = ParallelPoolingMixer(output_dim)
                
            self.aux_weight = config.aux_modality_weight
        else:
            self.use_cmnext = False
        
        if config and config.model_type == 'honeybee':
            self.use_honeybee = True
            self.honeybee_projectors = nn.ModuleDict()
            
            if getattr(config, 'enable_multiview_fusion', False):
                if 'rgb' in config.enabled_modalities:
                    self.honeybee_projectors['rgb_fused'] = HoneybeeProjector(
                        input_dim=clip_hidden_size,  
                        output_dim=output_dim,       
                        num_tokens=config.honeybee_num_tokens,  
                        num_layers=config.honeybee_c_abs_layers,  
                        pooling_size=config.honeybee_pooling_size  
                    )
                
                if 'depth' in config.enabled_modalities:
                    self.honeybee_projectors['depth_fused'] = HoneybeeProjector(
                        input_dim=clip_hidden_size,  
                        output_dim=output_dim,       
                        num_tokens=config.honeybee_num_tokens,  
                        num_layers=config.honeybee_c_abs_layers,  
                        pooling_size=config.honeybee_pooling_size  
                    )
                
                if 'event' in config.enabled_modalities:
                    self.honeybee_projectors['event_fused'] = HoneybeeProjector(
                        input_dim=clip_hidden_size,  
                        output_dim=output_dim,       
                        num_tokens=config.honeybee_num_tokens,  
                        num_layers=config.honeybee_c_abs_layers,  
                        pooling_size=config.honeybee_pooling_size  
                    )
            else:
                if 'rgb' in config.enabled_modalities:
                    for view_name in ['FRONT', 'BACK', 'LEFT', 'RIGHT']:
                        self.honeybee_projectors[f'rgb_{view_name}'] = HoneybeeProjector(
                            input_dim=clip_hidden_size,  
                            output_dim=output_dim,       
                            num_tokens=config.honeybee_num_tokens,  
                            num_layers=config.honeybee_c_abs_layers,  
                            pooling_size=config.honeybee_pooling_size  
                        )
                
                if 'depth' in config.enabled_modalities:
                    for view_name in ['FRONT', 'BACK', 'LEFT', 'RIGHT']:
                        self.honeybee_projectors[f'depth_{view_name}'] = HoneybeeProjector(
                            input_dim=clip_hidden_size,  
                            output_dim=output_dim,       
                            num_tokens=config.honeybee_num_tokens,  
                            num_layers=config.honeybee_c_abs_layers,  
                            pooling_size=config.honeybee_pooling_size  
                        )
                
                if 'event' in config.enabled_modalities:
                    for view_name in ['FRONT', 'BACK', 'LEFT', 'RIGHT']:
                        self.honeybee_projectors[f'event_{view_name}'] = HoneybeeProjector(
                            input_dim=clip_hidden_size,  
                            output_dim=output_dim,       
                            num_tokens=config.honeybee_num_tokens,  
                            num_layers=config.honeybee_c_abs_layers,  
                            pooling_size=config.honeybee_pooling_size  
                        )
        else:
            self.use_honeybee = False
            
        if config and config.model_type == 'pargo':
            self.use_pargo = True
            self.pargo_projectors = nn.ModuleDict()
            
            if getattr(config, 'enable_multiview_fusion', False):
                if 'rgb' in config.enabled_modalities:
                    self.pargo_projectors['rgb_fused'] = ParGoProjector(
                        input_dim=clip_hidden_size,  
                        output_dim=output_dim,       
                        num_tokens=config.pargo_num_tokens,  
                        partial_layers=config.pargo_partial_layers,  
                        global_layers=config.pargo_global_layers,  
                        fusion_dim=config.pargo_fusion_dim,  
                        temperature=config.pargo_temperature  
                    )
                
                if 'depth' in config.enabled_modalities:
                    self.pargo_projectors['depth_fused'] = ParGoProjector(
                        input_dim=clip_hidden_size,  
                        output_dim=output_dim,       
                        num_tokens=config.pargo_num_tokens,  
                        partial_layers=config.pargo_partial_layers,  
                        global_layers=config.pargo_global_layers,  
                        fusion_dim=config.pargo_fusion_dim,  
                        temperature=config.pargo_temperature  
                    )
                
                if 'event' in config.enabled_modalities:
                    self.pargo_projectors['event_fused'] = ParGoProjector(
                        input_dim=clip_hidden_size,  
                        output_dim=output_dim,       
                        num_tokens=config.pargo_num_tokens,  
                        partial_layers=config.pargo_partial_layers,  
                        global_layers=config.pargo_global_layers,  
                        fusion_dim=config.pargo_fusion_dim,  
                        temperature=config.pargo_temperature  
                    )
            else:
                if 'rgb' in config.enabled_modalities:
                    for view_name in ['FRONT', 'BACK', 'LEFT', 'RIGHT']:
                        self.pargo_projectors[f'rgb_{view_name}'] = ParGoProjector(
                            input_dim=clip_hidden_size,  
                            output_dim=output_dim,       
                            num_tokens=config.pargo_num_tokens,  
                            partial_layers=config.pargo_partial_layers,  
                            global_layers=config.pargo_global_layers,  
                            fusion_dim=config.pargo_fusion_dim,  
                            temperature=config.pargo_temperature  
                        )
                
                if 'depth' in config.enabled_modalities:
                    for view_name in ['FRONT', 'BACK', 'LEFT', 'RIGHT']:
                        self.pargo_projectors[f'depth_{view_name}'] = ParGoProjector(
                            input_dim=clip_hidden_size,  
                            output_dim=output_dim,       
                            num_tokens=config.pargo_num_tokens,  
                            partial_layers=config.pargo_partial_layers,  
                            global_layers=config.pargo_global_layers,  
                            fusion_dim=config.pargo_fusion_dim,  
                            temperature=config.pargo_temperature  
                        )
                
                if 'event' in config.enabled_modalities:
                    for view_name in ['FRONT', 'BACK', 'LEFT', 'RIGHT']:
                        self.pargo_projectors[f'event_{view_name}'] = ParGoProjector(
                            input_dim=clip_hidden_size,  
                            output_dim=output_dim,       
                            num_tokens=config.pargo_num_tokens,  
                            partial_layers=config.pargo_partial_layers,  
                            global_layers=config.pargo_global_layers,  
                            fusion_dim=config.pargo_fusion_dim,  
                            temperature=config.pargo_temperature  
                        )
        else:
            self.use_pargo = False
            
        if config and config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            self.use_multiview_fusion = True
            self.multiview_fusion_encoder = MultiViewFusionEncoder(config)
            if config.model_type == 'multiview_fusion':
                self.multiview_fusion_proj = nn.Linear(config.multiview_hidden_dim, output_dim)  
        else:
            self.use_multiview_fusion = False
        
        if config and config.model_type == 'ablation':
            self.use_multiview_ablation = True
            self.multiview_ablation_encoder = MultiViewAblationEncoder(config)
            self.ablation_2d_proj = nn.Linear(config.multiview_hidden_dim, output_dim)  
        else:
            self.use_multiview_ablation = False
        
        if config and config.model_type == 'multiview_fusion_honeybee':
            self.use_multiview_fusion_honeybee = True
            self.multiview_fusion_honeybee_encoder = MultiViewFusionHoneybeeEncoder(config, output_dim)
        else:
            self.use_multiview_fusion_honeybee = False
        
        if config and config.model_type == 'multiview_fusion_pargo':
            self.use_multiview_fusion_pargo = True
            self.multiview_fusion_pargo_encoder = MultiViewFusionPargoEncoder(config, output_dim)
        else:
            self.use_multiview_fusion_pargo = False

    def _process_multiview_modality(self, images: torch.Tensor, pos_embedding: nn.Parameter, mask: torch.Tensor = None) -> torch.Tensor:
        if images is None:
            return None
            
        B, N, C, H, W = images.shape
        
        images_flat = images.view(B * N, C, H, W)
        if C == 1:
            images_flat = torch.cat([images_flat] * 3, dim=1)  
        
        clip_output = self.clip_vision(pixel_values=images_flat)
        features_flat = clip_output.pooler_output  
        
        features_reshaped = features_flat.view(B, N, -1)  
        
        if N <= pos_embedding.shape[0]:
            pos_emb = pos_embedding[:N].unsqueeze(0)  
        else:
            extra_embeddings = pos_embedding[-1:].repeat(N - pos_embedding.shape[0], 1)
            pos_emb = torch.cat([pos_embedding, extra_embeddings], dim=0).unsqueeze(0)
        
        features_with_pos = features_reshaped + pos_emb
        
        projected_features = self.vision_proj(features_with_pos.view(B * N, -1))
        final_features = projected_features.view(B, N, -1)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(final_features)
            final_features = final_features * mask_expanded.float()
        
        return final_features

    def _log_missing_views_stats(self):
        if hasattr(self, 'missing_views_stats') and self.missing_views_stats:
            logger.info("=" * 50)
            logger.info("=" * 50)
            
            total_missing = 0
            for modality, stats in self.missing_views_stats.items():
                missing_count = stats['missing']
                total_batches = stats['total_batches']
                avg_missing = missing_count / total_batches if total_batches > 0 else 0
                
                logger.info(f"  - total missing views: {missing_count}")
                logger.info(f"  - total batches: {total_batches}")
                logger.info(f"  - average missing per batch: {avg_missing:.2f} views")
                
                total_missing += missing_count
            
            logger.info("-" * 30)
            logger.info(f"total missing views: {total_missing}")
            logger.info("=" * 50)
        else:
            logger.info("no missing views found")

    def _update_missing_views_stats(self, modality_type: str, missing_count: int):
        if not hasattr(self, 'missing_views_stats'):
            self.missing_views_stats = {}
        if modality_type not in self.missing_views_stats:
            self.missing_views_stats[modality_type] = {'missing': 0, 'total_batches': 0}
        self.missing_views_stats[modality_type]['missing'] += missing_count
        self.missing_views_stats[modality_type]['total_batches'] += 1

    def _check_non_fusion_missing_views(self, images: torch.Tensor, mask: torch.Tensor, modality_type: str):
        if images is not None and mask is not None:
            B, max_views = mask.shape
            for b in range(B):
                actual_views = mask[b].sum().item()
                missing_views = max(0, 4 - actual_views)
                if missing_views > 0:
                    logger.warning(f"sample {b}: {modality_type} modality missing {missing_views} views, actual {actual_views} views")
                    self._update_missing_views_stats(f"{modality_type}_non_fusion", missing_views)

    def _process_modality_with_fusion(self, images: torch.Tensor, modality_type: str) -> torch.Tensor:
        
        if images is None:
            return None
            
        B, N, C, H, W = images.shape
        
        missing_views = max(0, 4 - N)
        if missing_views > 0:
            logger.warning(f"{modality_type} modality missing {missing_views} views, actual {N} views, will flexibly concatenate")
            self._update_missing_views_stats(modality_type, missing_views)
        
        if N >= 4:
            views_to_use = images[:, :4]  # [B, 4, C, H, W]
        else:
            views_to_use = []
            for i in range(4):
                view_idx = min(i, N - 1)  
                views_to_use.append(images[:, view_idx])
            views_to_use = torch.stack(views_to_use, dim=1)  
        
        if C == 1:
            views_to_use = views_to_use.repeat(1, 1, 3, 1, 1)  
        
        front, back, left, right = views_to_use[:, 0], views_to_use[:, 1], views_to_use[:, 2], views_to_use[:, 3]
        
        top_row = torch.cat([front, back], dim=-1)  
        
        bottom_row = torch.cat([left, right], dim=-1)  
        
        fused_image = torch.cat([top_row, bottom_row], dim=-2)  
        
        resized_fused = F.interpolate(fused_image, size=(224, 224), mode='bilinear', align_corners=False)
        
        clip_output = self.clip_vision(pixel_values=resized_fused)
        features = clip_output.pooler_output  
        
        if self.use_honeybee:
            patch_features = clip_output.last_hidden_state[:, 1:]  
            
            projector_key = f'{modality_type}_fused'
            
            if projector_key in self.honeybee_projectors:
                honeybee_features = self.honeybee_projectors[projector_key](patch_features)  
                projected_features = honeybee_features.mean(dim=1)  
            else:
                projected_features = self.vision_proj(features)  
                
        elif self.use_pargo:
            patch_features = clip_output.last_hidden_state[:, 1:]  
            
            projector_key = f'{modality_type}_fused'
            
            if projector_key in self.pargo_projectors:
                pargo_features = self.pargo_projectors[projector_key](patch_features)  # [B, 64, 4096]
                projected_features = pargo_features.mean(dim=1)  
            else:
                projected_features = self.vision_proj(features)  
        else:
            projected_features = self.vision_proj(features)  
        
        return projected_features

    def _apply_cmnext_fusion(self, rgb_features: torch.Tensor, depth_features: torch.Tensor = None, 
                           event_features: torch.Tensor = None, depth_mask: torch.Tensor = None,
                           event_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        fused_features = {}
        
        if rgb_features is not None:
            enhanced_rgb = rgb_features
            
            if depth_features is not None and hasattr(self, 'depth_sq_hub'):
                try:
                    B, N_rgb, D = rgb_features.shape
                    B, N_depth, D = depth_features.shape
                    
                    rgb_flat = rgb_features.view(B, N_rgb, D)
                    depth_flat = depth_features.view(B, N_depth, D)
                    
                    enhanced_rgb = self.depth_sq_hub(rgb_flat, depth_flat, depth_mask)
                except Exception as e:
                    logger.warning(f"Depth SQ-Hub fusion failed: {e}, using original RGB features")
                    enhanced_rgb = rgb_features
                
            if event_features is not None and hasattr(self, 'event_sq_hub'):
                try:
                    B, N_rgb, D = enhanced_rgb.shape
                    B, N_event, D = event_features.shape
                    
                    enhanced_rgb_flat = enhanced_rgb.view(B, N_rgb, D)
                    event_flat = event_features.view(B, N_event, D)
                    
                    enhanced_rgb = self.event_sq_hub(enhanced_rgb_flat, event_flat, event_mask)
                except Exception as e:
                    logger.warning(f"Event SQ-Hub fusion failed: {e}, keeping previous enhanced RGB")
            
            if hasattr(self, 'rgb_ppx'):
                try:
                    enhanced_rgb = self.rgb_ppx(enhanced_rgb)
                except Exception as e:
                    logger.warning(f"RGB PPX mixing failed: {e}, using non-mixed features")
            
            fused_features['rgb_views'] = enhanced_rgb
        
        if depth_features is not None:
            if hasattr(self, 'depth_ppx'):
                try:
                    depth_features = self.depth_ppx(depth_features)
                except Exception as e:
                    logger.warning(f"Depth PPX mixing failed: {e}")
            fused_features['depth_views'] = depth_features
            
        if event_features is not None:
            if hasattr(self, 'event_ppx'):
                try:
                    event_features = self.event_ppx(event_features)
                except Exception as e:
                    logger.warning(f"Event PPX mixing failed: {e}")
            fused_features['event_views'] = event_features
        
        return fused_features

    def _process_multiview_modality_honeybee(self, images: torch.Tensor, modality_type: str, 
                                           pos_embedding: nn.Parameter, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        if images is None:
            return {}
            
        B, N, C, H, W = images.shape
        view_names = ['FRONT', 'BACK', 'LEFT', 'RIGHT']  
        
        if not hasattr(self, 'honeybee_projectors'):
            logger.error("Honeybee projectors not found")
            return {}
        
        enhanced_features = {}
        
        for view_idx in range(N):
            if view_idx >= len(view_names):
                break
                
            view_name = view_names[view_idx]
            projector_key = f'{modality_type}_{view_name}'
            
            if projector_key not in self.honeybee_projectors:
                logger.warning(f"Honeybee projector not found for {projector_key}")
                continue
            
            if mask is not None and not mask[0, view_idx]:  # 假设batch内mask相同
                continue
            
            view_images = images[:, view_idx]  # [B, C, H, W]
            
            if C == 1:
                view_images = torch.cat([view_images] * 3, dim=1)  # [B, 3, H, W]
            
            clip_output = self.clip_vision(pixel_values=view_images)
            
            patch_features = clip_output.last_hidden_state[:, 1:]  
            
            if view_idx < pos_embedding.shape[0]:
                pos_emb = pos_embedding[view_idx:view_idx+1].unsqueeze(0)  # [1, 1, 768]
                pos_emb = pos_emb.expand(B, 1, -1)  # [B, 1, 768]
                
                pos_emb_expanded = pos_emb.expand(B, patch_features.shape[1], -1)  # [B, 49, 768]
                patch_features = patch_features + pos_emb_expanded
            
            try:
                projector = self.honeybee_projectors[projector_key]
                enhanced_feature = projector(patch_features)  # [B, 64, 4096]
                enhanced_features[view_name] = enhanced_feature
            except Exception as e:
                logger.warning(f"Honeybee projection failed for {projector_key}: {e}")
                pooled_feature = patch_features.mean(dim=1)  # [B, 768]
                projected_feature = self.vision_proj(pooled_feature).unsqueeze(1)  # [B, 1, 4096]
                enhanced_features[view_name] = projected_feature
        
        return enhanced_features

    def _process_multiview_modality_pargo(self, images: torch.Tensor, modality_type: str, 
                                         pos_embedding: nn.Parameter, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        if images is None:
            return {}
            
        B, N, C, H, W = images.shape
        view_names = ['FRONT', 'BACK', 'LEFT', 'RIGHT']  
        
        if not hasattr(self, 'pargo_projectors'):
            logger.error("ParGo projectors not found")
            return {}
        
        enhanced_features = {}
        
        for view_idx in range(N):
            if view_idx >= len(view_names):
                break
                
            view_name = view_names[view_idx]
            projector_key = f'{modality_type}_{view_name}'
            
            if projector_key not in self.pargo_projectors:
                logger.warning(f"ParGo projector not found for {projector_key}")
                continue
            
            if mask is not None and not mask[0, view_idx]:  # 假设batch内mask相同
                continue
            
            view_images = images[:, view_idx]  # [B, C, H, W]
            
            if C == 1:
                view_images = torch.cat([view_images] * 3, dim=1)  # [B, 3, H, W]
            
            clip_output = self.clip_vision(pixel_values=view_images)
            
            patch_features = clip_output.last_hidden_state[:, 1:]  
            
            if view_idx < pos_embedding.shape[0]:
                pos_emb = pos_embedding[view_idx:view_idx+1].unsqueeze(0)  # [1, 1, 768]
                pos_emb = pos_emb.expand(B, 1, -1)  # [B, 1, 768]
                
                pos_emb_expanded = pos_emb.expand(B, patch_features.shape[1], -1)  # [B, 49, 768]
                patch_features = patch_features + pos_emb_expanded
            
            try:
                projector = self.pargo_projectors[projector_key]
                enhanced_feature = projector(patch_features)  # [B, num_tokens, 4096]
                enhanced_features[view_name] = enhanced_feature
            except Exception as e:
                logger.warning(f"ParGo projection failed for {projector_key}: {e}")
                pooled_feature = patch_features.mean(dim=1)  # [B, 768]
                projected_feature = self.vision_proj(pooled_feature).unsqueeze(1)  # [B, 1, 4096]
                enhanced_features[view_name] = projected_feature
        
        return enhanced_features

    def forward(self, rgb_images=None, depth_images=None, event_images=None, 
                rgb_mask=None, depth_mask=None, event_mask=None, 
                scene_pointcloud=None, scene_mask=None):
        features = {}
        
        enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
        
        if self.use_honeybee:
            if enable_fusion:
                if rgb_images is not None and (rgb_mask is None or rgb_mask.any()):
                    rgb_fused = self._process_modality_with_fusion(rgb_images, 'rgb')
                    if rgb_fused is not None:
                         features['rgb_fused'] = rgb_fused
                
                if depth_images is not None and (depth_mask is None or depth_mask.any()):
                    depth_fused = self._process_modality_with_fusion(depth_images, 'depth')
                    if depth_fused is not None:
                         features['depth_fused'] = depth_fused
                
                if event_images is not None and (event_mask is None or event_mask.any()):
                    event_fused = self._process_modality_with_fusion(event_images, 'event')
                    if event_fused is not None:
                         features['event_fused'] = event_fused
            else:
                if rgb_images is not None and (rgb_mask is None or rgb_mask.any()):
                    self._check_non_fusion_missing_views(rgb_images, rgb_mask, 'rgb')
                    rgb_features = self._process_multiview_modality_honeybee(
                        rgb_images, 'rgb', self.rgb_positional_embedding, rgb_mask
                    )
                    features.update({f'rgb_{k}': v for k, v in rgb_features.items()})

                if depth_images is not None and (depth_mask is None or depth_mask.any()):
                    self._check_non_fusion_missing_views(depth_images, depth_mask, 'depth')
                    depth_features = self._process_multiview_modality_honeybee(
                        depth_images, 'depth', self.depth_positional_embedding, depth_mask
                    )
                    features.update({f'depth_{k}': v for k, v in depth_features.items()})

                if event_images is not None and (event_mask is None or event_mask.any()):
                    self._check_non_fusion_missing_views(event_images, event_mask, 'event')
                    event_features = self._process_multiview_modality_honeybee(
                        event_images, 'event', self.event_positional_embedding, event_mask
                    )
                    features.update({f'event_{k}': v for k, v in event_features.items()})
                
        elif self.use_pargo:
            if enable_fusion:
                if rgb_images is not None and (rgb_mask is None or rgb_mask.any()):
                    rgb_fused = self._process_modality_with_fusion(rgb_images, 'rgb')
                    if rgb_fused is not None:
                        features['rgb_fused'] = rgb_fused
                
                if depth_images is not None and (depth_mask is None or depth_mask.any()):
                    depth_fused = self._process_modality_with_fusion(depth_images, 'depth')
                    if depth_fused is not None:
                        features['depth_fused'] = depth_fused
                
                if event_images is not None and (event_mask is None or event_mask.any()):
                    event_fused = self._process_modality_with_fusion(event_images, 'event')
                    if event_fused is not None:
                        features['event_fused'] = event_fused
            else:
                if rgb_images is not None and (rgb_mask is None or rgb_mask.any()):
                    self._check_non_fusion_missing_views(rgb_images, rgb_mask, 'rgb')
                    rgb_features = self._process_multiview_modality_pargo(
                        rgb_images, 'rgb', self.rgb_positional_embedding, rgb_mask
                    )
                    features.update({f'rgb_{k}': v for k, v in rgb_features.items()})

                if depth_images is not None and (depth_mask is None or depth_mask.any()):
                    self._check_non_fusion_missing_views(depth_images, depth_mask, 'depth')
                    depth_features = self._process_multiview_modality_pargo(
                        depth_images, 'depth', self.depth_positional_embedding, depth_mask
                    )
                    features.update({f'depth_{k}': v for k, v in depth_features.items()})

                if event_images is not None and (event_mask is None or event_mask.any()):
                    self._check_non_fusion_missing_views(event_images, event_mask, 'event')
                    event_features = self._process_multiview_modality_pargo(
                        event_images, 'event', self.event_positional_embedding, event_mask
                    )
                    features.update({f'event_{k}': v for k, v in event_features.items()})
                
        elif self.config.model_type == 'multiview_fusion':
            multiview_images_list = []
            
            if rgb_images is not None:
                if rgb_images.shape[1] < 4:
                    missing_views = 4 - rgb_images.shape[1]
                    logger.warning(f"MultiView-Fusion: RGB模态缺失{missing_views}个视角，实际{rgb_images.shape[1]}个")
                    self._update_missing_views_stats('rgb_multiview_fusion', missing_views)
                multiview_images_list.append(rgb_images[:, :4])
            
            if depth_images is not None:
                if depth_images.shape[1] < 4:
                    missing_views = 4 - depth_images.shape[1]
                    logger.warning(f"MultiView-Fusion: Depth模态缺失{missing_views}个视角，实际{depth_images.shape[1]}个")
                    self._update_missing_views_stats('depth_multiview_fusion', missing_views)
                depth_4_views = depth_images[:, :4]  # [B, 4, 1, 224, 224]
                if depth_4_views.shape[2] == 1:
                    depth_4_views = depth_4_views.repeat(1, 1, 3, 1, 1)  # [B, 4, 3, 224, 224]
                multiview_images_list.append(depth_4_views)
                
            if event_images is not None:
                if event_images.shape[1] < 4:
                    missing_views = 4 - event_images.shape[1]
                    logger.warning(f"MultiView-Fusion: Event模态缺失{missing_views}个视角，实际{event_images.shape[1]}个")
                    self._update_missing_views_stats('event_multiview_fusion', missing_views)
                multiview_images_list.append(event_images[:, :4])
            
            if len(multiview_images_list) == 3: 
                multiview_images = torch.cat(multiview_images_list, dim=1)  # [B, 12, 3, 224, 224]
                fusion_feature = self.multiview_fusion_encoder(multiview_images)  # [B, 512]
                fusion_feature = self.multiview_fusion_proj(fusion_feature)  # [B, 512] -> [B, 4096]
                features['multiview_fusion'] = fusion_feature
            else:
                available_modalities = []
                if rgb_images is not None: available_modalities.append('rgb')
                if depth_images is not None: available_modalities.append('depth')
                if event_images is not None: available_modalities.append('event')
                raise ValueError(f"MultiView-Fusion requires RGB、Depth、Event three modalities, but only {available_modalities} are present")
                    
        elif self.use_multiview_fusion_honeybee:
            modality_images = {}
            
            if rgb_images is not None and (rgb_mask is None or rgb_mask.any()):
                if rgb_images.shape[1] < 4:
                    missing_views = 4 - rgb_images.shape[1]
                    logger.warning(f"MultiView-Fusion-Honeybee: RGB modality is missing {missing_views} views, actually {rgb_images.shape[1]} views")
                    self._update_missing_views_stats('rgb_multiview_fusion_honeybee', missing_views)
                modality_images['rgb'] = rgb_images[:, :4]  
                
            if depth_images is not None and (depth_mask is None or depth_mask.any()):
                if depth_images.shape[1] < 4:
                    missing_views = 4 - depth_images.shape[1]
                    logger.warning(f"MultiView-Fusion-Honeybee: Depth modality is missing {missing_views} views, actually {depth_images.shape[1]} views")
                    self._update_missing_views_stats('depth_multiview_fusion_honeybee', missing_views)
                depth_4_views = depth_images[:, :4]  # [B, 4, 1, 224, 224]
                if depth_4_views.shape[2] == 1:
                    depth_4_views = depth_4_views.repeat(1, 1, 3, 1, 1)  # [B, 4, 3, 224, 224]
                modality_images['depth'] = depth_4_views
                
            if event_images is not None and (event_mask is None or event_mask.any()):
                if event_images.shape[1] < 4:
                    missing_views = 4 - event_images.shape[1]
                    logger.warning(f"MultiView-Fusion-Honeybee: Event modality is missing {missing_views} views, actually {event_images.shape[1]} views")
                    self._update_missing_views_stats('event_multiview_fusion_honeybee', missing_views)
                modality_images['event'] = event_images[:, :4]  
            
            if modality_images:
                fusion_honeybee_feature = self.multiview_fusion_honeybee_encoder(modality_images)  # [B, 4096]
                features['multiview_fusion_honeybee'] = fusion_honeybee_feature.unsqueeze(1)  # [B, 1, 4096]
            else:
                logger.warning("MultiView-Fusion-Honeybee mode: no valid modalities found")
            
            if scene_pointcloud is not None and (scene_mask is None or scene_mask.any()):
                lidar_feature = self.scene_encoder(scene_pointcloud)
                features['scene_pointcloud'] = lidar_feature
                
        elif self.config.model_type == 'ablation':
            modality_images = {}
            
            ablation_2d_modalities = getattr(self.config, 'ablation_2d_modalities', [])
            if not ablation_2d_modalities:
                ablation_2d_modalities = [m for m in self.config.enabled_modalities if m in ['rgb', 'depth', 'event']]
            
            if 'rgb' in ablation_2d_modalities and rgb_images is not None:
                modality_images['rgb'] = rgb_images[:, :4]  
                
            if 'depth' in ablation_2d_modalities and depth_images is not None:
                depth_4_views = depth_images[:, :4]  
                if depth_4_views.shape[2] == 1:
                    depth_4_views = depth_4_views.repeat(1, 1, 3, 1, 1)  
                modality_images['depth'] = depth_4_views
            
            if modality_images:
                ablation_2d_feature = self.multiview_ablation_encoder(modality_images)  
                ablation_2d_feature = self.ablation_2d_proj(ablation_2d_feature)  
                features['ablation_2d_fusion'] = ablation_2d_feature.unsqueeze(1)  
            else:
                logger.warning(f"MultiView-Ablation mode: no valid 2D modalities found, ablation_2d_modalities={ablation_2d_modalities}, enabled_modalities={self.config.enabled_modalities}")
            
            ablation_enable_lidar = getattr(self.config, 'ablation_enable_lidar', False)
            if not ablation_enable_lidar:
                ablation_enable_lidar = 'pointcloud' in self.config.enabled_modalities
                
            if ablation_enable_lidar and scene_pointcloud is not None and (scene_mask is None or scene_mask.any()):
                lidar_feature = self.scene_encoder(scene_pointcloud)  
                features['scene_pointcloud'] = lidar_feature  
        

        elif self.use_multiview_fusion_pargo:
            modality_images = {}
            
            if rgb_images is not None and (rgb_mask is None or rgb_mask.any()):
                if rgb_images.shape[1] < 4:
                    missing_views = 4 - rgb_images.shape[1]
                    logger.warning(f"MultiView-Fusion-Pargo: RGB modality is missing {missing_views} views, actually {rgb_images.shape[1]} views")
                    self._update_missing_views_stats('rgb_multiview_fusion_pargo', missing_views)
                modality_images['rgb'] = rgb_images[:, :4]  
                
            if depth_images is not None and (depth_mask is None or depth_mask.any()):
                if depth_images.shape[1] < 4:
                    missing_views = 4 - depth_images.shape[1]
                    logger.warning(f"MultiView-Fusion-Pargo: Depth modality is missing {missing_views} views, actually {depth_images.shape[1]} views")
                    self._update_missing_views_stats('depth_multiview_fusion_pargo', missing_views)
                depth_4_views = depth_images[:, :4]  
                if depth_4_views.shape[2] == 1:
                    depth_4_views = depth_4_views.repeat(1, 1, 3, 1, 1)  
                modality_images['depth'] = depth_4_views
                
            if event_images is not None and (event_mask is None or event_mask.any()):
                if event_images.shape[1] < 4:
                    missing_views = 4 - event_images.shape[1]
                    logger.warning(f"MultiView-Fusion-Pargo: Event modality is missing {missing_views} views, actually {event_images.shape[1]} views")
                    self._update_missing_views_stats('event_multiview_fusion_pargo', missing_views)
                modality_images['event'] = event_images[:, :4]  
            
            if modality_images:
                fusion_pargo_feature = self.multiview_fusion_pargo_encoder(modality_images)  
                features['multiview_fusion_pargo'] = fusion_pargo_feature.unsqueeze(1)  
            else:
                logger.warning("MultiView-Fusion-Pargo mode: no valid modalities found")
            
            if scene_pointcloud is not None and (scene_mask is None or scene_mask.any()):
                lidar_feature = self.scene_encoder(scene_pointcloud)
                features['scene_pointcloud'] = lidar_feature
                
        else:
            if enable_fusion:
                rgb_fused = None
                if rgb_images is not None and (rgb_mask is None or rgb_mask.any()):
                    rgb_fused = self._process_modality_with_fusion(rgb_images, 'rgb')
                    if rgb_fused is not None:
                        features['rgb_fused'] = rgb_fused

                depth_fused = None
                if depth_images is not None and (depth_mask is None or depth_mask.any()):
                    depth_fused = self._process_modality_with_fusion(depth_images, 'depth')
                    if depth_fused is not None:
                        features['depth_fused'] = depth_fused

                event_fused = None
                if event_images is not None and (event_mask is None or event_mask.any()):
                    event_fused = self._process_modality_with_fusion(event_images, 'event')
                    if event_fused is not None:
                        features['event_fused'] = event_fused

                if self.use_cmnext and rgb_fused is not None:
                    pass
            else:
                rgb_features = None
                if rgb_images is not None and (rgb_mask is None or rgb_mask.any()):
                    self._check_non_fusion_missing_views(rgb_images, rgb_mask, 'rgb')
                    rgb_features = self._process_multiview_modality(rgb_images, self.rgb_positional_embedding, rgb_mask)

                depth_features = None
                if depth_images is not None and (depth_mask is None or depth_mask.any()):
                    self._check_non_fusion_missing_views(depth_images, depth_mask, 'depth')
                    depth_features = self._process_multiview_modality(depth_images, self.depth_positional_embedding, depth_mask)

                event_features = None
                if event_images is not None and (event_mask is None or event_mask.any()):
                    self._check_non_fusion_missing_views(event_images, event_mask, 'event')
                    event_features = self._process_multiview_modality(event_images, self.event_positional_embedding, event_mask)

                if self.use_cmnext:
                    features.update(self._apply_cmnext_fusion(
                        rgb_features, depth_features, event_features, 
                        depth_mask, event_mask
                    ))
                else:
                    if rgb_features is not None:
                        features['rgb_views'] = rgb_features
                    if depth_features is not None:
                        features['depth_views'] = depth_features
                    if event_features is not None:
                        features['event_views'] = event_features

        if (self.config.model_type != 'ablation' and 
            scene_pointcloud is not None and (scene_mask is None or scene_mask.any())):
            scene_feature = self.scene_encoder(scene_pointcloud)
            features['scene_pointcloud'] = scene_feature

        self._log_missing_views_stats()

        return features


class ScenePointEncoder(nn.Module):

    def __init__(self, output_dim: int, input_feature_dim: int = 0):
        super().__init__()
        self.pointnet_encoder = PointNetPlusPlusEncoder(output_dim=output_dim, input_feature_dim=input_feature_dim)



    def forward(self, scene_pointcloud: Optional[torch.Tensor]):
        if scene_pointcloud is None:
            return None

        B, N, C = scene_pointcloud.shape
        
        if self.training and not scene_pointcloud.requires_grad:
            scene_pointcloud = scene_pointcloud.requires_grad_(True)
        
        normalized_cloud = self._normalize_scene_pointcloud(scene_pointcloud)
        
        try:
            encoded_features = self.pointnet_encoder(normalized_cloud)
            
            if encoded_features is None:
                return torch.zeros(B, self.pointnet_encoder.output_dim, device=scene_pointcloud.device)
            
            return encoded_features
            
        except Exception as e:
            logger.error(f"Scene point cloud encoding failed: {e}")
            return torch.zeros(B, self.pointnet_encoder.output_dim, device=scene_pointcloud.device)

    def _normalize_scene_pointcloud(self, pointcloud: torch.Tensor) -> torch.Tensor:
        B, N, C = pointcloud.shape
        normalized_clouds = torch.zeros_like(pointcloud)
        
        for b in range(B):
            points = pointcloud[b]
            
            if torch.sum(torch.abs(points)) > 1e-6:
                centroid = torch.mean(points, dim=0, keepdim=True)
                centered_points = points - centroid
                
                max_dist = torch.max(torch.norm(centered_points, dim=1))
                if max_dist > 1e-6:
                    normalized_points = centered_points / max_dist
                else:
                    normalized_points = centered_points
                
                normalized_clouds[b] = normalized_points
            else:
                normalized_clouds[b] = points
        
        return normalized_clouds

