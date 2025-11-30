#!/usr/bin/env python3


import os
import sys
import json
import math
import logging
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaForCausalLM, LlamaTokenizer, get_cosine_schedule_with_warmup, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, AutoModelForCausalLM
from tqdm.auto import tqdm
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import numpy as np
from PIL import Image
from torchvision import transforms
from configs import LEOConfig
from vision_encoder import UnifiedVisionEncoder

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






class LEOAgent(nn.Module):
    VIEW_NAMES = ['FRONT', 'BACK', 'LEFT', 'RIGHT']       # RGB: 4 views
    DEPTH_VIEW_NAMES = ['FRONT', 'BACK', 'LEFT', 'RIGHT'] # Depth: 4 views  
    EVENT_VIEW_NAMES = ['FRONT', 'BACK', 'LEFT', 'RIGHT'] # Event: 4 views

    def __init__(self, config: LEOConfig, llm_model: nn.Module, tokenizer, hidden_size: int):
        super().__init__()
        self.config = config
        self.llm_model = llm_model
        self.tokenizer = tokenizer

        # use new unified vision encoder, pass dynamic hidden_size
        self.vision_encoder = UnifiedVisionEncoder(
            hidden_size, 
            config=config  
        )
        
        # get special token ids and validate
        self.token_map = self.get_modality_token_ids()


    def _get_token_key_name(self, modality: str, view_name: str = None) -> str:
        
        if view_name:
            return f"{modality}_{view_name}"
        else:
            return f"{modality}_fused"

    def get_modality_token_ids(self):
        token_ids = {}

        # MultiView-Fusion mode: fusion token + scene pointcloud token
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            layout = getattr(self.config, 'fusion_token_layout', 'single')
            if layout == 'triple':
                for i in range(1, 4):
                    token = f'<FUSION{i}>'
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id == self.tokenizer.unk_token_id:
                        logger.warning(f"Token {token} not found in vocabulary!")
                    token_ids[f'multiview_fusion_{i}'] = token_id
            else:
                token = '<FUSION>'
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id == self.tokenizer.unk_token_id:
                    logger.warning(f"Token {token} not found in vocabulary!")
                token_ids['multiview_fusion'] = token_id
            

            if 'pointcloud' in self.config.enabled_modalities and self.config.pointcloud_mode == 'scene':
                token = '<SCENE_POINTCLOUD>'
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id == self.tokenizer.unk_token_id:
                    logger.warning(f"Token {token} not found in vocabulary!")
                token_ids['scene_pointcloud'] = token_id  
            return token_ids
        

        elif self.config.model_type == 'ablation':
            token = '<FUSION>'
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                logger.warning(f"Token {token} not found in vocabulary!")
            token_ids['ablation_2d_fusion'] = token_id
            

            if (hasattr(self.config, 'ablation_enable_lidar') and self.config.ablation_enable_lidar) or \
               ('pointcloud' in self.config.enabled_modalities):
                token = '<SCENE_POINTCLOUD>'
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id == self.tokenizer.unk_token_id:
                    logger.warning(f"Token {token} not found in vocabulary!")
                token_ids['scene_pointcloud'] = token_id  
            
            return token_ids


        enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
        
        # RGB tokens
        if 'rgb' in self.config.enabled_modalities:
            if enable_fusion:

                token = '<RGB>'
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id == self.tokenizer.unk_token_id:
                    logger.warning(f"Token {token} not found in vocabulary!")
                token_ids['rgb_fused'] = token_id
            else:

                for name in self.VIEW_NAMES:
                    token = f'<RGB_{name}>'
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id == self.tokenizer.unk_token_id:
                        logger.warning(f"Token {token} not found in vocabulary!")
                    token_ids[f'rgb_{name}'] = token_id
        

        if 'pointcloud' in self.config.enabled_modalities and self.config.pointcloud_mode == 'scene':
            token = '<SCENE_POINTCLOUD>'
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                logger.warning(f"Token {token} not found in vocabulary!")
            token_ids['scene_pointcloud'] = token_id
        
        # Depth tokens
        if 'depth' in self.config.enabled_modalities and self.config.model_type != 'cmnext':
            if enable_fusion:

                token = '<DEPTH>'
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id == self.tokenizer.unk_token_id:
                    logger.warning(f"Token {token} not found in vocabulary!")
                token_ids['depth_fused'] = token_id
            else:

                for name in self.DEPTH_VIEW_NAMES:
                    token = f'<DEPTH_{name}>'
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id == self.tokenizer.unk_token_id:
                        logger.warning(f"Token {token} not found in vocabulary!")
                    token_ids[f'depth_{name}'] = token_id

        # Event tokens
        if 'event' in self.config.enabled_modalities and self.config.model_type != 'cmnext':
            if enable_fusion:

                token = '<EVENT>'
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id == self.tokenizer.unk_token_id:
                    logger.warning(f"Token {token} not found in vocabulary!")
                token_ids['event_fused'] = token_id
            else:

                for name in self.EVENT_VIEW_NAMES:
                    token = f'<EVENT_{name}>'
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id == self.tokenizer.unk_token_id:
                        logger.warning(f"Token {token} not found in vocabulary!")
                    token_ids[f'event_{name}'] = token_id

        return token_ids

    def _validate_special_tokens(self):

        failed_tokens = []
        for token_name, token_id in self.token_map.items():
            if token_id == self.tokenizer.unk_token_id:
                failed_tokens.append(token_name)
        
        if failed_tokens:
            logger.error(f"❌ Failed to add tokens: {failed_tokens}")
            raise ValueError(f"Critical tokens failed to be added to tokenizer: {failed_tokens}")
        else:
            logger.info(f"✅ All {len(self.token_map)} special tokens successfully validated")
            
    def _pool_honeybee_features(self, features: torch.Tensor, pool_type: str = 'adaptive_avg') -> torch.Tensor:

        if features.dim() != 3:
            logger.warning(f"Expected 3D features [B, N, D], got {features.shape}")
            return features
            
        B, N, D = features.shape
        
        if pool_type == 'adaptive_avg':

            pooled = features.mean(dim=1)  # [B, D]
        elif pool_type == 'max':

            pooled = features.max(dim=1)[0]  # [B, D]
        elif pool_type == 'attention_weighted':

            attention_weights = torch.softmax(features.sum(dim=-1), dim=-1)  # [B, N]
            pooled = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)  # [B, D]
        else:
            logger.warning(f"Unknown pool_type: {pool_type}, using adaptive_avg")
            pooled = features.mean(dim=1)
            
        return pooled

    def _pool_pargo_features(self, features: torch.Tensor, pool_type: str = 'adaptive_avg') -> torch.Tensor:

        if features.dim() != 3:
            logger.warning(f"Expected 3D features [B, N, D], got {features.shape}")
            return features
            
        B, N, D = features.shape
        
        if pool_type == 'adaptive_avg':

            pooled = features.mean(dim=1)  # [B, D]
        elif pool_type == 'max':

            pooled = features.max(dim=1)[0]  # [B, D]
        elif pool_type == 'attention_weighted':

            attention_weights = torch.softmax(features.sum(dim=-1), dim=-1)  # [B, N]
            pooled = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)  # [B, D]
        else:
            logger.warning(f"Unknown pool_type: {pool_type}, using adaptive_avg")
            pooled = features.mean(dim=1)
            
        return pooled

    def forward(self, batch, **kwargs):
        input_ids = batch['input_ids']
        labels = batch.get('labels')
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids).clone()
        
        
        # --- 1. get modality features ---
        vision_features = self.vision_encoder(
            rgb_images=batch.get('rgb_images'),
            depth_images=batch.get('depth_images'),  
            event_images=batch.get('event_images'),  
            rgb_mask=batch.get('rgb_mask'), 
            depth_mask=batch.get('depth_mask'),
            event_mask=batch.get('event_mask'),
            scene_pointcloud=batch.get('scene_pointcloud'),
            scene_mask=batch.get('scene_mask')
        )

        # --- 2. token replacement ---

        enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
        
        for b in range(input_ids.shape[0]):
            if self.config.model_type == 'honeybee':

                if enable_fusion:

                    if 'rgb' in self.config.enabled_modalities and 'rgb_fused' in vision_features and 'rgb_fused' in self.token_map:
                        token_id = self.token_map['rgb_fused']
                        feature = vision_features['rgb_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    
                    if 'depth' in self.config.enabled_modalities and 'depth_fused' in vision_features and 'depth_fused' in self.token_map:
                        token_id = self.token_map['depth_fused']
                        feature = vision_features['depth_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    
                    if 'event' in self.config.enabled_modalities and 'event_fused' in vision_features and 'event_fused' in self.token_map:
                        token_id = self.token_map['event_fused']
                        feature = vision_features['event_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                else:

                    if 'rgb' in self.config.enabled_modalities and 'rgb_mask' in batch:
                        for i, name in enumerate(self.VIEW_NAMES):
                            if i < batch['rgb_mask'].shape[1] and batch['rgb_mask'][b, i]:
                                feature_key = f'rgb_{name}'
                                token_key = f'rgb_{name}'
                                if feature_key in vision_features and token_key in self.token_map:
                                    token_id = self.token_map[token_key]
                                    enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                    pooled_feature = self._pool_honeybee_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)

                    if 'depth' in self.config.enabled_modalities and 'depth_mask' in batch:
                        for i, name in enumerate(self.DEPTH_VIEW_NAMES):
                            if i < batch['depth_mask'].shape[1] and batch['depth_mask'][b, i]:
                                feature_key = f'depth_{name}'
                                token_key = f'depth_{name}'
                                if feature_key in vision_features and token_key in self.token_map:
                                    token_id = self.token_map[token_key]
                                    enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                    pooled_feature = self._pool_honeybee_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)

                    if 'event' in self.config.enabled_modalities and 'event_mask' in batch:
                        for i, name in enumerate(self.EVENT_VIEW_NAMES):
                            if i < batch['event_mask'].shape[1] and batch['event_mask'][b, i]:
                                feature_key = f'event_{name}'
                                token_key = f'event_{name}'
                                if feature_key in vision_features and token_key in self.token_map:
                                    token_id = self.token_map[token_key]
                                    enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                    pooled_feature = self._pool_honeybee_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)

            elif self.config.model_type == 'pargo':

                if enable_fusion:

                    if 'rgb' in self.config.enabled_modalities and 'rgb_fused' in vision_features and 'rgb_fused' in self.token_map:
                        token_id = self.token_map['rgb_fused']
                        feature = vision_features['rgb_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    
                    if 'depth' in self.config.enabled_modalities and 'depth_fused' in vision_features and 'depth_fused' in self.token_map:
                        token_id = self.token_map['depth_fused']
                        feature = vision_features['depth_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    
                    if 'event' in self.config.enabled_modalities and 'event_fused' in vision_features and 'event_fused' in self.token_map:
                        token_id = self.token_map['event_fused']
                        feature = vision_features['event_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                else:

                    if 'rgb' in self.config.enabled_modalities and 'rgb_mask' in batch:
                        for i, name in enumerate(self.VIEW_NAMES):
                            if i < batch['rgb_mask'].shape[1] and batch['rgb_mask'][b, i]:
                                feature_key = f'rgb_{name}'
                                token_key = f'rgb_{name}'
                                if feature_key in vision_features and token_key in self.token_map:
                                    token_id = self.token_map[token_key]
                                    enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                    pooled_feature = self._pool_pargo_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)

                    if 'depth' in self.config.enabled_modalities and 'depth_mask' in batch:
                        for i, name in enumerate(self.DEPTH_VIEW_NAMES):
                            if i < batch['depth_mask'].shape[1] and batch['depth_mask'][b, i]:
                                feature_key = f'depth_{name}'
                                token_key = f'depth_{name}'
                                if feature_key in vision_features and token_key in self.token_map:
                                    token_id = self.token_map[token_key]
                                    enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                    pooled_feature = self._pool_pargo_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)

                    if 'event' in self.config.enabled_modalities and 'event_mask' in batch:
                        for i, name in enumerate(self.EVENT_VIEW_NAMES):
                            if i < batch['event_mask'].shape[1] and batch['event_mask'][b, i]:
                                feature_key = f'event_{name}'
                                token_key = f'event_{name}'
                                if feature_key in vision_features and token_key in self.token_map:
                                    token_id = self.token_map[token_key]
                                    enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                    pooled_feature = self._pool_pargo_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)

            elif self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:

                if self.config.model_type == 'multiview_fusion':
                    feature_key = 'multiview_fusion'
                elif self.config.model_type == 'multiview_fusion_honeybee':
                    feature_key = 'multiview_fusion_honeybee'
                elif self.config.model_type == 'multiview_fusion_pargo':
                    feature_key = 'multiview_fusion_pargo'
                
                if feature_key not in vision_features:
                    raise RuntimeError(f"critical error: MultiView-Fusion mode missing necessary feature '{feature_key}'. Available features: {list(vision_features.keys())}")
                
                layout = getattr(self.config, 'fusion_token_layout', 'single')
                fusion_output = vision_features[feature_key][b]
                if layout == 'triple':

                    if fusion_output.dim() == 1:

                        fusion_output = fusion_output.unsqueeze(0).repeat(3, 1)
                    if fusion_output.shape[0] < 3:

                        rep = 3 - fusion_output.shape[0]
                        fusion_output = torch.cat([fusion_output, fusion_output[:rep]], dim=0)
                    for i in range(3):
                        token_key = f'multiview_fusion_{i+1}'
                        if token_key not in self.token_map:
                            raise RuntimeError(f"missing token {token_key}, available: {list(self.token_map.keys())}")
                        token_id = self.token_map[token_key]
                        feat_i = fusion_output[i]
                        success = self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feat_i)
                        if not success:
                            raise RuntimeError(f"token replacement failed: token {token_key}")
                else:
                    if 'multiview_fusion' not in self.token_map:
                        raise RuntimeError(f"critical error: MultiView-Fusion mode missing necessary token 'multiview_fusion'. Available tokens: {list(self.token_map.keys())}")
                    token_id = self.token_map['multiview_fusion']
                    fusion_feature = fusion_output  # [4096] or [1,4096]
                    if fusion_feature.dim() > 1 and fusion_feature.shape[0] == 1:
                        fusion_feature = fusion_feature.squeeze(0)
                    success = self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, fusion_feature)
                    if not success:
                        raise RuntimeError(f"critical MultiView-Fusion token replacement failed, Token ID: {token_id}, mode: {self.config.model_type}")

            elif self.config.model_type == 'ablation':

                if 'ablation_2d_fusion' in vision_features and 'ablation_2d_fusion' in self.token_map:
                    token_id = self.token_map['ablation_2d_fusion']
                    fusion_feature = vision_features['ablation_2d_fusion'][b, 0]  # [B, 1, 4096] -> [4096]
                    success = self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, fusion_feature)
                    if not success:
                        raise RuntimeError(f"critical Ablation 2D fusion token replacement failed, Token ID: {token_id}")
                
                if (self.config.ablation_enable_lidar and 
                    'scene_pointcloud' in vision_features and 'scene_pointcloud' in self.token_map):
                    token_id = self.token_map['scene_pointcloud']
                    lidar_feature = vision_features['scene_pointcloud'][b]  # [B, 4096] -> [4096]
                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, lidar_feature)

            else:

                if enable_fusion:
                    if 'rgb' in self.config.enabled_modalities and 'rgb_fused' in vision_features and 'rgb_fused' in self.token_map:
                        token_id = self.token_map['rgb_fused']
                        feature = vision_features['rgb_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    
                    if 'depth' in self.config.enabled_modalities and 'depth_fused' in vision_features and 'depth_fused' in self.token_map:
                        token_id = self.token_map['depth_fused']
                        feature = vision_features['depth_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    
                    if 'event' in self.config.enabled_modalities and 'event_fused' in vision_features and 'event_fused' in self.token_map:
                        token_id = self.token_map['event_fused']
                        feature = vision_features['event_fused'][b]
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                else:
                    if 'rgb' in self.config.enabled_modalities and 'rgb_views' in vision_features and 'rgb_mask' in batch:
                        for i, name in enumerate(self.VIEW_NAMES):
                            if batch['rgb_mask'][b, i]:
                                token_id = self.token_map[f'rgb_{name}']
                                feature = vision_features['rgb_views'][b, i]
                                self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)

                    if (self.config.model_type == 'baseline' and 
                        'depth' in self.config.enabled_modalities and 
                        'depth_views' in vision_features and 'depth_mask' in batch):
                        for i, name in enumerate(self.DEPTH_VIEW_NAMES):
                            if i < batch['depth_mask'].shape[1] and batch['depth_mask'][b, i]:
                                token_key = f'depth_{name}'
                                if token_key in self.token_map:  
                                    token_id = self.token_map[token_key]
                                    feature = vision_features['depth_views'][b, i]
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                                else:
                                    logger.warning(f"Token {token_key} not found in token_map for baseline mode")

                    if (self.config.model_type == 'baseline' and 
                        'event' in self.config.enabled_modalities and 
                        'event_views' in vision_features and 'event_mask' in batch):
                        for i, name in enumerate(self.EVENT_VIEW_NAMES):
                            if i < batch['event_mask'].shape[1] and batch['event_mask'][b, i]:
                                token_key = f'event_{name}'
                                if token_key in self.token_map:  
                                    token_id = self.token_map[token_key]
                                    feature = vision_features['event_views'][b, i]
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                                else:
                                    logger.warning(f"Token {token_key} not found in token_map for baseline mode")

            if (self.config.pointcloud_mode == 'scene' and
                'pointcloud' in self.config.enabled_modalities):
                
                if 'scene_pointcloud' not in vision_features:
                    raise RuntimeError(f"critical error: enabled scene pointcloud but missing 'scene_pointcloud' feature. Available features: {list(vision_features.keys())}")
                
                if 'scene_pointcloud' not in self.token_map:
                    raise RuntimeError(f"critical error: enabled scene pointcloud but missing 'scene_pointcloud' token. Available tokens: {list(self.token_map.keys())}")
                
                if 'scene_mask' not in batch:
                    raise RuntimeError(f"critical error: enabled scene pointcloud but batch missing 'scene_mask'")
                
                if batch['scene_mask'][b]:
                    token_id = self.token_map['scene_pointcloud']
                    feature = vision_features['scene_pointcloud'][b]
                    success = self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    if not success:
                        raise RuntimeError(f"critical scene pointcloud token replacement failed, Token ID: {token_id}")
                else:
                    logger.warning(f"sample {b} scene pointcloud mask is False, skipping pointcloud token replacement")
        
        
        self._validate_token_replacement_success(vision_features, input_ids.shape[0])
        
        dummy_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        if vision_features:
            for key, features in vision_features.items():
                if features is not None:
                    dummy_loss = dummy_loss + 1e-6 * torch.mean(features)  
        
        # --- 3. LLM inference ---
        llm_outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
            **kwargs
        )
        
     
        if hasattr(llm_outputs, 'loss') and llm_outputs.loss is not None:
            llm_outputs.loss = llm_outputs.loss + dummy_loss
        
        llm_outputs.inputs_embeds = inputs_embeds
        
        return llm_outputs
    
    def generate(self, batch, max_new_tokens=50, do_sample=False, **kwargs):

        self.eval()
        
        with torch.no_grad():
            # 1. modality features processing (same logic as forward)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # get input embeddings
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids).clone()
            
            # debug: check data passed to vision_encoder
            logger.info(f"debug: data passed to vision_encoder:")
            for key in ['rgb_images', 'depth_images', 'event_images', 'scene_pointcloud']:
                value = batch.get(key)
                if value is not None:
                    logger.info(f"   {key}: {value.shape}")
                else:
                    logger.info(f"   {key}: None")
            
            # get vision features
            vision_features = self.vision_encoder(
                rgb_images=batch.get('rgb_images'),
                depth_images=batch.get('depth_images'),
                event_images=batch.get('event_images'),
                rgb_mask=batch.get('rgb_mask'),
                depth_mask=batch.get('depth_mask'),
                event_mask=batch.get('event_mask'),
                scene_pointcloud=batch.get('scene_pointcloud'),
                scene_mask=batch.get('scene_mask')
            )
            

            target_dtype = inputs_embeds.dtype
            

            for b in range(input_ids.shape[0]):
                if self.config.model_type == 'honeybee':
                    
                    enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
                    
                    if enable_fusion:

                        if 'rgb' in self.config.enabled_modalities and 'rgb_fused' in vision_features and 'rgb_fused' in self.token_map:
                            token_id = self.token_map['rgb_fused']
                            feature = vision_features['rgb_fused'][b]  # [4096]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                        
                        if 'depth' in self.config.enabled_modalities and 'depth_fused' in vision_features and 'depth_fused' in self.token_map:
                            token_id = self.token_map['depth_fused']
                            feature = vision_features['depth_fused'][b]  # [4096]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                        
                        if 'event' in self.config.enabled_modalities and 'event_fused' in vision_features and 'event_fused' in self.token_map:
                            token_id = self.token_map['event_fused']
                            feature = vision_features['event_fused'][b]  # [4096]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    else:

                        if 'rgb' in self.config.enabled_modalities and 'rgb_mask' in batch:
                            for i, name in enumerate(self.VIEW_NAMES):
                                if i < batch['rgb_mask'].shape[1] and batch['rgb_mask'][b, i]:
                                    feature_key = f'rgb_{name}'
                                    token_key = f'rgb_{name}'
                                    if feature_key in vision_features and token_key in self.token_map:
                                        token_id = self.token_map[token_key]
                                        enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                        pooled_feature = self._pool_honeybee_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                        pooled_feature = pooled_feature.to(dtype=target_dtype)
                                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)
                        

                        if 'depth' in self.config.enabled_modalities and 'depth_mask' in batch:
                            for i, name in enumerate(self.DEPTH_VIEW_NAMES):
                                if i < batch['depth_mask'].shape[1] and batch['depth_mask'][b, i]:
                                    feature_key = f'depth_{name}'
                                    token_key = f'depth_{name}'
                                    if feature_key in vision_features and token_key in self.token_map:
                                        token_id = self.token_map[token_key]
                                        enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                        pooled_feature = self._pool_honeybee_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                        pooled_feature = pooled_feature.to(dtype=target_dtype)
                                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)
                        
                        if 'event' in self.config.enabled_modalities and 'event_mask' in batch:
                            for i, name in enumerate(self.EVENT_VIEW_NAMES):
                                if i < batch['event_mask'].shape[1] and batch['event_mask'][b, i]:
                                    feature_key = f'event_{name}'
                                    token_key = f'event_{name}'
                                    if feature_key in vision_features and token_key in self.token_map:
                                        token_id = self.token_map[token_key]
                                        enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                        pooled_feature = self._pool_honeybee_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                        pooled_feature = pooled_feature.to(dtype=target_dtype)
                                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)
                
                elif self.config.model_type == 'pargo':

                    enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
                    
                    if enable_fusion:

                        if 'rgb' in self.config.enabled_modalities and 'rgb_fused' in vision_features and 'rgb_fused' in self.token_map:
                            token_id = self.token_map['rgb_fused']
                            feature = vision_features['rgb_fused'][b]  # [4096]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                        
                        if 'depth' in self.config.enabled_modalities and 'depth_fused' in vision_features and 'depth_fused' in self.token_map:
                            token_id = self.token_map['depth_fused']
                            feature = vision_features['depth_fused'][b]  # [4096]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                        
                        if 'event' in self.config.enabled_modalities and 'event_fused' in vision_features and 'event_fused' in self.token_map:
                            token_id = self.token_map['event_fused']
                            feature = vision_features['event_fused'][b]  # [4096]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    else:

                        if 'rgb' in self.config.enabled_modalities and 'rgb_mask' in batch:
                            for i, name in enumerate(self.VIEW_NAMES):
                                if i < batch['rgb_mask'].shape[1] and batch['rgb_mask'][b, i]:
                                    feature_key = f'rgb_{name}'
                                    token_key = f'rgb_{name}'
                                    if feature_key in vision_features and token_key in self.token_map:
                                        token_id = self.token_map[token_key]
                                        enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                        pooled_feature = self._pool_pargo_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                        pooled_feature = pooled_feature.to(dtype=target_dtype)
                                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)
                        
                        if 'depth' in self.config.enabled_modalities and 'depth_mask' in batch:
                            for i, name in enumerate(self.DEPTH_VIEW_NAMES):
                                if i < batch['depth_mask'].shape[1] and batch['depth_mask'][b, i]:
                                    feature_key = f'depth_{name}'
                                    token_key = f'depth_{name}'
                                    if feature_key in vision_features and token_key in self.token_map:
                                        token_id = self.token_map[token_key]
                                        enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                        pooled_feature = self._pool_pargo_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                        pooled_feature = pooled_feature.to(dtype=target_dtype)
                                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)
                        
                        if 'event' in self.config.enabled_modalities and 'event_mask' in batch:
                            for i, name in enumerate(self.EVENT_VIEW_NAMES):
                                if i < batch['event_mask'].shape[1] and batch['event_mask'][b, i]:
                                    feature_key = f'event_{name}'
                                    token_key = f'event_{name}'
                                    if feature_key in vision_features and token_key in self.token_map:
                                        token_id = self.token_map[token_key]
                                        enhanced_feature = vision_features[feature_key][b]  # [64, 4096]
                                        pooled_feature = self._pool_pargo_features(enhanced_feature.unsqueeze(0)).squeeze(0)  # [4096]
                                        pooled_feature = pooled_feature.to(dtype=target_dtype)
                                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, pooled_feature)
                
                elif self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:

                    if self.config.model_type == 'multiview_fusion':
                        feature_key = 'multiview_fusion'
                    elif self.config.model_type == 'multiview_fusion_honeybee':
                        feature_key = 'multiview_fusion_honeybee'
                    elif self.config.model_type == 'multiview_fusion_pargo':
                        feature_key = 'multiview_fusion_pargo'
                    
                    if feature_key in vision_features:
                        layout = getattr(self.config, 'fusion_token_layout', 'single')
                        fusion_feature = vision_features[feature_key][b]  # [4096] or [3, 4096] or [1, 4096]
                        if layout == 'triple' and self.config.model_type == 'multiview_fusion':
                            if fusion_feature.dim() == 1:
                                fusion_feature = fusion_feature.unsqueeze(0).repeat(3, 1)  # [3, 4096]
                            if fusion_feature.dim() == 2 and fusion_feature.shape[0] >= 3:
                                for i in range(3):
                                    token_key = f'multiview_fusion_{i+1}'
                                    if token_key not in self.token_map:
                                        raise RuntimeError(f"missing token {token_key}, available: {list(self.token_map.keys())}")
                                    token_id = self.token_map[token_key]
                                    feat_i = fusion_feature[i].to(dtype=target_dtype)
                                    success = self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feat_i)
                                    if not success:
                                        raise RuntimeError(f"critical MultiView-Fusion triple token replacement failed, Token: {token_key}")
                            else:
                                for i in range(3):
                                    token_key = f'multiview_fusion_{i+1}'
                                    token_id = self.token_map[token_key]
                                    feat_i = fusion_feature[0].to(dtype=target_dtype)
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feat_i)
                        else:
                            if 'multiview_fusion' not in self.token_map and self.config.model_type == 'multiview_fusion':
                                raise RuntimeError(f"critical error: missing necessary token 'multiview_fusion'. Available tokens: {list(self.token_map.keys())}")
                            token_id = self.token_map.get('multiview_fusion')
                            if token_id is not None:
                                if fusion_feature.dim() > 1 and fusion_feature.shape[0] == 1:
                                    fusion_feature = fusion_feature.squeeze(0)
                                fusion_feature = fusion_feature.to(dtype=target_dtype)
                                success = self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, fusion_feature)
                                if not success:
                                    raise RuntimeError(f"critical MultiView-Fusion token replacement failed, Token ID: {token_id}, mode: {self.config.model_type}")
                
                elif self.config.model_type == 'ablation':

                    if 'ablation_2d_fusion' in vision_features and 'ablation_2d_fusion' in self.token_map:
                        token_id = self.token_map['ablation_2d_fusion']
                        fusion_feature = vision_features['ablation_2d_fusion'][b, 0]  # [B, 1, 4096] -> [4096]
                        fusion_feature = fusion_feature.to(dtype=target_dtype)
                        success = self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, fusion_feature)
                        if not success:
                            raise RuntimeError(f"critical Ablation 2D fusion token replacement failed, Token ID: {token_id}")
                        logger.info(f"replaced ablation_2d_fusion token: {fusion_feature.shape}")
                    
                    if (self.config.ablation_enable_lidar and 
                        'scene_pointcloud' in vision_features and 'scene_pointcloud' in self.token_map):
                        token_id = self.token_map['scene_pointcloud']
                        lidar_feature = vision_features['scene_pointcloud'][b]  # [B, 4096] -> [4096]
                        lidar_feature = lidar_feature.to(dtype=target_dtype)
                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, lidar_feature)
                        logger.info(f"replaced scene_pointcloud token: {lidar_feature.shape}")
                
                else:
                    enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
                    
                    if enable_fusion:
                        if 'rgb' in self.config.enabled_modalities and 'rgb_fused' in vision_features and 'rgb_fused' in self.token_map:
                            token_id = self.token_map['rgb_fused']
                            feature = vision_features['rgb_fused'][b]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                        
                        if 'depth' in self.config.enabled_modalities and 'depth_fused' in vision_features and 'depth_fused' in self.token_map:
                            token_id = self.token_map['depth_fused']
                            feature = vision_features['depth_fused'][b]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                        
                        if 'event' in self.config.enabled_modalities and 'event_fused' in vision_features and 'event_fused' in self.token_map:
                            token_id = self.token_map['event_fused']
                            feature = vision_features['event_fused'][b]
                            feature = feature.to(dtype=target_dtype)
                            self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    else:
                        if 'rgb' in self.config.enabled_modalities and 'rgb_views' in vision_features and 'rgb_mask' in batch:
                            for i, name in enumerate(self.VIEW_NAMES):
                                if batch['rgb_mask'][b, i]:
                                    token_id = self.token_map[f'rgb_{name}']
                                    feature = vision_features['rgb_views'][b, i]
                                    feature = feature.to(dtype=target_dtype)
                                    self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    
                        
                        if (self.config.model_type == 'baseline' and 
                            'depth' in self.config.enabled_modalities and 
                            'depth_views' in vision_features and 'depth_mask' in batch):
                            for i, name in enumerate(self.DEPTH_VIEW_NAMES):
                                if i < batch['depth_mask'].shape[1] and batch['depth_mask'][b, i]:
                                    token_key = f'depth_{name}'
                                    if token_key in self.token_map:
                                        token_id = self.token_map[token_key]
                                        feature = vision_features['depth_views'][b, i]
                                        feature = feature.to(dtype=target_dtype)
                                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                        
                        if (self.config.model_type == 'baseline' and 
                            'event' in self.config.enabled_modalities and 
                            'event_views' in vision_features and 'event_mask' in batch):
                            for i, name in enumerate(self.EVENT_VIEW_NAMES):
                                if i < batch['event_mask'].shape[1] and batch['event_mask'][b, i]:
                                    token_key = f'event_{name}'
                                    if token_key in self.token_map:
                                        token_id = self.token_map[token_key]
                                        feature = vision_features['event_views'][b, i]
                                        feature = feature.to(dtype=target_dtype)
                                        self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                
                if (self.config.pointcloud_mode == 'scene' and
                    'pointcloud' in self.config.enabled_modalities and 
                    'scene_pointcloud' in vision_features and 
                    'scene_mask' in batch and batch['scene_mask'][b]):
                    token_id = self.token_map['scene_pointcloud']
                    feature = vision_features['scene_pointcloud'][b]
                    feature = feature.to(dtype=target_dtype)
                    success = self._replace_token_in_sequence(inputs_embeds, input_ids, b, token_id, feature)
                    if not success:
                        raise RuntimeError(f"critical scene pointcloud token replacement failed, Token ID: {token_id}")
            
            attention_mask = attention_mask.to(dtype=torch.long)
            
            stop_words = ["Assistant:", "Human:", "\n\nHuman:", "\n\nAssistant:", "User:", "AI:"]
            stop_token_ids = []
            
            if hasattr(self, 'tokenizer'):
                for stop_word in stop_words:
                    stop_tokens = self.tokenizer.encode(stop_word, add_special_tokens=False)
                    if stop_tokens:
                        stop_token_ids.extend(stop_tokens)
                
                if self.tokenizer.eos_token_id is not None:
                    stop_token_ids.append(self.tokenizer.eos_token_id)
                
                stop_token_ids = list(set(stop_token_ids))
            

            generation_kwargs = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'max_new_tokens': max_new_tokens,
                'do_sample': do_sample,
                'use_cache': True,
                'return_dict_in_generate': False,
                **kwargs
            }
            
            if hasattr(self, 'tokenizer'):
                if self.tokenizer.pad_token_id is not None:
                    generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
                if self.tokenizer.eos_token_id is not None:
                    generation_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
            
            if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
                generation_kwargs.update({
                    'early_stopping': True,          
                    'repetition_penalty': 1.2,       
                    'length_penalty': 1.0,           
                    'no_repeat_ngram_size': 3,       
                })
                
                if stop_token_ids:
                    try:
                        from transformers import StoppingCriteriaList, StoppingCriteria
                        
                        class StopWordsStoppingCriteria(StoppingCriteria):
                            def __init__(self, stop_token_ids, tokenizer):
                                self.stop_token_ids = stop_token_ids
                                self.tokenizer = tokenizer
                            
                            def __call__(self, input_ids, scores, **kwargs):
                                for stop_id in self.stop_token_ids:
                                    if input_ids[0][-1] == stop_id:
                                        return True
                                return False
                        
                        stopping_criteria = StoppingCriteriaList([
                            StopWordsStoppingCriteria(stop_token_ids, self.tokenizer)
                        ])
                        generation_kwargs['stopping_criteria'] = stopping_criteria
                        
                    except ImportError:
                        pass
            
            input_length = inputs_embeds.size(1)
            
            with torch.no_grad():
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in ['pad_token_id', 'eos_token_id', 'stopping_criteria']}
                
                final_kwargs = {**generation_kwargs}
                final_kwargs.update(filtered_kwargs)
                
                generated_ids = self.llm_model.generate(**final_kwargs)
            
            if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo'] and hasattr(self, 'tokenizer'):
                for i in range(generated_ids.size(0)):
                    sequence = generated_ids[i]
                    if sequence.size(0) > input_length:
                        new_tokens = sequence[input_length:]
                        decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        
                        if 'Assistant' in decoded_text:
                            assistant_pos = decoded_text.find('Assistant')
                            if assistant_pos > 0:  
                                clean_text = decoded_text[:assistant_pos].strip()
                                if clean_text:  
                                    clean_tokens = self.tokenizer.encode(clean_text, add_special_tokens=False)
                                    if clean_tokens:
                                        original_input = sequence[:input_length]
                                        new_sequence = torch.cat([
                                            original_input,
                                            torch.tensor(clean_tokens, device=sequence.device)
                                        ])
                                        if new_sequence.size(0) < generated_ids.size(1):
                                            pad_length = generated_ids.size(1) - new_sequence.size(0)
                                            pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                                            padding = torch.full((pad_length,), pad_token, device=sequence.device)
                                            new_sequence = torch.cat([new_sequence, padding])
                                        
                                        generated_ids[i] = new_sequence[:generated_ids.size(1)]
            
            if generated_ids.size(1) <= input_length:
                try:
                    generated_ids = self.llm_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_length=input_length + max_new_tokens,
                        do_sample=False,
                        pad_token_id=getattr(self.tokenizer, 'pad_token_id', None),
                        eos_token_id=getattr(self.tokenizer, 'eos_token_id', None),
                        early_stopping=True,
                        repetition_penalty=1.2 if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo'] else 1.0,
                    )
                except Exception as e:
                    logger.error(f"retry generation failed: {e}")
            
            return generated_ids
    
    def _replace_token_in_sequence(self, inputs_embeds, input_ids, batch_idx, token_id, feature_vector):
        token_indices = (input_ids[batch_idx] == token_id).nonzero(as_tuple=True)[0]
        
        if len(token_indices) == 0:
            token_name = None
            for name, tid in self.token_map.items():
                if tid == token_id:
                    token_name = name
                    break
            
            sequence_tokens = [self.tokenizer.decode([tid]) for tid in input_ids[batch_idx]]
            raise RuntimeError(
                f"critical error: necessary token ID {token_id} ('{token_name}') not found in input sequence."
                f"this indicates data preprocessing issues.\n"
                f"current mode: {self.config.model_type}\n"
                f"sequence content: {sequence_tokens}\n"
                f"expected token map: {self.token_map}"
            )
        
        expected_dim = inputs_embeds.shape[-1]
        actual_dim = feature_vector.shape[-1]
        
        if actual_dim != expected_dim:
            raise ValueError(f"feature dimension mismatch: expected {expected_dim} dimensions, actual {actual_dim} dimensions. Token ID: {token_id}")
        
        token_idx = token_indices[0].item()
        
        if token_idx >= inputs_embeds.shape[1]:
            raise ValueError(f"token index out of bounds: token_idx={token_idx}, sequence length={inputs_embeds.shape[1]}. Token ID: {token_id}")
        
        try:
            inputs_embeds[batch_idx, token_idx] = feature_vector.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            return True
            
        except Exception as e:
            raise RuntimeError(f"token replacement failed: {str(e)}. Token ID: {token_id}, Batch: {batch_idx}, Position: {token_idx}")

    def _validate_token_replacement_success(self, vision_features: dict, batch_size: int):
        
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            expected_features = []
            
            if self.config.model_type == 'multiview_fusion':
                expected_features.append('multiview_fusion')
            elif self.config.model_type == 'multiview_fusion_honeybee':
                expected_features.append('multiview_fusion_honeybee')
            elif self.config.model_type == 'multiview_fusion_pargo':
                expected_features.append('multiview_fusion_pargo')
            
            if 'pointcloud' in self.config.enabled_modalities:
                expected_features.append('scene_pointcloud')
            
            for feature_key in expected_features:
                if feature_key not in vision_features:
                    raise RuntimeError(f"critical validation failed: {self.config.model_type} mode missing critical feature '{feature_key}'")
                
                feature = vision_features[feature_key]
                if feature is None:
                        raise RuntimeError(f"critical validation failed: {self.config.model_type} mode feature '{feature_key}' is None")
                
                if feature.shape[0] != batch_size:
                    raise RuntimeError(f"critical validation failed: feature '{feature_key}' batch size mismatch, expected {batch_size}, actual {feature.shape[0]}")
        
        # MultiView-Ablation模式验证  
        elif self.config.model_type == 'ablation':
            if 'ablation_2d_fusion' not in vision_features:
                raise RuntimeError(f"critical validation failed: MultiView-Ablation mode missing 'ablation_2d_fusion' feature")
                
            if (self.config.ablation_enable_lidar and 
                'pointcloud' in self.config.enabled_modalities and
                'scene_pointcloud' not in vision_features):
                raise RuntimeError(f"critical validation failed: Ablation mode enabled LiDAR but missing 'scene_pointcloud' feature")
        
        else:
            required_modalities = []
            if 'rgb' in self.config.enabled_modalities:
                required_modalities.append('rgb')
            if 'depth' in self.config.enabled_modalities and self.config.model_type != 'cmnext':
                required_modalities.append('depth')  
            if 'event' in self.config.enabled_modalities and self.config.model_type != 'cmnext':
                required_modalities.append('event')  
            
            enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
            
            for modality in required_modalities:
                if enable_fusion:
                    feature_key = f'{modality}_fused'
                else:
                    feature_key = f'{modality}_views'
                    
                if feature_key not in vision_features:
                    raise RuntimeError(f"critical validation failed: {self.config.model_type} mode missing '{feature_key}' feature")
        
        logger.info(f"token replacement validation passed: {self.config.model_type} mode, available features: {list(vision_features.keys())}")
        

class LEOTrainer:
    def __init__(self, config: LEOConfig):
        self.config = config
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb",
            project_dir=os.path.join(config.output_dir, "logs"),
            kwargs_handlers=[ddp_kwargs]
        )
        self.device = self.accelerator.device
        
        logger.info(f"Trainer initialized, using device: {self.device}")

        self.tokenizer = self._setup_tokenizer()
        
        self.model = self._setup_agent()
        
        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler(self.model)
        
    def _setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.llm_name_or_path, use_fast=False)
        tokenizer.padding_side = 'right'
        
        self._add_special_tokens(tokenizer)
        
        special_tokens_to_check = {}
        
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo', 'ablation']:
            enable_fusion = True
        else:
            enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
    
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            if 'rgb' in self.config.enabled_modalities:
                layout = getattr(self.config, 'fusion_token_layout', 'single')
                if layout == 'triple':
                    special_tokens_to_check['<FUSION1>'] = 'multiview_fusion_1'
                    special_tokens_to_check['<FUSION2>'] = 'multiview_fusion_2'
                    special_tokens_to_check['<FUSION3>'] = 'multiview_fusion_3'
                else:
                    special_tokens_to_check['<FUSION>'] = 'multiview_fusion'  
            
            if 'pointcloud' in self.config.enabled_modalities and self.config.pointcloud_mode == 'scene':
                special_tokens_to_check['<SCENE_POINTCLOUD>'] = 'scene_pointcloud'
        
        elif self.config.model_type == 'ablation':
            special_tokens_to_check['<FUSION>'] = 'ablation_2d_fusion'
            
            if (hasattr(self.config, 'ablation_enable_lidar') and self.config.ablation_enable_lidar) or \
               ('pointcloud' in self.config.enabled_modalities):
                special_tokens_to_check['<SCENE_POINTCLOUD>'] = 'scene_pointcloud'  
        
        else:
            if enable_fusion:
                if 'rgb' in self.config.enabled_modalities:
                    special_tokens_to_check['<RGB>'] = 'rgb_fused'
                
                if 'depth' in self.config.enabled_modalities and self.config.model_type != 'cmnext':
                    special_tokens_to_check['<DEPTH>'] = 'depth_fused'
                
                if 'event' in self.config.enabled_modalities and self.config.model_type != 'cmnext':
                    special_tokens_to_check['<EVENT>'] = 'event_fused'
                        
            else:
                if 'rgb' in self.config.enabled_modalities:
                    special_tokens_to_check.update({
                        f'<RGB_{name}>': f'rgb_{name}' for name in LEOAgent.VIEW_NAMES
                    })
                
                if self.config.model_type != 'cmnext':
                    if 'depth' in self.config.enabled_modalities:
                        special_tokens_to_check.update({
                            f'<DEPTH_{name}>': f'depth_{name}' for name in LEOAgent.DEPTH_VIEW_NAMES
                        })
                    
                    if 'event' in self.config.enabled_modalities:
                        special_tokens_to_check.update({
                            f'<EVENT_{name}>': f'event_{name}' for name in LEOAgent.EVENT_VIEW_NAMES
                        })
            
            if 'pointcloud' in self.config.enabled_modalities and self.config.pointcloud_mode == 'scene':
                special_tokens_to_check['<SCENE_POINTCLOUD>'] = 'scene_pointcloud'

        missing_tokens = [
            name for token, name in special_tokens_to_check.items() 
            if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id
        ]
        if missing_tokens:
            raise ValueError(f"critical error: the following special tokens cannot be found in the tokenizer after addition: {missing_tokens}")
            
        return tokenizer

    def _add_special_tokens(self, tokenizer):
        special_tokens_to_add = []
        
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo', 'ablation']:
            enable_fusion = True
        else:
            enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
        
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            if 'rgb' in self.config.enabled_modalities:
                layout = getattr(self.config, 'fusion_token_layout', 'single')
                if layout == 'triple':
                    special_tokens_to_add.extend(['<FUSION1>', '<FUSION2>', '<FUSION3>'])
                else:
                    special_tokens_to_add.append('<FUSION>')  
            
            if 'pointcloud' in self.config.enabled_modalities and self.config.pointcloud_mode == 'scene':
                special_tokens_to_add.append('<SCENE_POINTCLOUD>')
        
        elif self.config.model_type == 'ablation':
            special_tokens_to_add.append('<FUSION>')
            
            if self.config.ablation_enable_lidar and 'pointcloud' in self.config.enabled_modalities:
                special_tokens_to_add.append('<SCENE_POINTCLOUD>')
        
        else:
            if enable_fusion:
                if 'rgb' in self.config.enabled_modalities:
                    special_tokens_to_add.append('<RGB>')
                
                if 'depth' in self.config.enabled_modalities and self.config.model_type != 'cmnext':
                    special_tokens_to_add.append('<DEPTH>')
                
                if 'event' in self.config.enabled_modalities and self.config.model_type != 'cmnext':
                    special_tokens_to_add.append('<EVENT>')
                        
            else:
                if 'rgb' in self.config.enabled_modalities:
                    special_tokens_to_add.extend([f'<RGB_{name}>' for name in LEOAgent.VIEW_NAMES])
                
                if self.config.model_type != 'cmnext':
                    if 'depth' in self.config.enabled_modalities:
                        special_tokens_to_add.extend([f'<DEPTH_{name}>' for name in LEOAgent.DEPTH_VIEW_NAMES])
                    
                    if 'event' in self.config.enabled_modalities:
                        special_tokens_to_add.extend([f'<EVENT_{name}>' for name in LEOAgent.EVENT_VIEW_NAMES])
            
            if 'pointcloud' in self.config.enabled_modalities and self.config.pointcloud_mode == 'scene':
                special_tokens_to_add.append('<SCENE_POINTCLOUD>')

        if not special_tokens_to_add:
            logger.warning("no special tokens added for any modality, because all modalities are disabled.")
            return

        num_added = tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens_to_add
        })
        
        if num_added > 0:
            logger.info(f"added {num_added} new special tokens to the tokenizer.")
            logger.info(f"added tokens: {special_tokens_to_add}")
            logger.info(f"fusion mode: {'enabled' if enable_fusion else 'disabled'}")

    def _setup_agent(self):
        logger.info("setting up LLM and LEOAgent...")
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        llm_model.resize_token_embeddings(len(self.tokenizer))
        
        if self.config.use_lora:
            logger.info("applying LoRA configuration...")
            
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=self.config.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=["embed_tokens", "lm_head"],  
            )
            llm_model = get_peft_model(llm_model, lora_config)
            llm_model.print_trainable_parameters()
        



        hidden_size = llm_model.config.hidden_size
        
        agent = LEOAgent(self.config, llm_model, self.tokenizer, hidden_size)
        
        agent._validate_special_tokens()
        
        self._validate_training_setup(agent)
        
        for param in agent.vision_encoder.parameters():
            param.requires_grad = True
        
        logger.info("LEOAgent setup completed.")
        return agent
    
    def _validate_training_setup(self, agent):
        logger.info("performing training setup validation...")
        
        expected_tokens = []
        
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            layout = getattr(self.config, 'fusion_token_layout', 'single')
            if self.config.model_type == 'multiview_fusion' and layout == 'triple':
                expected_tokens.extend(['multiview_fusion_1', 'multiview_fusion_2', 'multiview_fusion_3'])
            else:
                expected_tokens.append('multiview_fusion')
            if 'pointcloud' in self.config.enabled_modalities:
                expected_tokens.append('scene_pointcloud')
        elif self.config.model_type == 'ablation':
            expected_tokens.append('ablation_2d_fusion')
            if (hasattr(self.config, 'ablation_enable_lidar') and self.config.ablation_enable_lidar) or \
               ('pointcloud' in self.config.enabled_modalities):
                expected_tokens.append('scene_pointcloud')  
        else:
            enable_fusion = getattr(self.config, 'enable_multiview_fusion', False)
            if enable_fusion:
                if 'rgb' in self.config.enabled_modalities:
                    if self.config.model_type == 'baseline':
                        expected_tokens.append('rgb_fused')
                    elif self.config.model_type == 'honeybee':
                        expected_tokens.append('rgb_fused')
        
        for token_name in expected_tokens:
            if token_name not in agent.token_map:
                raise RuntimeError(f"training setup validation failed: missing required token '{token_name}'. available tokens: {list(agent.token_map.keys())}")
            
            token_id = agent.token_map[token_name]
            if token_id == self.tokenizer.unk_token_id:
                raise RuntimeError(f"training setup validation failed: token '{token_name}' mapped to unk_token ({token_id}), this means the token was not correctly added to the vocabulary")
        
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            if not hasattr(agent.vision_encoder, 'multiview_fusion_encoder'):
                raise RuntimeError(f"training setup validation failed: {self.config.model_type} mode missing multiview_fusion_encoder")
            
            if self.config.model_type == 'multiview_fusion' and not hasattr(agent.vision_encoder, 'multiview_fusion_proj'):
                raise RuntimeError(f"training setup validation failed: multiview_fusion mode missing multiview_fusion_proj projection layer")
        
        if self.config.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            if not getattr(self.config, 'enable_multiview_fusion', True):
                logger.warning(f"{self.config.model_type} mode should force enable multiview_fusion, but it is not set in the configuration")
        
        logger.info("training setup validation passed, all components are ready")

    def create_optimizer_and_scheduler(self, model):
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )

        if self.config.max_train_steps is None:
            num_training_steps = "TBD" 
        else:
            num_training_steps = self.config.max_train_steps
        
        logger.info(f"optimizer and scheduler will be configured for {num_training_steps} steps.")
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_train_steps if self.config.max_train_steps else 1000000 
        )
        return optimizer, lr_scheduler

    def train(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="multimodal-vicuna",  
                config={
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "model": self.config.llm_name_or_path,
                    "use_lora": self.config.use_lora,
                    "lora_rank": self.config.lora_rank if self.config.use_lora else None,
                    "enabled_modalities": self.config.enabled_modalities,
                    "pointcloud_mode": self.config.pointcloud_mode,
                },
                init_kwargs={"wandb": {"name": f"leo-fusion-{self.config.pointcloud_mode}"}}
            )
        print("=== parameters status before prepare ===")
        llm_params_with_grad = 0
        llm_params_total = 0
        for name, param in self.model.llm_model.named_parameters():
            llm_params_total += 1
            if param.requires_grad:
                llm_params_with_grad += 1
            else:
                print(f"LLM parameters without gradient: {name}")
        print(f"LLM parameters: {llm_params_with_grad}/{llm_params_total} need gradient")

        embed_layer = self.model.llm_model.get_input_embeddings()
        if hasattr(embed_layer, 'weight'):
            print(f"Embedding layer parameters need gradient: {embed_layer.weight.requires_grad}")
        elif hasattr(embed_layer, 'original_module') and hasattr(embed_layer.original_module, 'weight'):
            print(f"Embedding layer parameters need gradient: {embed_layer.original_module.weight.requires_grad}")
        elif hasattr(embed_layer, 'modules_to_save') and 'default' in embed_layer.modules_to_save:
            original_embed = embed_layer.modules_to_save['default']
            print(f"Embedding layer parameters need gradient: {original_embed.weight.requires_grad}")
        else:
            print(f"Embedding layer type: {type(embed_layer)}")
            print("cannot directly access embedding layer weights, but is wrapped by LoRA")

        self.model, self.optimizer, train_loader, eval_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, eval_loader, self.lr_scheduler
        )

        if self.config.max_train_steps is None:
            num_update_steps_per_epoch = math.ceil(len(train_loader) / self.config.gradient_accumulation_steps)
            self.config.max_train_steps = self.config.num_epochs * num_update_steps_per_epoch
            
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_train_steps
            )
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
            logger.info(f"max_train_steps not set, dynamically calculated to: {self.config.max_train_steps}")

        total_batch_size = self.config.batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        logger.info("***** start training *****")
        logger.info(f"  total batch size = {len(train_loader)}")
        logger.info(f"  total epochs = {self.config.num_epochs}")
        logger.info(f"  batch size per device = {self.config.batch_size}")
        logger.info(f"  total training batch size (with parallel and accumulation) = {total_batch_size}")
        logger.info(f"  gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  total optimization steps = {self.config.max_train_steps}")
        
        progress_bar = tqdm(range(self.config.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(self.config.num_epochs):
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
            self.model.train()
            for step, batch in enumerate(train_loader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(batch)
                    
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                    elif isinstance(outputs, dict) and 'loss' in outputs and outputs['loss'] is not None:
                        loss = outputs['loss']
                    else:
                        raise ValueError(f"Cannot find loss in outputs: {outputs}")
                    
                    print(f"Loss: {loss.item():.4f}")
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        total_norm = 0.0
                        param_count = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                param_count += 1
                        
                        if param_count > 0:
                            total_norm = total_norm ** (1. / 2)
                            self.accelerator.log({
                                "grad_norm": total_norm
                            }, step=completed_steps)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    
                if completed_steps % self.config.logging_steps == 0:
                    self.accelerator.log({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0], "epoch": epoch, "step": completed_steps}, step=completed_steps)
                    logger.info(f"Epoch {epoch}, Step {completed_steps}: Loss: {loss.item():.4f}")

                if completed_steps % self.config.save_steps == 0:
                    self.save_checkpoint(completed_steps)

                if completed_steps >= self.config.max_train_steps:
                    break
                
                if completed_steps % self.config.eval_steps == 0 and eval_loader:
                    val_loss = self.compute_validation_loss(eval_loader)
                    self.accelerator.log({
                        "val_loss": val_loss,
                        "train_val_gap": loss.item() - val_loss
                    }, step=completed_steps)
                    logger.info(f"Step {completed_steps}: Val Loss: {val_loss:.4f}")
            
            self.save_checkpoint(completed_steps, epoch_suffix=f"epoch-{epoch+1}")
            logger.info(f"Epoch {epoch+1} completed, checkpoint saved")
            
            if eval_loader and self.config.eval_steps:

                logger.info(f"Epoch {epoch+1} completed, evaluation started")
                val_loss = self.compute_validation_loss(eval_loader)
                self.accelerator.log({
                    f"epoch_{epoch+1}_val_loss": val_loss,
                    "epoch": epoch + 1
                }, step=completed_steps)
                logger.info(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

            if completed_steps >= self.config.max_train_steps:
                break
        
        self.accelerator.wait_for_everyone()
        self.save_checkpoint(completed_steps, is_final=True)
        if self.accelerator.is_main_process:
            self.accelerator.end_training()
        logger.info("training completed.")

    def compute_validation_loss(self, eval_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                if batch is None:
                    continue
                outputs = self.model(batch)
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    num_batches += 1
                
                if num_batches >= self.config.max_eval_batches:   
                    break
        
        self.model.train()  
        return total_loss / max(num_batches, 1)


    def save_checkpoint(self, step: int, is_final: bool = False, epoch_suffix: str = None):
        if not self.accelerator.is_local_main_process:
            return
        
        if is_final:
            checkpoint_dir = os.path.join(self.config.output_dir, "final_checkpoint")
        elif epoch_suffix:
            checkpoint_dir = os.path.join(self.config.output_dir, epoch_suffix)
        else:
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        if self.config.use_lora:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped_model, 'llm_model'):
                unwrapped_model.llm_model.save_pretrained(checkpoint_dir)
                logger.info(f"LoRA configuration and weights saved to: {checkpoint_dir}")
            else:
                logger.warning("cannot access llm_model, using accelerate to save")
        
        self.accelerator.save_state(checkpoint_dir)
        logger.info(f"checkpoint saved to: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        self.accelerator.load_state(checkpoint_path)
        logger.info(f"Loaded model state from {checkpoint_path}")

    def validate_data_structure(self):
        data_path = Path(self.config.data_dir)
        
        required_files = ['train.json', 'val.json']
        for file_name in required_files:
            file_path = data_path / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"missing prebuilt index file: {file_path}")
        
        required_dirs = ['depth', 'event', 'lidar', 'multiview']
        for dir_name in required_dirs:
            dir_path = data_path / dir_name
            if not dir_path.exists():
                logger.warning(f"missing data directory: {dir_path}")
        
        logger.info("data structure validation passed")

    def analyze_dataset(self):
        from fusion_data_loader import analyze_dataset_statistics
        logger.info("analyzing dataset...")
        analyze_dataset_statistics(self.config.data_dir)

    def preview_data_samples(self, num_samples=3):
        logger.info("=" * 50)
        logger.info("Data Sample Preview")
        logger.info("=" * 50)
        
        if not hasattr(self, 'train_dataloader'):
            logger.warning("Train dataloader not available for preview.")
            return

        for i in range(min(num_samples, len(self.train_dataloader.dataset))):
            try:
                sample = self.train_dataloader.dataset[i]
                if sample is None:
                    logger.info(f"Sample {i+1}: Load failed")
                    continue
                    
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Input length: {len(sample['input_ids'])}")
                logger.info(f"  Label length: {len(sample['labels'])}")
                
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: {value.shape}")
                logger.info("")

            except Exception as e:
                logger.error(f"Failed to preview sample {i+1}: {e}")

def main():



if __name__ == "__main__":
    main() 