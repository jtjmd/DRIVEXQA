#!/usr/bin/env python3


import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_json_serialize(obj):

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj


current_dir = Path(__file__).parent
project_root = current_dir.parent
trainer_dir = project_root / "trainer"


if str(trainer_dir) not in sys.path:
    sys.path.insert(0, str(trainer_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


try:
    from configs import get_ablation_config, FusionDataConfig
    from fusion_data_loader import create_fusion_dataloader
    from leo_authentic_trainer import LEOAgent
    from evaluation_metrics import EvaluationMetrics
except ImportError as e:
    logger.error(f"failure: {e}")
    logger.error(f"Python: {sys.path}")
    logger.error(f"trainer: {trainer_dir}")
    raise


class GPTEvaluationRunner:

    
    def __init__(self, 
                 checkpoint_path: str,
                 model_type: str = "ablation",
                 data_dir: str = "./data_rebuilt",
                 modalities: List[str] = None,
                 enable_lidar: bool = True,
                  enable_multiview_fusion: bool = False,
                  fusion_variant: Optional[str] = None,
                  fusion_token_layout: Optional[str] = None,
                  attention_type: str = "both",
                 eval_split: str = "val",
                 batch_size: int = 8,
                 num_workers: int = 4,
                 max_eval_samples: Optional[int] = None):
       
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.data_dir = data_dir
        self.eval_split = eval_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_eval_samples = max_eval_samples

        self.modalities = modalities
        self.enable_lidar = enable_lidar
        self.enable_multiview_fusion = enable_multiview_fusion
        self.fusion_variant = fusion_variant
        self.fusion_token_layout = fusion_token_layout
        self.attention_type = attention_type
        
        
        self.config = self._create_config()
        self.config.data_dir = self.data_dir
        
       
        self.tokenizer = None
        self.model = None
        self.evaluator = None
        

    
    def _create_config(self):
        
        logger.info(f"{self.model_type} ...")
        

        from configs import LEOConfig
        config = LEOConfig(model_type=self.model_type)
        

        supported_types = [
            'baseline', 'cmnext', 'honeybee', 'pargo',
            'multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo',
            'ablation'
        ]
        
        if self.model_type not in supported_types:
            raise ValueError(f"not support: {self.model_type}。support: {supported_types}")
        

        if self.model_type == 'ablation' and self.modalities is not None:
            logger.info(f"modal: {self.modalities}")
            config.enabled_modalities = self.modalities

            config.ablation_2d_modalities = [m for m in self.modalities if m in ['rgb', 'depth', 'event']]
            logger.info(f"applied modalities: {config.enabled_modalities}")
            logger.info(f"ablation_2d_modalities: {config.ablation_2d_modalities}")
        

        if self.model_type == 'ablation' and self.attention_type != "both":
            config.attention_type = self.attention_type
            logger.info(f"attention_type: {config.attention_type}")

        if self.model_type == 'ablation' and self.fusion_variant is not None:
            config.fusion_variant = self.fusion_variant
            logger.info(f"fusion_variant: {config.fusion_variant}")
        

        if hasattr(self, 'enable_multiview_fusion') and self.enable_multiview_fusion is not None:
            config.enable_multiview_fusion = self.enable_multiview_fusion

        if hasattr(self, 'fusion_variant') and self.fusion_variant is not None and hasattr(config, 'fusion_variant'):
            config.fusion_variant = self.fusion_variant
        if hasattr(self, 'fusion_token_layout') and self.fusion_token_layout is not None and hasattr(config, 'fusion_token_layout'):
            config.fusion_token_layout = self.fusion_token_layout
            config.fusion_num_tokens = 3 if self.fusion_token_layout == 'triple' else 1
        
        logger.info(f" {self.model_type} ")
        logger.info(f"   modal: {config.enabled_modalities}")
        logger.info(f"   pointcloud: {config.pointcloud_mode}")
        logger.info(f"   fusion: {getattr(config, 'enable_multiview_fusion', False)}")
        if hasattr(config, 'fusion_variant'):
            logger.info(f"  head: {getattr(config, 'fusion_variant', 'gap')}")
        logger.info(f"   size: {config.image_size}")
        return config
    
    def setup(self):

        logger.info("environment setup...")
        

        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        logger.info(f"use device: {self.device}")
        

        self.tokenizer, self.model = self._load_model()
        

        self.metrics_calculator = EvaluationMetrics(device=self.device)
        

        data_dir_path = Path(self.data_dir)
        self.original_qa_dir = str(data_dir_path.parent / "data" / "qa")
        

        try:
            from enhanced_evaluator import EnhancedEvaluator
            from metadata_extractor import MetadataExtractor
            

            api_key = getattr(self, 'openai_api_key', None)
            enable_gpt = getattr(self, 'enable_gpt_eval', False)  
            disable_gpt = getattr(self, 'disable_gpt_eval', True)  

            

            if api_key and enable_gpt and not disable_gpt:
                self.enhanced_evaluator = EnhancedEvaluator(
                    openai_api_key=api_key,
                    original_qa_dir=self.original_qa_dir,
                    enable_gpt_eval=enable_gpt,
                    gpt_max_workers=8
                )
                self.metadata_extractor = MetadataExtractor()
                logger.info(" GPT setup sucess")
            else:
                self.enhanced_evaluator = None
                self.metadata_extractor = None
                logger.info("gpt fail")
        except ImportError:
            self.enhanced_evaluator = None
            self.metadata_extractor = None
            logger.warning("skpi gpt")
        except Exception as e:

            logger.warning(f"GPT error: {e}")
            logger.warning(f"error tpye: {type(e).__name__}")
            import traceback
            logger.warning(f"error: {traceback.format_exc()}")
            self.enhanced_evaluator = None
            self.metadata_extractor = None

        
        logger.info("setup finished")
    
    def _load_model(self):

        logger.info("load model...")
        

        checkpoint_path = Path(self.checkpoint_path)
        potential_tokenizer_paths = [
            checkpoint_path,
            checkpoint_path / "final_checkpoint",
            checkpoint_path.parent / "final_checkpoint" if checkpoint_path.name == "final_checkpoint" else None
        ]
        
        tokenizer = None
        for tokenizer_path in potential_tokenizer_paths:
            if tokenizer_path is None:
                continue
                
            tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
            if tokenizer_config_path.exists():
                try:
                    logger.info(f"try {tokenizer_path} to load tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(tokenizer_path),
                        use_fast=False,
                        trust_remote_code=True,
                        use_auth_token=False,
                        local_files_only=True,
                        resume_download=False
                    )
                    logger.info(f"Tokenizer loaded: {tokenizer_path}")
                    break
                except Exception as e:
                    logger.warning(f"{tokenizer_path} tokenizer loaded with error: {e}")
                    continue
        
        if tokenizer is None:
            logger.warning("can't find tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_name_or_path,
                use_fast=False,
                trust_remote_code=True,
                use_auth_token=False,
                local_files_only=True,
                resume_download=False
            )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"tokenizer length: {len(tokenizer)}")
        
        
        logger.info(f"Loading LLM model: {self.config.llm_name_or_path}")
        
        if self.accelerator.is_main_process:
            llm_model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None, 
                use_auth_token=False,
                local_files_only=True,
                resume_download=False,
            )
            

            original_device = next(llm_model.parameters()).device
            llm_model = llm_model.cpu()
            
            if len(tokenizer) != llm_model.config.vocab_size:
                logger.info(f"token embeddings: {llm_model.config.vocab_size} → {len(tokenizer)}")
                llm_model.resize_token_embeddings(len(tokenizer))
            
            llm_model = llm_model.to(self.accelerator.device)
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None,
                use_auth_token=False,
                local_files_only=True,
                resume_download=False,
            )
            llm_model = llm_model.to(self.accelerator.device)
            

            if len(tokenizer) != llm_model.config.vocab_size:
                old_embeddings = llm_model.get_input_embeddings()
                new_embeddings = torch.nn.Embedding(
                    len(tokenizer), 
                    old_embeddings.embedding_dim,
                    dtype=old_embeddings.weight.dtype,
                    device=old_embeddings.weight.device
                )
                with torch.no_grad():
                    new_embeddings.weight[:old_embeddings.num_embeddings] = old_embeddings.weight
                llm_model.set_input_embeddings(new_embeddings)
                
                if hasattr(llm_model, 'lm_head') and llm_model.lm_head is not None:
                    if llm_model.lm_head.weight.shape[0] != len(tokenizer):
                        old_lm_head = llm_model.lm_head
                        new_lm_head = torch.nn.Linear(
                            old_lm_head.in_features,
                            len(tokenizer),
                            bias=old_lm_head.bias is not None,
                            dtype=old_lm_head.weight.dtype,
                            device=old_lm_head.weight.device
                        )
                        with torch.no_grad():
                            new_lm_head.weight[:old_lm_head.out_features] = old_lm_head.weight
                            if old_lm_head.bias is not None:
                                new_lm_head.bias[:old_lm_head.out_features] = old_lm_head.bias
                        llm_model.lm_head = new_lm_head
                
                llm_model.config.vocab_size = len(tokenizer)
        
  
        self.accelerator.wait_for_everyone()
        

        if self.config.use_lora:
            from peft import PeftModel
            

            checkpoint_path = Path(self.checkpoint_path)
            

            potential_lora_paths = [
                checkpoint_path,  
                checkpoint_path / "final_checkpoint",  
                checkpoint_path.parent / "final_checkpoint" if checkpoint_path.name == "final_checkpoint" else None
            ]
            
            lora_loaded = False
            for lora_path in potential_lora_paths:
                if lora_path is None:
                    continue
                    
                adapter_config_path = lora_path / "adapter_config.json"
                if adapter_config_path.exists():
                    try:
                        logger.info(f"{lora_path} LoRA...")
                        llm_model = PeftModel.from_pretrained(
                            llm_model, 
                            str(lora_path),
                            local_files_only=True,
                            force_download=False
                        )
                        logger.info(f"LoRA loaded from: {lora_path}")
                        lora_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f" From {lora_path} error : {e}")
                        continue
                else:
                    logger.debug(f"not exist: {adapter_config_path}")
            
            if not lora_loaded:
                logger.warning("can't find LoRA")

                self.config.use_lora = False
                logger.info("LoRA false")
            else:


                self._restore_special_token_embeddings(llm_model, tokenizer)
        

        hidden_size = llm_model.config.hidden_size
        agent = LEOAgent(self.config, llm_model, tokenizer, hidden_size)
        

        self._load_vision_encoder_weights(agent)
        

        valid_tokens = sum(1 for token_id in agent.token_map.values() 
                          if token_id != tokenizer.unk_token_id)
        total_tokens = len(agent.token_map)
        
        logger.info(f"token state: {valid_tokens}/{total_tokens} valid")
        
        
        agent.eval()
        

        agent = self.accelerator.prepare(agent)
        
       
        return tokenizer, agent
    
    def _load_vision_encoder_weights(self, agent):
        
        logger.info("loading vision encoder weight...")
        
        checkpoint_path = Path(self.checkpoint_path)
        

        potential_weight_paths = []
        

        for pattern in ["*.safetensors", "*.bin"]:
            potential_weight_paths.extend(checkpoint_path.glob(pattern))
        

        for subdir in checkpoint_path.iterdir():
            if subdir.is_dir():
                for pattern in ["*.safetensors", "*.bin"]:
                    potential_weight_paths.extend(subdir.glob(pattern))
        

        if checkpoint_path.name in ["final_checkpoint", "best_checkpoint", "last_checkpoint"]:
            parent_dir = checkpoint_path.parent
            for pattern in ["*.safetensors", "*.bin"]:
                potential_weight_paths.extend(parent_dir.glob(pattern))
        

        potential_weight_paths = list(set(potential_weight_paths))
        potential_weight_paths.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)
        
        logger.info(f"find {len(potential_weight_paths)}:")
        for i, path in enumerate(potential_weight_paths[:5]): 
            file_size = path.stat().st_size / (1024**2) if path.exists() else 0  # MB
            logger.info(f"  {i+1}. {path.name} ({file_size:.1f}MB)")
        if len(potential_weight_paths) > 5:
            logger.info(f"  and other {len(potential_weight_paths) - 5} files")
        
        state_dict_loaded = False
        
        for weight_path in potential_weight_paths:
            if weight_path is None or not weight_path.exists():
                continue
                
            try:
                logger.info(f"loading from {weight_path}...")
                
                if weight_path.suffix == '.safetensors':
                    from safetensors import safe_open
                    state_dict = {}
                    with safe_open(weight_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if 'vision_encoder' in key:
                                state_dict[key] = f.get_tensor(key)
                else:

                    full_state_dict = torch.load(weight_path, map_location='cpu')
                    state_dict = {k: v for k, v in full_state_dict.items() if 'vision_encoder' in k}
                
                if state_dict:

                    vision_state_dict = {}
                    multiview_count = 0
                    clip_count = 0
                    
                    for key, value in state_dict.items():
                        if 'vision_encoder' in key:

                            new_key = key.replace('vision_encoder.', '', 1)
                            vision_state_dict[new_key] = value

                            if 'multiview' in key or 'ablation' in key:
                                multiview_count += 1
                            elif 'clip_vision' in key:
                                clip_count += 1
                    
                    if vision_state_dict:

                        try:
                            missing_keys, unexpected_keys = agent.vision_encoder.load_state_dict(
                                vision_state_dict, strict=False
                            )
                            
                            logger.info(f"Vision encoder loaded from: {weight_path}")

                            
                            if missing_keys:
                                logger.info(f"  - missing keys: {len(missing_keys)} 个")

                                critical_missing = [k for k in missing_keys if 'multiview' in k or 'ablation' in k]
                                if critical_missing:
                                    logger.warning(f"  - critical missing: {len(critical_missing)} 个")
                                    for key in critical_missing[:5]:
                                        logger.warning(f"     {key}")
                                
                            if unexpected_keys:
                                logger.debug(f"  - unexpected keys: {len(unexpected_keys)} 个")
                            

                            weight_categories = self._categorize_loaded_weights(vision_state_dict)
                            

                            for category, count in weight_categories.items():
                                if count > 0:
                                    logger.info(f"  - {category}: {count} parameters")
                            

                            if len(vision_state_dict) > 0:
                                logger.info(" Vision encoder loaded")
                                logger.info(f"  -  {len(vision_state_dict)} ")
                                state_dict_loaded = True
                                break
                            else:
                                logger.warning(f"no vision encoder weight")
                            
                        except Exception as load_e:
                            logger.error(f"load_error: {load_e}")
                            continue
                    else:
                        logger.warning(f"no vision_encoder weight")
                else:
                    logger.warning(f"no vision_encoder weight")
                    
            except Exception as e:
                logger.warning(f"weight error: {e}")
                continue
        
        if not state_dict_loaded:
            logger.error("no vision encoder weights")

        else:
            logger.info(" Vision encoder loaded!")
    
    def _restore_special_token_embeddings(self, llm_model, tokenizer):

        try:

            embeddings = llm_model.get_input_embeddings()
            

            special_tokens = []
            if self.model_type == "ablation":

                fusion_token = '<FUSION>'
                scene_token = '<SCENE_POINTCLOUD>'
                
                fusion_id = tokenizer.convert_tokens_to_ids(fusion_token)
                scene_id = tokenizer.convert_tokens_to_ids(scene_token)
                
                if fusion_id != tokenizer.unk_token_id:
                    special_tokens.append((fusion_token, fusion_id))
                if scene_id != tokenizer.unk_token_id:
                    special_tokens.append((scene_token, scene_id))
            elif self.model_type in ["multiview_fusion", "multiview_fusion_honeybee", "multiview_fusion_pargo"]:

                fusion_token = '<FUSION>'
                scene_token = '<SCENE_POINTCLOUD>'
                
                fusion_id = tokenizer.convert_tokens_to_ids(fusion_token)
                scene_id = tokenizer.convert_tokens_to_ids(scene_token)
                
                if fusion_id != tokenizer.unk_token_id:
                    special_tokens.append((fusion_token, fusion_id))
                if scene_id != tokenizer.unk_token_id:
                    special_tokens.append((scene_token, scene_id))
            elif self.model_type == "pargo":

                pargo_tokens = ['<RGB_PARGO>', '<DEPTH_PARGO>', '<EVENT_PARGO>', '<SCENE_POINTCLOUD>']
                
                for token in pargo_tokens:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if token_id != tokenizer.unk_token_id:
                        special_tokens.append((token, token_id))
            elif self.model_type == "honeybee":

                honeybee_tokens = ['<RGB_ENHANCED>', '<DEPTH_ENHANCED>', '<EVENT_ENHANCED>', '<SCENE_POINTCLOUD>']
                
                for token in honeybee_tokens:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if token_id != tokenizer.unk_token_id:
                        special_tokens.append((token, token_id))

            logger.info(f"check {len(special_tokens)}token embeddings...")
            
            for token_name, token_id in special_tokens:
                if token_id < embeddings.num_embeddings:
                    embedding_vector = embeddings.weight[token_id]
                    

                    embedding_std = embedding_vector.std().item()
                    embedding_mean = embedding_vector.mean().item()
                    
                    logger.info(f"  {token_name} (ID: {token_id}): mean={embedding_mean:.4f}, std={embedding_std:.4f}")
                    

                    if embedding_std > 0.5 or abs(embedding_mean) > 0.5:
                        logger.warning(f"   embedding may not trained correctly {embedding_std:.4f}")

                        
        except Exception as e:
            logger.warning(f"token embeddings error: {e}")
    
    def _categorize_loaded_weights(self, state_dict):
  
        categories = {
            'CLIP weight': 0,
            'MultiView/Ablation encoder': 0,
            'Honeybee': 0,
            'Pargo': 0,
            'CMNext': 0,
            'projector': 0,
            'others': 0
        }
        
        for key in state_dict.keys():
            key_lower = key.lower()
            

            if 'clip_vision' in key_lower:
                categories['CLIP weight'] += 1
            elif any(x in key_lower for x in ['multiview', 'ablation']):
                categories['MultiView/Ablation encoder'] += 1
            elif 'honeybee' in key_lower:
                categories['Honeybee'] += 1
            elif 'pargo' in key_lower:
                categories['Pargo'] += 1
            elif any(x in key_lower for x in ['cmnext', 'depth_sq_hub']):
                categories['CMNext'] += 1
            elif 'vision_proj' in key_lower or 'proj.weight' in key_lower:
                categories['projector'] += 1
            else:
                categories['others'] += 1
        
        return categories
    
    def create_dataloader(self) -> DataLoader:

        logger.info("create dataloader...")
        

        data_config = FusionDataConfig.from_leo_config(self.config)
        

        dataloader = create_fusion_dataloader(
            config=data_config,
            tokenizer=self.tokenizer,
            split=self.eval_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            model_type=self.config.model_type,
            enable_multiview_fusion=self.config.enable_multiview_fusion,
            enable_metadata_extraction=True
        )
        

        return dataloader
    

    
    def evaluate(self, max_samples: Optional[int] = None) -> Dict:


        logger.info(f"Process {self.accelerator.process_index}/{self.accelerator.num_processes}")
        logger.info(f"Device: {self.device}")
        

        dataloader = self.create_dataloader()
        

        self.eval_dataloader = self.accelerator.prepare(dataloader)
        
        self.model.eval()
        all_predictions = []
        all_references = []
        all_images = []
        all_questions = []
        all_scene_ids = []
        all_conditions = []
        all_timestamps = []
        
        samples_processed = 0
        successful_multimodal_batches = 0
        total_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.eval_dataloader, 
                desc=f"Evaluating GPU-{self.accelerator.process_index}",
                disable=not self.accelerator.is_main_process
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                total_batches += 1
                
                try:
                    if batch is None:
                        logger.warning(f"跳过空batch {batch_idx}")
                        continue
                    

                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    

                    model_batch = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'rgb_images': batch.get('rgb_images', None),
                        'depth_images': batch.get('depth_images', None),
                        'event_images': batch.get('event_images', None),
                        'scene_pointcloud': batch.get('scene_pointcloud', None),
                        'rgb_mask': batch.get('rgb_mask', None),
                        'depth_mask': batch.get('depth_mask', None),
                        'event_mask': batch.get('event_mask', None),
                        'scene_mask': batch.get('scene_mask', None)
                    }
                    

                    has_multimodal_data = False
                    if 'rgb_mask' in batch and batch['rgb_mask'].any():
                        has_multimodal_data = True
                    if 'depth_mask' in batch and batch['depth_mask'].any():
                        has_multimodal_data = True
                    if 'event_mask' in batch and batch['event_mask'].any():
                        has_multimodal_data = True
                    if 'scene_mask' in batch and batch['scene_mask'].any():
                        has_multimodal_data = True
                    
                    if has_multimodal_data:
                        successful_multimodal_batches += 1
                    
                    batch_size = input_ids.size(0)
                    

                    assistant_patterns = ["Assistant:", "Assistant", "AI:", "AI"]
                    

                    for i in range(batch_size):
                        input_ids_sample = input_ids[i]
                        generation_start = None
                        

                        if batch_idx == 0 and i == 0:
                            decoded_input = self.tokenizer.decode(input_ids_sample, skip_special_tokens=False)
                            logger.info(f"full input: {decoded_input[:300]}...")
                            

                            logger.info("modalities check:")
                            for key in ['rgb_images', 'depth_images', 'event_images', 'scene_pointcloud']:
                                if key in model_batch and model_batch[key] is not None:
                                    shape = model_batch[key].shape
                                    logger.info(f"   {key}: {shape}")
                                else:
                                    logger.warning(f"   {key}: None ")
                            
                            for key in ['rgb_mask', 'depth_mask', 'event_mask', 'scene_mask']:
                                if key in model_batch and model_batch[key] is not None:
                                    mask_info = model_batch[key][i] if len(model_batch[key]) > i else None
                                    if mask_info is not None:
                                        logger.info(f"   {key}: {mask_info.sum().item()}/{mask_info.numel()} valid")
                                    else:
                                        logger.warning(f"   {key}: {i} none")
                                else:
                                    logger.warning(f"   {key}: None ")
                        

                        for pattern in assistant_patterns:
                            pattern_tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
                            if len(pattern_tokens) > 0:
                                pattern_token_id = pattern_tokens[-1]
                                positions = (input_ids_sample == pattern_token_id).nonzero(as_tuple=False)
                                if len(positions) > 0:
                                    generation_start = positions[-1].item() + 1
                                    if batch_idx == 0 and i == 0:

                                    break
                        

                        if generation_start is None:
                            question_marks = ["?", ":", "？", "："]
                            for mark in question_marks:
                                mark_token = self.tokenizer.encode(mark, add_special_tokens=False)
                                if len(mark_token) > 0:
                                    mark_id = mark_token[0]
                                    positions = (input_ids_sample == mark_id).nonzero(as_tuple=False)
                                    if len(positions) > 0:
                                        generation_start = positions[-1].item() + 1
                                        if batch_idx == 0 and i == 0:

                                        break

                        if generation_start is None:
                            generation_start = int(input_ids_sample.size(0) * 0.8)
                            if batch_idx == 0 and i == 0:

                        

                        generation_start = min(generation_start, input_ids_sample.size(0) - 1)
                        

                        input_for_generation = input_ids_sample[:generation_start].unsqueeze(0)
                        

                        gen_batch = {
                            'input_ids': input_for_generation,
                            'attention_mask': attention_mask[i:i+1, :input_for_generation.size(1)],
                            'rgb_images': model_batch['rgb_images'][i:i+1] if model_batch['rgb_images'] is not None else None,
                            'depth_images': model_batch['depth_images'][i:i+1] if model_batch['depth_images'] is not None else None,
                            'event_images': model_batch['event_images'][i:i+1] if model_batch['event_images'] is not None else None,
                            'scene_pointcloud': model_batch['scene_pointcloud'][i:i+1] if model_batch['scene_pointcloud'] is not None else None,
                            'rgb_mask': model_batch['rgb_mask'][i:i+1] if model_batch['rgb_mask'] is not None else None,
                            'depth_mask': model_batch['depth_mask'][i:i+1] if model_batch['depth_mask'] is not None else None,
                            'event_mask': model_batch['event_mask'][i:i+1] if model_batch['event_mask'] is not None else None,
                            'scene_mask': model_batch['scene_mask'][i:i+1] if model_batch['scene_mask'] is not None else None
                        }
                        

                        try:
                            with torch.no_grad():

                                if hasattr(self.model, 'module'):

                                    generated_ids = self.model.module.generate(
                                        gen_batch,
                                        max_new_tokens=120,  
                                        do_sample=False,    
                                        temperature=1.0,
                                        repetition_penalty=1.1,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id
                                    )
                                else:
                                    generated_ids = self.model.generate(
                                        gen_batch,
                                        max_new_tokens=120, 
                                        do_sample=False,    
                                        temperature=1.0,
                                        repetition_penalty=1.1,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id
                                    )
                            

                            

                            if generated_ids.size(1) > 0:

                                if generated_ids.size(1) > input_for_generation.size(1):
  
                                    new_tokens = generated_ids[0, input_for_generation.size(1):]
                                    prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                                    if batch_idx == 0 and i < 2:  
                                        logger.info(f"{batch_idx}-{i} original: '{prediction}'")

                                else:

                                    prediction = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
                                    if batch_idx == 0 and i < 2:  
                                        logger.info(f"{batch_idx}-{i} original: '{prediction}'")                                    

                                

                                if "Assistant:" in prediction:
                                    parts = prediction.split("Assistant:")
                                    if len(parts) > 1:
     
                                        prediction = parts[-1].strip()
                                    else:
     
                                        prediction = prediction.replace("Assistant:", "").strip()
                                
  
                                if "Human:" in prediction:
                                    prediction = prediction.split("Human:")[0].strip()
                                
 
                                prediction = prediction.lstrip(": ").strip()
                                

                                
      
                          
                                if batch_idx == 0 and i < 2: 
                                    logger.info(f"{batch_idx}-{i} final prediction: '{prediction}'")
                                
                            else:
                                prediction = ""
                            
                        except Exception as e:
                            logger.warning(f"error{i}: {e}")
      
                            
                        all_predictions.append(prediction)
                        

                        if 'question' in batch and 'answer' in batch and i < len(batch['question']) and i < len(batch['answer']):
                            question = batch['question'][i]
                            answer = batch['answer'][i]
                            
                            all_questions.append(question)
                            all_references.append(answer)
                        else:
                            if 'full_conversation' in batch and i < len(batch['full_conversation']):
                                full_text = batch['full_conversation'][i]
                                
                                if "Human:" in full_text and "Assistant:" in full_text:
                                    parts = full_text.split("Human:")
                                    if len(parts) > 1:
                                        human_assistant_part = parts[-1]
                                        if "Assistant:" in human_assistant_part:
                                            question_part, answer_part = human_assistant_part.split("Assistant:", 1)
                                            all_questions.append(question_part.strip())
                                            all_references.append(answer_part.strip())
                                        else:
                                            all_questions.append(human_assistant_part.strip())
                                            all_references.append("")
                                    else:
                                        all_questions.append("")
                                        all_references.append("")
                                else:
                                    all_questions.append("Unknown question")
                                    all_references.append("Unknown answer")
                            else:
                                all_questions.append("Unknown question")
                                all_references.append("No reference")
                        

                        all_images.append(f"image_{samples_processed + i}")
                        

                        scene_id = batch.get('scene_id', [f'scene_{samples_processed + i}'])[i] if 'scene_id' in batch and i < len(batch['scene_id']) else f'scene_{samples_processed + i}'
                        condition = batch.get('condition', ['normal'])[i] if 'condition' in batch and i < len(batch['condition']) else 'normal'
                        timestamp = batch.get('timestamp', ['unknown'])[i] if 'timestamp' in batch and i < len(batch['timestamp']) else 'unknown'
                        
 

                        all_scene_ids.append(scene_id)
                        all_conditions.append(condition)
                        all_timestamps.append(timestamp)
                    
                    samples_processed += batch_size
                    

                    if self.accelerator.is_main_process:
                        progress_bar.set_postfix({
                            'samples': samples_processed,
                            'multimodal_batches': f"{successful_multimodal_batches}/{total_batches}",
                            'batch': f"{batch_idx + 1}/{len(self.eval_dataloader)}"
                        })
                    


                    if max_samples and samples_processed >= max_samples:
                        logger.info(f"Reached max samples limit: {max_samples}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        

        logger.info(f"📊 Evaluation Summary:")
        logger.info(f"  └─ Total batches processed: {total_batches}")
        logger.info(f"  └─ Batches with multimodal data: {successful_multimodal_batches}")
        logger.info(f"  └─ Multimodal success rate: {successful_multimodal_batches/max(total_batches, 1):.2%}")
        
 
        logger.info(f"Process {self.accelerator.process_index}: Collected {len(all_predictions)} samples")
        
 
        if self.accelerator.num_processes > 1:
  
            import tempfile
            import pickle
            
            temp_dir = Path("temp_eval_results")
            temp_dir.mkdir(exist_ok=True)
            
            process_results = {
                'predictions': all_predictions,
                'references': all_references,
                'questions': all_questions,
                'scene_ids': all_scene_ids,
                'conditions': all_conditions,
                'timestamps': all_timestamps,
                'total_batches': total_batches,
                'successful_multimodal_batches': successful_multimodal_batches
            }
            
   
            temp_file = temp_dir / f"process_{self.accelerator.process_index}.pkl"
            with open(temp_file, 'wb') as f:
                pickle.dump(process_results, f)
            
            logger.info(f"Process {self.accelerator.process_index}: 中间结果已保存到 {temp_file}")
      
            self.accelerator.wait_for_everyone()
            
      
            if self.accelerator.is_main_process:
                all_gathered_predictions = []
                all_gathered_references = []
                all_gathered_questions = []
                all_gathered_scene_ids = []
                all_gathered_conditions = []
                all_gathered_timestamps = []
                total_gathered_batches = 0
                total_successful_batches = 0
                
       
                for process_idx in range(self.accelerator.num_processes):
                    process_file = temp_dir / f"process_{process_idx}.pkl"
                    if process_file.exists():
                        try:
                            with open(process_file, 'rb') as f:
                                proc_results = pickle.load(f)
                            
                            all_gathered_predictions.extend(proc_results['predictions'])
                            all_gathered_references.extend(proc_results['references'])
                            all_gathered_questions.extend(proc_results['questions'])
                            all_gathered_scene_ids.extend(proc_results['scene_ids'])
                            all_gathered_conditions.extend(proc_results['conditions'])
                            all_gathered_timestamps.extend(proc_results['timestamps'])
                            total_gathered_batches += proc_results['total_batches']
                            total_successful_batches += proc_results['successful_multimodal_batches']
                            
            
                            
                   
                            process_file.unlink()
                            
                        except Exception as e:
                            logger.error(f"results fail: {e}")
                    else:
                        logger.warning(f"results none: {process_file}")
                

                all_predictions = all_gathered_predictions
                all_references = all_gathered_references
                all_questions = all_gathered_questions
                all_scene_ids = all_gathered_scene_ids
                all_conditions = all_gathered_conditions
                all_timestamps = all_gathered_timestamps
                total_batches = total_gathered_batches
                successful_multimodal_batches = total_successful_batches

                
    
                try:
                    temp_dir.rmdir()
                except:
                    pass
        
  
        if self.accelerator.is_main_process and len(all_predictions) > 0:
            logger.info(f"Evaluation completed. Processed {len(all_predictions)} samples")
            
   
            try:
                metrics = self.metrics_calculator.compute_all_metrics(
                    predictions=all_predictions,
                    references=all_references
                )
                
                logger.info("Metrics computation completed")
                for metric, value in metrics.items():
                    logger.info(f"{metric}: {value:.4f}")
                
 
                results = {
                    'config': {
                        'model_type': self.model_type,
                        'modalities': self.modalities,
                        'modalities_2d': getattr(self, 'modalities_2d', []),
                        'enable_lidar': self.enable_lidar,
                        'checkpoint_path': self.checkpoint_path,
                        'eval_split': self.eval_split,
                        'total_samples': len(all_predictions)
                    },
                    'metrics': metrics,
                    'predictions': all_predictions,
                    'ground_truths': all_references,
                    'questions': all_questions,
                    'sample_results': {
                        'predictions': all_predictions,
                        'ground_truths': all_references,
                        'questions': all_questions
                    },
                    'num_samples': len(all_predictions),
                    'multimodal_success_rate': successful_multimodal_batches/max(total_batches, 1),
                    'total_batches': total_batches,
                    'successful_multimodal_batches': successful_multimodal_batches
                }
                
          
                enable_gpt = getattr(self, 'enable_gpt_eval', False)
                disable_gpt = getattr(self, 'disable_gpt_eval', False)
                has_evaluator = self.enhanced_evaluator is not None
                has_extractor = self.metadata_extractor is not None
                

                
                if (has_evaluator and has_extractor and enable_gpt and not disable_gpt):
                    
                    logger.info("start gpt...")
                    try:
        
                        all_metadata = []
                        for i in range(len(all_predictions)):
  
                            scene_id = all_scene_ids[i] if i < len(all_scene_ids) else f'scene_{i}'
                            condition = all_conditions[i] if i < len(all_conditions) else 'normal'
                            timestamp = all_timestamps[i] if i < len(all_timestamps) else 'unknown'
                            
                            sample_for_metadata = {
                                'scene_id': scene_id,
                                'question': all_questions[i],
                                'answer': all_references[i],
                                'condition': condition,
                                'timestamp': timestamp
                            }
                            
      
                            extracted_metadata = self.metadata_extractor.create_sample_metadata(
                                sample_for_metadata, 
                                original_qa_dir=Path(self.original_qa_dir)  
                            )

                            
                            metadata = {
                                'sample_id': f'{self.model_type}_sample_{i}',
                                'question': all_questions[i],
                                'prediction': all_predictions[i],
                                'reference': all_references[i],
                                'scene_id': scene_id,
                                'condition': condition,
                                'timestamp': datetime.now().isoformat(),
                                **extracted_metadata
                            }
                            all_metadata.append(metadata)
                        
      
                        enhanced_results = self.enhanced_evaluator.evaluate_predictions(
                            predictions=all_predictions,
                            references=all_references,
                            questions=all_questions,
                            metadata_samples=all_metadata
                        )
                        
         
                        if enhanced_results and 'gpt_analysis' in enhanced_results:
                            results['gpt_analysis'] = enhanced_results['gpt_analysis']
                            results['comprehensive_analysis'] = enhanced_results.get('comprehensive_analysis', {})
                            results['category_analysis'] = enhanced_results.get('category_analysis', {})
                            
  
                            gpt_analysis = enhanced_results.get('gpt_analysis', {})
                            gpt_results_list = enhanced_results.get('gpt_results', [])
                            
                 
                            gpt_scores_list = []
                            if gpt_results_list:
                                for gpt_result in gpt_results_list:
                                    gpt_score = gpt_result.get('gpt_score', 0)
                                    if isinstance(gpt_score, (int, float)):
                                        gpt_scores_list.append(gpt_score)
                            
                   
                            if gpt_scores_list:
                                results['gpt_scores'] = sum(gpt_scores_list) / len(gpt_scores_list)
                            else:
                                results['gpt_scores'] = 0
                            
              
                            if 'overall_score' in gpt_analysis:
                                results['gpt_overall_stats'] = gpt_analysis['overall_score']
                            else:
                                results['gpt_overall_stats'] = {}
                            
             
                            if 'category_analysis' in enhanced_results:
                                results['gpt_category_scores'] = enhanced_results['category_analysis']
                            else:
                                results['gpt_category_scores'] = {}
                            
                            logger.info(f"GPT: {results.get('gpt_scores', 0)}")
                            logger.info(f"GPToverall: {bool(results.get('gpt_overall_stats'))}")
                            logger.info(f"category: {len(results.get('gpt_category_scores', {}))}")
                            
                            logger.info(" GPT finished")
                        else:
                            logger.warning("gpt none")

                            results['gpt_scores'] = 0
                            results['gpt_category_scores'] = {}
                            results['gpt_overall_stats'] = {}
                            
                    except Exception as e:
                        logger.warning(f"GPT error: {e}")


                        results['gpt_scores'] = 0
                        results['gpt_category_scores'] = {}
                        results['gpt_overall_stats'] = {}
                else:


                    results['gpt_scores'] = 0
                    results['gpt_category_scores'] = {}
                    results['gpt_overall_stats'] = {}
                
                return results
                
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")
  
                return {
                    'config': {
                        'model_type': self.model_type,
                        'modalities': self.modalities,
                        'modalities_2d': getattr(self, 'modalities_2d', []),
                        'enable_lidar': self.enable_lidar,
                        'checkpoint_path': self.checkpoint_path,
                        'eval_split': self.eval_split,
                        'total_samples': len(all_predictions)
                    },
                    'metrics': {},
                    'predictions': all_predictions,
                    'ground_truths': all_references,
                    'questions': all_questions,
                    'sample_results': {
                        'predictions': all_predictions,
                        'ground_truths': all_references,
                        'questions': all_questions
                    },
                    'num_samples': len(all_predictions),
                    'multimodal_success_rate': successful_multimodal_batches/max(total_batches, 1),
                    'total_batches': total_batches,
                    'successful_multimodal_batches': successful_multimodal_batches,
     
                    'gpt_scores': 0,
                    'gpt_category_scores': {},
                    'gpt_overall_stats': {}
                }
        else:
            return None
    
    def save_results(self, results: Dict, output_file: str):

    
        if not self.accelerator.is_main_process:
            return
            
 
        if output_file.endswith('.json'):
            result_dir = Path(output_file).with_suffix('')
        else:
            result_dir = Path(output_file)
        
        logger.info(f"results: {result_dir}")
        result_dir.mkdir(parents=True, exist_ok=True)
        

        safe_results = safe_json_serialize(results)
        

        safe_results['timestamp'] = datetime.now().isoformat()
        

        predictions = []
        ground_truths = []
        questions = []
        

        if 'predictions' in safe_results and 'ground_truths' in safe_results:
            predictions = safe_results['predictions']
            ground_truths = safe_results['ground_truths']
            questions = safe_results.get('questions', [])

        elif 'sample_results' in safe_results:
            sample_results = safe_results['sample_results']
            if isinstance(sample_results, dict):
                predictions = sample_results.get('predictions', [])
                ground_truths = sample_results.get('ground_truths', [])
                questions = sample_results.get('questions', [])
   
        elif 'sample_results' in safe_results and isinstance(safe_results['sample_results'], list):
            sample_list = safe_results['sample_results']
            predictions = [item.get('prediction', '') for item in sample_list]
            ground_truths = [item.get('reference', '') for item in sample_list]
            questions = [item.get('question', '') for item in sample_list]
        
        logger.info(f" data: predictions={len(predictions)}, ground_truths={len(ground_truths)}")
        

        if predictions and ground_truths:
            comparisons_file = result_dir / "comparisons.json"
            comparisons_data = {
                'timestamp': safe_results['timestamp'],
                'total_count': len(predictions),
                'comparisons': []
            }
            
            for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                comparison = {
                    'sample_id': i,
                    'prediction': pred,
                    'ground_truth': gt,
                    'question': questions[i] if i < len(questions) else '',
                    'match': pred.strip().lower() == gt.strip().lower() if isinstance(pred, str) and isinstance(gt, str) else pred == gt
                }
                comparisons_data['comparisons'].append(comparison)
            
            with open(comparisons_file, 'w', encoding='utf-8') as f:
                json.dump(comparisons_data, f, ensure_ascii=False, indent=2)
            logger.info(f"compare results: {comparisons_file}")
        else:

        
        full_results_file = result_dir / "full_results.json"
        simplified_results = {
            'timestamp': safe_results['timestamp'],
            'config': safe_results.get('config', {}),
            'metrics': safe_results.get('metrics', {}) or safe_results.get('traditional_metrics', {}),
            'gpt_scores': safe_results.get('gpt_scores', 0),
            'gpt_category_scores': safe_results.get('gpt_category_scores', {}),
            'gpt_overall_stats': safe_results.get('gpt_overall_stats', {}),
            'summary': {
                'total_samples': len(predictions),
                'model_type': getattr(self, 'model_type', 'unknown'),
                'modalities': getattr(self, 'modalities', []),
                'modalities_2d': getattr(self, 'modalities_2d', []),
                'enable_lidar': getattr(self, 'enable_lidar', False)
            }
        }
        

        if simplified_results['gpt_scores'] > 0:
            simplified_results['gpt_avg_score'] = simplified_results['gpt_scores']

        
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        logger.info(f"results: {full_results_file}")
        
        logger.info(f"results in: {result_dir}")



def parse_args():

    parser = argparse.ArgumentParser(description='GPTevalscript')
    

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='checkpoint')
    parser.add_argument('--model_type', type=str, default='ablation',
                        choices=['baseline', 'honeybee', 'pargo', 'cmnext', 
                                'multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo', 
                                'ablation'], 
                        help='model')
    parser.add_argument('--data_dir', type=str, default='./data_rebuilt',
                        help='datapath')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='outputpath')
    

    parser.add_argument('--modalities', type=str, nargs='+', 
                        default=None,  
                        choices=['rgb', 'depth', 'event', 'pointcloud'],
                        help='modal')
    parser.add_argument('--enable_lidar', action='store_true', default=False,
                        help='LiDAR')
    parser.add_argument('--attention_type', type=str, default='both',
                        choices=['spatial', 'channel', 'both'],
                        help='attn')
    parser.add_argument('--enable_multiview_fusion', action='store_true', default=False,
                        help='2x2')
    parser.add_argument('--fusion_variant', type=str, default=None,
                        choices=[None, 'gap', 'qattn', 'qattn_spectral', 'qattn_depthgate'],
                        help='fusion_variant')
    parser.add_argument('--fusion_token_layout', type=str, default=None,
                        choices=[None, 'single', 'triple'],
                        help='single or triple')
    

    parser.add_argument('--eval_split', type=str, default='val',
                        choices=['val', 'test'],
                        help='val or test')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='maxsamples')
    

    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API')
    parser.add_argument('--disable_gpt_eval', action='store_true', default=False,
                        help='disable gpt')
    
    return parser.parse_args()


def main():

    args = parse_args()
    

    evaluator = GPTEvaluationRunner(
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        data_dir=args.data_dir,
        modalities=args.modalities,
        enable_lidar=args.enable_lidar,
        enable_multiview_fusion=args.enable_multiview_fusion,
        fusion_variant=args.fusion_variant,
        fusion_token_layout=args.fusion_token_layout,
        attention_type=args.attention_type,
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_samples=args.max_samples
    )
    

    evaluator.openai_api_key = args.openai_api_key
    evaluator.enable_gpt_eval = not args.disable_gpt_eval
    evaluator.disable_gpt_eval = args.disable_gpt_eval
    
    logger.info(f"gpt setup: enable_gpt_eval={evaluator.enable_gpt_eval}, disable_gpt_eval={evaluator.disable_gpt_eval}")
    

    evaluator.setup()
    

    if args.output_dir:
        output_file_base = args.output_dir
    else:
        modalities_str = '_'.join(args.modalities) if args.modalities else 'all'
        lidar_str = 'lidar' if args.enable_lidar else 'nolidar'
        
        if args.model_type == "ablation":
            output_file_base = f"gpt_eval_results_ablation_{modalities_str}_{lidar_str}_{args.eval_split}"
        else:
            fusion_str = 'fusion' if args.enable_multiview_fusion else 'nofusion'
            output_file_base = f"gpt_eval_results_{args.model_type}_{fusion_str}_{args.eval_split}"
    

    
   
    results = evaluator.evaluate(max_samples=args.max_samples)
    
    
    if results is not None:
   
        metrics = results.get('metrics', {})
        logger.info("results:")
        logger.info(f"   modeltypes: {args.model_type}")
        logger.info(f"   modalites: {args.modalities}")
        logger.info(f"   LiDAR: {args.enable_lidar}")
        logger.info(f"   samples: {results.get('config', {}).get('total_samples', 0)}")
        
 
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"   {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"   {metric_name}: {metric_value}")
        

        evaluator.save_results(results, output_file_base)
        
        logger.info(f"🎉 {args.model_type} saved in {output_file_base}")
    else:

        logger.info("saved")


if __name__ == "__main__":
    main()