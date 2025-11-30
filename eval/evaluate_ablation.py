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
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Module import failed: {e}")
    logger.error(f"Current Python path: {sys.path}")
    logger.error(f"trainer directory: {trainer_dir}")
    raise


class AblationEvaluator:
    
    def __init__(self, 
                 checkpoint_path: str,
                 data_dir: str = "./data_rebuilt",
                 modalities_2d: List[str] = None,
                 enable_lidar: bool = True,
                 eval_split: str = "val",
                 batch_size: int = 8,
                 num_workers: int = 4,
                 max_eval_samples: Optional[int] = None):

        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.modalities_2d = modalities_2d or ['rgb', 'depth', 'event']
        self.enable_lidar = enable_lidar
        self.eval_split = eval_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_eval_samples = max_eval_samples
        
        self.config = get_ablation_config(
            modalities_2d=self.modalities_2d,
            enable_lidar=self.enable_lidar
        )
        self.config.data_dir = self.data_dir
        
        self.tokenizer = None
        self.model = None
        self.evaluator = None
        
        logger.info(f"Initializing Ablation evaluator:")
        logger.info(f"   Checkpoint path: {self.checkpoint_path}")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   2D modalities: {self.modalities_2d}")
        logger.info(f"   Enable LiDAR: {self.enable_lidar}")
        logger.info(f"   Evaluation split: {self.eval_split}")
        logger.info(f"   Batch size: {self.batch_size}")
    
    def setup(self):
        logger.info("Setting up evaluation environment...")
        
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer, self.model = self._load_model()
        
        self.metrics_calculator = EvaluationMetrics(device=self.device)
        
        logger.info("Evaluation environment setup completed")
    
    def _load_model(self):
        logger.info("Loading model and tokenizer...")
        
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
                    logger.info(f"Trying to load tokenizer from {tokenizer_path}...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(tokenizer_path),
                        use_fast=False,
                        trust_remote_code=True,
                        use_auth_token=False,
                        local_files_only=True,
                        resume_download=False
                    )
                    logger.info(f"Tokenizer successfully loaded from: {tokenizer_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
                    continue
        
        if tokenizer is None:
            logger.warning("No tokenizer found in checkpoint, using base tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_name_or_path,
                use_fast=False,
                trust_remote_code=True,
                use_auth_token=False,
                local_files_only=True,
                resume_download=False
            )
        

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded successfully, vocabulary size: {len(tokenizer)}")
        
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
                logger.info(f"Adjusting token embeddings: {llm_model.config.vocab_size} → {len(tokenizer)}")
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
                        logger.info(f"Trying to load LoRA weights from {lora_path}...")
                        llm_model = PeftModel.from_pretrained(
                            llm_model, 
                            str(lora_path),
                            local_files_only=True,
                            force_download=False
                        )
                        logger.info(f"LoRA weights successfully loaded from: {lora_path}")
                        lora_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA weights from {lora_path}: {e}")
                        continue
                else:
                    logger.debug(f"LoRA configuration file does not exist: {adapter_config_path}")
            
            if not lora_loaded:
                logger.warning("No LoRA weights found, using base model (may result in performance degradation)")
                self.config.use_lora = False
                logger.info("use_lora set to False in configuration")
        
        hidden_size = llm_model.config.hidden_size
        agent = LEOAgent(self.config, llm_model, tokenizer, hidden_size)
        
        self._load_vision_encoder_weights(agent)
        
        valid_tokens = sum(1 for token_id in agent.token_map.values() 
                          if token_id != tokenizer.unk_token_id)
        total_tokens = len(agent.token_map)
        
        logger.info(f"Special token status: {valid_tokens}/{total_tokens} valid")
        if valid_tokens == 0:
            logger.warning("All special tokens are invalid, model will run in pure text mode")
            logger.warning("This means multimodal functionality is disabled, only text inference is possible")
        elif valid_tokens < total_tokens:
            logger.warning(f"Only {valid_tokens}/{total_tokens} special tokens are valid")
        
        agent.eval()
        
        agent = self.accelerator.prepare(agent)
        
        logger.info("Model loaded and prepared successfully")
        
        return tokenizer, agent
    
    def _load_vision_encoder_weights(self, agent):
        logger.info("Loading vision encoder weights...")
        
        checkpoint_path = Path(self.checkpoint_path)
        
        potential_weight_paths = [
            checkpoint_path / "model.safetensors",
            checkpoint_path / "pytorch_model.bin", 
            checkpoint_path / "final_checkpoint" / "model.safetensors",
            checkpoint_path / "final_checkpoint" / "pytorch_model.bin",
            checkpoint_path.parent / "final_checkpoint" / "model.safetensors" if checkpoint_path.name == "final_checkpoint" else None
        ]
        
        state_dict_loaded = False
        
        for weight_path in potential_weight_paths:
            if weight_path is None or not weight_path.exists():
                continue
                
            try:
                logger.info(f"Trying to load weights from {weight_path}...")
                
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
                            
                            logger.info(f"Vision encoder weights successfully loaded from: {weight_path}")
                            logger.info(f"  - Total loaded parameters: {len(vision_state_dict)}")
                            logger.info(f"  - MultiView/Ablation parameters: {multiview_count}")
                            logger.info(f"  - CLIP parameters: {clip_count}")
                            
                            if missing_keys:
                                logger.info(f"  - Missing parameters: {len(missing_keys)}")
                                critical_missing = [k for k in missing_keys if 'multiview' in k or 'ablation' in k]
                                if critical_missing:
                                    logger.warning(f"  - Critical missing parameters: {len(critical_missing)}")
                                    for key in critical_missing[:5]:
                                        logger.warning(f"    ❌ {key}")
                                
                            if unexpected_keys:
                                logger.debug(f"  - Unexpected parameters: {len(unexpected_keys)}")
                            
                            key_components = [
                                'multiview_ablation_encoder.modality_to_hidden.weight',
                                'multiview_ablation_encoder.spatial_attention.q_proj.weight',
                                'multiview_ablation_encoder.channel_attention.q_proj.weight',
                                'multiview_ablation_encoder.final_proj.weight'
                            ]
                            
                            loaded_components = 0
                            for comp in key_components:
                                if comp in vision_state_dict:
                                    loaded_components += 1
                                    logger.debug(f"    ✓ {comp}")
                                else:
                                    logger.warning(f"     Missing critical component: {comp}")
                            
                            logger.info(f"  - Critical components loaded: {loaded_components}/{len(key_components)}")
                            
                            if multiview_count > 0:
                                logger.info("Vision encoder weights successfully loaded!")
                                state_dict_loaded = True
                                break
                            else:
                                logger.warning("No critical vision encoder weights found")
                            
                        except Exception as load_e:
                            logger.error(f"Weight loading to model failed: {load_e}")
                            logger.debug("Model expected key examples:")
                            for name, _ in list(agent.vision_encoder.named_parameters())[:5]:
                                logger.debug(f"  Model: {name}")
                            logger.debug("Weight file key examples:")
                            for key in list(vision_state_dict.keys())[:5]:
                                logger.debug(f"  File: {key}")
                            continue
                    else:
                        logger.warning(f"No vision_encoder related weights found in {weight_path}")
                else:
                    logger.warning(f"No vision_encoder weights found in {weight_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to load weights from {weight_path}: {e}")
                continue
        
        if not state_dict_loaded:
            logger.error("No or failed to load vision encoder weights!")
            logger.error("This will result in poor generation quality because the projection layer uses random weights")
            logger.error("It is recommended to check the checkpoint directory and file integrity")
        else:
            logger.info("Vision encoder weights loaded successfully, model should be able to work normally!")
    
    def create_dataloader(self) -> DataLoader:
        logger.info("Creating data loader...")
        
        data_config = FusionDataConfig.from_leo_config(self.config)
        
        dataloader = create_fusion_dataloader(
            config=data_config,
            tokenizer=self.tokenizer,
            split=self.eval_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            model_type=self.config.model_type,
            enable_multiview_fusion=True
        )
        
        logger.info(f"Data loader created successfully, batch count: {len(dataloader)}")
        return dataloader
    
    def _clean_generated_text(self, text: str) -> str:
        import re
        
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()
        
        if not text or text.isspace() or all(c in '.,!?;:' for c in text):
            return ""
        
        return text
    
    def _truncate_to_first_sentence(self, text: str) -> str:
        if not text:
            return text
        
        text = text.strip()
        
        sentence_end = text.find('.')
        if sentence_end != -1:
            return text[:sentence_end + 1].strip()
        else:
            return text



    
    def evaluate(self, max_samples: Optional[int] = None) -> Dict:
        logger.info("Starting Ablation evaluation...")
        logger.info(f"Process {self.accelerator.process_index}/{self.accelerator.num_processes}")
        logger.info(f"Device: {self.device}")
        
        dataloader = self.create_dataloader()
        
        self.eval_dataloader = self.accelerator.prepare(dataloader)
        
        self.model.eval()
        all_predictions = []
        all_references = []
        all_images = []
        all_questions = []
        
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
                            logger.info(f"Full input: {decoded_input[:300]}...")
                        
                        for pattern in assistant_patterns:
                            pattern_tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
                            if len(pattern_tokens) > 0:
                                pattern_token_id = pattern_tokens[-1]
                                positions = (input_ids_sample == pattern_token_id).nonzero(as_tuple=False)
                                if len(positions) > 0:
                                    generation_start = positions[-1].item() + 1
                                    if batch_idx == 0 and i == 0:
                                        logger.info(f"Found {pattern} position: {generation_start}")
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
                                            logger.info(f"Found punctuation position: {generation_start}")
                                        break
                        
                        if generation_start is None:
                            generation_start = int(input_ids_sample.size(0) * 0.8)
                            if batch_idx == 0 and i == 0:
                                logger.warning(f"Using fallback position: {generation_start}")
                        
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
                                        max_new_tokens=32,
                                        do_sample=False
                                    )
                                else:
                                    generated_ids = self.model.generate(
                                        gen_batch,
                                        max_new_tokens=32,
                                        do_sample=False
                                    )
                            
                            if batch_idx == 0 and i == 0:
                                logger.info(f"Generation before length: {input_for_generation.size(1)}")
                                logger.info(f"Generation after length: {generated_ids.size(1)}")
                                logger.info(f"Generated complete sequence: {self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)}")
                            
                            if generated_ids.size(1) > 0:
                                if generated_ids.size(1) > input_for_generation.size(1):
                                    new_tokens = generated_ids[0, input_for_generation.size(1):]
                                    prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                                    
                                    if batch_idx == 0 and i == 0:
                                        logger.info(f"New tokens from complete sequence: {new_tokens.tolist()}")
                                        logger.info(f"Decoded prediction: '{prediction}'")
                                else:
                                    prediction = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
                                    
                                    if batch_idx == 0 and i == 0:
                                        logger.info(f"Directly decoded generated tokens: {generated_ids[0].tolist()}")
                                        logger.info(f"Decoded prediction: '{prediction}'")
                                
                                if "Assistant:" in prediction:
                                    prediction = prediction.split("Assistant:")[0].strip()
                                
                                if "Human:" in prediction:
                                    prediction = prediction.split("Human:")[0].strip()
                                
                                if prediction.startswith(":"):
                                    prediction = prediction[1:].strip()
                                
                                prediction = self._clean_generated_text(prediction)
                                
                                prediction = self._truncate_to_first_sentence(prediction)
                                
                                if batch_idx == 0 and i == 0:
                                    logger.info(f"Post-processing prediction: '{prediction[:100]}...'")
                            else:
                                prediction = ""
                                if batch_idx == 0 and i == 0:
                                    logger.warning("Generated sequence is empty!")
                            
                            if batch_idx == 0 and i == 0:
                                logger.info("Using LEOAgent.generate method completed")
                            
                        except Exception as e:
                            logger.warning(f"Generation failed, sample {i}: {e}")
                            if batch_idx == 0 and i == 0:
                                logger.error(f"Detailed error: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            
                        
                        all_predictions.append(prediction)
                        
                        if 'question' in batch and 'answer' in batch and i < len(batch['question']) and i < len(batch['answer']):
                            question = batch['question'][i]
                            answer = batch['answer'][i]
                            
                            all_questions.append(question)
                            all_references.append(answer)
                            
                            if batch_idx == 0 and i == 0:
                                logger.info(f"Ablation question: {question[:100]}...")
                                logger.info(f"Ablation reference answer: {answer[:100]}...")
                                logger.info(f"Ablation prediction: {prediction[:100]}...")
                                
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
        
        if self.accelerator.is_main_process:
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
                        'modalities_2d': self.config.enabled_modalities,
                        'enable_lidar': self.config.pointcloud_mode == 'token',
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
                
                return results
                
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")
                return {
                    'config': {
                        'modalities_2d': self.config.enabled_modalities,
                        'enable_lidar': self.config.pointcloud_mode == 'token',
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
                    'successful_multimodal_batches': successful_multimodal_batches
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
        
        logger.info(f"💾 创建结果文件夹: {result_dir}")
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
        
        logger.info(f"Extracted data: predictions={len(predictions)}, ground_truths={len(ground_truths)}")
        
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
            logger.info(f"Comparisons analysis saved to: {comparisons_file}")
        else:
            logger.warning("No complete predictions and ground_truths data found, skipping comparisons.json generation")
        
        full_results_file = result_dir / "full_results.json"
        simplified_results = {
            'timestamp': safe_results['timestamp'],
            'config': safe_results.get('config', {}),
            'metrics': safe_results.get('metrics', {}) or safe_results.get('traditional_metrics', {}),
            'gpt_scores': safe_results.get('gpt_scores', {}),
            'gpt_category_scores': safe_results.get('gpt_category_scores', {}),
            'summary': {
                'total_samples': len(predictions),
                'model_type': 'ablation',
                'modalities': safe_results.get('config', {}).get('modalities_2d', []),
                'enable_lidar': safe_results.get('config', {}).get('enable_lidar', False)
            }
        }
        
        if simplified_results['gpt_scores']:
            gpt_values = [v for v in simplified_results['gpt_scores'].values() if isinstance(v, (int, float))]
            if gpt_values:
                simplified_results['gpt_avg_score'] = sum(gpt_values) / len(gpt_values)
        
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Simplified results saved to: {full_results_file}")
        
        logger.info(f"All results saved to folder: {result_dir}")
        logger.info("Folder contents:")
        if predictions and ground_truths:
            logger.info("   comparisons.json - predictions vs ground truths comparison")
        logger.info("   full_results.json - scores and configuration information (no detailed predictions and questions)")


def parse_args():
    parser = argparse.ArgumentParser(description='MultiView-Ablation ablation evaluation')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, default='./data_rebuilt',
                        help='Data directory path')
    parser.add_argument('--output_file', type=str, 
                        help='Output results file path (optional)')
    
    parser.add_argument('--modalities', type=str, nargs='+', 
                        default=['rgb', 'depth', 'event'],
                        choices=['rgb', 'depth', 'event'],
                        help='List of modalities to participate in 2D fusion')
    parser.add_argument('--enable_lidar', action='store_true', default=False,
                        help='Whether to enable LiDAR independent token')
    
    parser.add_argument('--eval_split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Evaluation data split')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='Maximum number of evaluation samples (optional)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    evaluator = AblationEvaluator(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        modalities_2d=args.modalities,
        enable_lidar=args.enable_lidar,
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_samples=args.max_eval_samples
    )
    
    evaluator.setup()
    
    if args.output_file:
        output_file_base = args.output_file
        if output_file_base.endswith('.json'):
            output_file_base = output_file_base[:-5]
    else:
        modalities_str = '_'.join(args.modalities)
        lidar_str = 'lidar' if args.enable_lidar else 'nolidar'
        output_file_base = f"ablation_results_{modalities_str}_{lidar_str}_{args.eval_split}"
    

    

    results = evaluator.evaluate(max_samples=evaluator.max_eval_samples)
    

    if results is not None:
 
        metrics = results['metrics']

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"   {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"   {metric_name}: {metric_value}")
        

        evaluator.save_results(results, output_file_base)
        
        logger.info(f"Ablation results saved in: {output_file_base}")
    else:

        logger.info("waiting..")


if __name__ == "__main__":
    main()