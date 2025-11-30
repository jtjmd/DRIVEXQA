#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import time

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from functools import partial
import sys
from pathlib import Path

try:
    from configs import FusionDataConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from configs import FusionDataConfig

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not installed")

logger = logging.getLogger(__name__)


class FusionDataset(Dataset):
    
    VIEW_NAMES = ['FRONT', 'BACK', 'LEFT', 'RIGHT']           
    DEPTH_VIEW_NAMES = ['FRONT', 'BACK', 'LEFT', 'RIGHT']   
    EVENT_VIEW_NAMES = ['FRONT', 'BACK', 'LEFT', 'RIGHT']   
    
    def __init__(self, config: FusionDataConfig, tokenizer, split: str, model_type: str = 'baseline', enable_multiview_fusion: bool = False, enable_metadata_extraction: bool = False):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.model_type = model_type
        self.enable_multiview_fusion = enable_multiview_fusion  
        self.enable_metadata_extraction = enable_metadata_extraction  
        self.data_dir = Path(config.data_dir)
        self.project_root = self.data_dir.parent
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size), interpolation=Image.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size), interpolation=Image.NEAREST, antialias=False),
            transforms.ToTensor()
        ])
        self.event_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size), interpolation=Image.BILINEAR, antialias=True),
            transforms.ToTensor()
        ])
        
        
    
    def _load_data(self) -> List[Dict]:
        split_file = self.data_dir / "split_index.json"
        if not split_file.exists():
            raise FileNotFoundError(f"split index file not found: {split_file}")
        
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                split_index = json.load(f)
        except Exception as e:
            raise RuntimeError(f"failed to read split index file: {e}")
        
        batch_key = f'{self.split}_batches'
        batch_names = split_index.get(batch_key, [])
        if not batch_names:
            raise ValueError(f"no batch data found for {self.split}, check key: {batch_key}")
        
        logger.info(f"loading {self.split} set: {len(batch_names)} batch files")
        
        data_packed_dir = self.data_dir / "data_packed"
        if not data_packed_dir.exists():
            raise FileNotFoundError(f"batch data directory not found: {data_packed_dir}")
        
        samples = []
        failed_batches = []
        
        for i, batch_name in enumerate(batch_names):
            batch_file = data_packed_dir / batch_name
            if not batch_file.exists():
                logger.warning(f"batch file not found: {batch_file}")
                failed_batches.append(batch_name)
                continue
                
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                if not isinstance(batch_data, list):
                    logger.warning(f"batch file format error: {batch_name} (not a list)")
                    failed_batches.append(batch_name)
                    continue
                
                samples.extend(batch_data)
                if i % 10 == 0:  
                    logger.info(f"loaded batch {i+1}/{len(batch_names)}: +{len(batch_data)} samples")
                    
            except Exception as e:
                logger.error(f"failed to load batch file {batch_file}: {e}")
                failed_batches.append(batch_name)
        
        if failed_batches:
            logger.warning(f"{len(failed_batches)} batch files failed to load: {failed_batches[:5]}...")
        
        if not samples:
            raise RuntimeError(f"failed to load any {self.split} data")
        
        logger.info(f"successfully loaded {len(samples)} samples ({len(failed_batches)} batch files failed)")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx >= len(self.samples):
            logger.warning(f"index out of bounds: {idx} >= {len(self.samples)}")
            return None
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                sample = self.samples[idx]
                result = self._process_sample(sample, idx)  
                if result is not None:
                    return result
                else:
                    logger.debug(f"sample processing returned None: {idx} (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.warning(f"failed to process sample {idx} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"failed to process sample {idx} (final attempt {attempt + 1}/{max_retries})")
        
        return None
    
    def _process_sample(self, sample: Dict, idx: int) -> Optional[Dict[str, Any]]:
        try:
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            modality_data = sample.get('modality_data', {})
            json_metadata = sample.get('metadata', {})
            
            if not question or not answer:
                logger.debug("skipping sample with empty question or answer")
                return None
            
            with ThreadTimeout(30):  
                rgb_images, rgb_mask = self._load_multiview_images(modality_data, 'rgb')
                depth_images, depth_mask = self._load_multiview_images(modality_data, 'depth')
                event_images, event_mask = self._load_multiview_images(modality_data, 'event')
                scene_pointcloud, scene_mask = self._load_scene_pointcloud(modality_data.get('lidar'))
            
            input_text = self._build_text(question, answer)
            
            try:
                tokenized = self.tokenizer(
                    input_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
            except Exception as e:
                logger.warning(f"tokenization failed: {e}")
                return None
            
            input_ids = tokenized.input_ids.squeeze(0)
            
            try:
                extended_input_ids, labels = self._create_labels(input_text, answer, input_ids)
                
                final_input_ids = torch.tensor(extended_input_ids, dtype=torch.long)
                final_labels = torch.tensor(labels, dtype=torch.long)
                
                attention_mask = torch.ones_like(final_input_ids)
                
            except Exception as e:
                logger.warning(f"failed to create labels: {e}")
                return None
            
            result = {
                "input_ids": final_input_ids,
                "labels": final_labels,
                "attention_mask": attention_mask,
                "original_text": input_text,
                "question": question,
                "answer": answer,
                "rgb_images": rgb_images,
                "rgb_mask": rgb_mask,
                "depth_images": depth_images,
                "depth_mask": depth_mask,
                "event_images": event_images,
                "event_mask": event_mask,
                "scene_pointcloud": scene_pointcloud,
                "scene_mask": scene_mask
            }
            
            if self.enable_metadata_extraction:
                sample_metadata = {}
                
                if json_metadata:
                    if 'base_scene' in json_metadata:
                        sample_metadata['scene_id'] = sample.get('scene_id', f'scene_{idx}')
                        if idx < 3:
                            logger.info(f"sample {idx} directly using JSON scene_id: {sample_metadata['scene_id']}")
                    
                    conditions = json_metadata.get('conditions', [])
                    if conditions:
                        weather_keywords = ['sun', 'cloud', 'fog', 'rain', 'night', 'sunny', 'cloudy', 'foggy', 'rainy']
                        sensor_issues = ['overexposure', 'underexposure', 'eventlowres', 'lidarjitter', 'motionblur']
                        
                        for condition in conditions:
                            condition_lower = condition.lower()
                            
                            if condition_lower in weather_keywords:
                                sample_metadata['weather'] = condition_lower
                                if idx < 3:
                                    logger.info(f"sample {idx} extracted weather from JSON: {condition_lower}")
                            
                            elif condition_lower in sensor_issues:
                                sample_metadata['sensor_impact'] = condition_lower
                                sample_metadata['condition'] = condition_lower
                                if idx < 3:
                                    logger.info(f"sample {idx} extracted sensor issue from JSON: {condition_lower}")
                
                if not sample_metadata and 'rgb' in modality_data and modality_data['rgb']:
                    rgb_paths = modality_data['rgb']
                    if rgb_paths and 'front' in rgb_paths:  
                        front_path = rgb_paths['front']
                        if idx < 3:  
                            logger.info(f"sample {idx} alternative path parsing: {front_path}")
                        
                        path_parts = Path(front_path).parts
                        for part in path_parts:
                            if 'MAP_' in part:
                                sample_metadata['scene_id'] = part
                                if idx < 3:
                                    logger.info(f"sample {idx} extracted scene_id from path: {part}")
                                break
                
                if 'scene_id' not in sample_metadata:
                    sample_metadata['scene_id'] = f'scene_{idx}'
                
                if 'condition' not in sample_metadata:
                    sample_metadata['condition'] = 'normal'
                sample_metadata['sample_index'] = idx
                
                result.update({
                    "scene_id": sample_metadata.get('scene_id', f'scene_{idx}'),
                    "condition": sample_metadata.get('condition', 'normal'),
                    "timestamp": sample.get('timestamp', 'unknown'),  
                    "sample_metadata": sample_metadata
                })
            
            return result
            
        except Exception as e:
            logger.warning(f"sample processing exception: {e}")
            return None
    
    def _build_text(self, question: str, answer: str) -> str:
        system_prompt = "You are an AI driving assistant that perceives and understands road environments through multi-modal sensor fusion including RGB cameras, depth sensors, LiDAR, and event cameras."
        
        modality_tokens = []
        
        if self.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo', 'ablation']:
            layout = getattr(self.config, 'fusion_token_layout', 'single')
            if self.model_type == 'multiview_fusion' and layout == 'triple':
                modality_tokens.extend(['<FUSION1>', '<FUSION2>', '<FUSION3>'])
            else:
                modality_tokens.append('<FUSION>')
            
            if 'pointcloud' in self.config.enabled_modalities:
                modality_tokens.append('<SCENE_POINTCLOUD>')
            
            modality_section = ' '.join(modality_tokens)
            return f"System: {system_prompt}\n{modality_section}\nHuman: {question}\nAssistant:"
        
        elif 'rgb' in self.config.enabled_modalities:
            if self.enable_multiview_fusion:
                rgb_tokens = '<RGB>'
            else:
                rgb_tokens = ' '.join([f'<RGB_{name}>' for name in self.VIEW_NAMES])
            modality_tokens.append(rgb_tokens)
        
        if self.model_type not in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo', 'ablation'] and 'depth' in self.config.enabled_modalities:
            if self.enable_multiview_fusion:
                if self.model_type != 'cmnext':
                    depth_tokens = '<DEPTH>'
                    modality_tokens.append(depth_tokens)
            else:
                if self.model_type != 'cmnext':
                    depth_tokens = ' '.join([f'<DEPTH_{name}>' for name in self.DEPTH_VIEW_NAMES])
                    modality_tokens.append(depth_tokens)
        
        if self.model_type not in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo', 'ablation'] and 'event' in self.config.enabled_modalities:
            if self.enable_multiview_fusion:
                if self.model_type != 'cmnext':
                    event_tokens = '<EVENT>'
                    modality_tokens.append(event_tokens)
            else:
                if self.model_type != 'cmnext':
                    event_tokens = ' '.join([f'<EVENT_{name}>' for name in self.EVENT_VIEW_NAMES])
                    modality_tokens.append(event_tokens)
        
        if self.model_type not in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo', 'ablation'] and 'pointcloud' in self.config.enabled_modalities:
            modality_tokens.append('<SCENE_POINTCLOUD>')
        
        modality_section = ' '.join(modality_tokens)
        
        return f"System: {system_prompt}\n{modality_section}\nHuman: {question}\nAssistant:"
    
    def _create_labels(self, input_text: str, answer: str, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        IGNORE_INDEX = -100
        
        labels = [IGNORE_INDEX] * len(input_ids)
        
        try:
            answer_tokens = self.tokenizer.encode(f" {answer}", add_special_tokens=False)
                    
            extended_input_ids = input_ids.tolist() + answer_tokens
            extended_labels = labels + answer_tokens  
            
            return extended_input_ids, extended_labels
            
        except Exception as e:
            logger.warning(f"failed to tokenize answer: {e}")
            return input_ids.tolist(), labels
    
    def _load_multiview_images(self, modality_data: Dict, modality: str) -> Tuple[torch.Tensor, torch.Tensor]:
        modality_info = modality_data.get(modality, {})
        
        if modality == 'rgb':
            view_names = self.VIEW_NAMES
            transform = self.rgb_transform
            num_views = self.config.num_rgb_views
            channels = 3
        elif modality == 'depth':
            view_names = self.DEPTH_VIEW_NAMES
            transform = self.depth_transform
            num_views = self.config.num_depth_views
            channels = 1
        elif modality == 'event':
            view_names = self.EVENT_VIEW_NAMES
            transform = self.event_transform
            num_views = self.config.num_event_views
            channels = 3
        else:
            raise ValueError(f"不支持的模态: {modality}")
        
        images = torch.zeros(num_views, channels, self.config.image_size, self.config.image_size)
        mask = torch.zeros(num_views, dtype=torch.bool)
        
        successful_loads = 0
        for i, view_name in enumerate(view_names):
            view_key = view_name.lower()
            if view_key in modality_info:
                try:
                    image_path = modality_info[view_key]
                    
                    if not os.path.isabs(image_path):
                        full_path = self.project_root / image_path
                    else:
                        full_path = Path(image_path)
                    
                    if not full_path.exists():
                        logger.debug(f"{modality} image not found: {full_path}")
                        continue
                    
                    if modality == 'depth':
                        image = Image.open(full_path).convert('L')
                    else:
                        image = Image.open(full_path).convert('RGB')

                    if image.size[0] == 0 or image.size[1] == 0:
                        logger.debug(f"{modality} image size invalid: {full_path}")
                        continue
                    
                    transformed = transform(image)
                    images[i] = transformed
                    mask[i] = True
                    successful_loads += 1
                    
                except Exception as e:
                    logger.debug(f"failed to load {modality} image {view_key}: {e}")
                    continue
        
        if successful_loads == 0:
            logger.debug(f"failed to load any {modality} image")
        
        return images, mask
    
    def _load_scene_pointcloud(self, lidar_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not lidar_path or 'pointcloud' not in self.config.enabled_modalities:
            return torch.zeros(self.config.scene_pointcloud_size, 3), torch.tensor(False)
        
        try:
            if not os.path.isabs(lidar_path):
                full_path = self.project_root / lidar_path
            else:
                full_path = Path(lidar_path)
            
            if not full_path.exists():
                logger.debug(f"pointcloud file not found: {full_path}")
                return torch.zeros(self.config.scene_pointcloud_size, 3), torch.tensor(False)
            
            if HAS_OPEN3D:
                try:
                    pcd = o3d.io.read_point_cloud(str(full_path))
                    points = np.asarray(pcd.points, dtype=np.float32)
                except Exception as e:
                    logger.debug(f"failed to load pointcloud with Open3D: {e}")
                    return torch.zeros(self.config.scene_pointcloud_size, 3), torch.tensor(False)
            else:
                points = np.random.randn(1000, 3).astype(np.float32)
            
            if len(points) == 0:
                logger.debug("pointcloud is empty")
                return torch.zeros(self.config.scene_pointcloud_size, 3), torch.tensor(False)
            
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
            
            if len(points) == 0:
                logger.debug("filtered pointcloud is empty")
                return torch.zeros(self.config.scene_pointcloud_size, 3), torch.tensor(False)
            
            if len(points) > self.config.scene_pointcloud_size:
                indices = np.random.choice(len(points), self.config.scene_pointcloud_size, replace=False)
                points = points[indices]
            elif len(points) < self.config.scene_pointcloud_size:
                n_repeat = self.config.scene_pointcloud_size // len(points) + 1
                points = np.tile(points, (n_repeat, 1))[:self.config.scene_pointcloud_size]
            
            return torch.tensor(points, dtype=torch.float32), torch.tensor(True)
        
        except Exception as e:
            logger.debug(f"failed to load pointcloud {lidar_path}: {e}")
            return torch.zeros(self.config.scene_pointcloud_size, 3), torch.tensor(False)


class ThreadTimeout:
    def __init__(self, timeout_seconds):
        self.timeout = timeout_seconds
        self.result = None
        self.exception = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.debug(f"execution exception: {exc_type.__name__}: {exc_val}")
        return False


def fusion_collate_fn(batch: List[Optional[Dict[str, Any]]], pad_token_id: int) -> Optional[Dict[str, Any]]:
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        logger.warning("batch has no valid samples, returning None")
        return None
    
    if len(valid_batch) < len(batch):
        logger.debug(f"batch has {len(batch) - len(valid_batch)} invalid samples filtered")
    
    batch_size = len(valid_batch)
    
    try:
        max_length = max(item['input_ids'].shape[0] for item in valid_batch)
        
        input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        
        for i, item in enumerate(valid_batch):
            seq_len = item['input_ids'].shape[0]
            input_ids[i, :seq_len] = item['input_ids']
            attention_mask[i, :seq_len] = item['attention_mask']
            labels[i, :seq_len] = item['labels']
        
        try:
            rgb_images = torch.stack([item['rgb_images'] for item in valid_batch])
            rgb_mask = torch.stack([item['rgb_mask'] for item in valid_batch])
            depth_images = torch.stack([item['depth_images'] for item in valid_batch])
            depth_mask = torch.stack([item['depth_mask'] for item in valid_batch])
            event_images = torch.stack([item['event_images'] for item in valid_batch])
            event_mask = torch.stack([item['event_mask'] for item in valid_batch])
            scene_pointcloud = torch.stack([item['scene_pointcloud'] for item in valid_batch])
            scene_mask = torch.stack([item['scene_mask'] for item in valid_batch])
        except Exception as e:
            logger.error(f"failed to stack multimodal data: {e}")
            return None
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'rgb_images': rgb_images,
            'rgb_mask': rgb_mask,
            'depth_images': depth_images,
            'depth_mask': depth_mask,
            'event_images': event_images,
            'event_mask': event_mask,
            'scene_pointcloud': scene_pointcloud,
            'scene_mask': scene_mask,
            'original_text': [item['original_text'] for item in valid_batch],
            'question': [item.get('question', 'Unknown question') for item in valid_batch],
            'answer': [item.get('answer', 'Unknown answer') for item in valid_batch]
        }
        
        if 'scene_id' in valid_batch[0]:
            result['scene_id'] = [item.get('scene_id', f'scene_{i}') for i, item in enumerate(valid_batch)]
        if 'condition' in valid_batch[0]:
            result['condition'] = [item.get('condition', 'normal') for item in valid_batch]
        if 'timestamp' in valid_batch[0]:
            result['timestamp'] = [item.get('timestamp', 'unknown') for item in valid_batch]
        if 'sample_metadata' in valid_batch[0]:
            result['sample_metadata'] = [item.get('sample_metadata', {}) for item in valid_batch]
        
        return result
        
    except Exception as e:
        logger.error(f"failed to execute collate function: {e}")
        return None


def create_fusion_dataloader(
    config: FusionDataConfig, 
    tokenizer, 
    split: str, 
    batch_size: int, 
    num_workers: int, 
    shuffle: bool, 
    json_file: str = None,
    model_type: str = 'baseline',
    enable_multiview_fusion: bool = False,
    enable_metadata_extraction: bool = False
) -> DataLoader:
    logger.info(f"creating {split} dataloader (model_type={model_type}, enable_multiview_fusion={enable_multiview_fusion}, enable_metadata_extraction={enable_metadata_extraction})")
    
    try:
        dataset = FusionDataset(config, tokenizer, split, model_type, enable_multiview_fusion, enable_metadata_extraction)
    except Exception as e:
        logger.error(f"failed to create dataset: {e}")
        raise
    
    collate_fn = partial(fusion_collate_fn, pad_token_id=tokenizer.pad_token_id)
    
    actual_num_workers = min(4, max(0, num_workers))
    
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'num_workers': actual_num_workers,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False,
        'persistent_workers': actual_num_workers > 0,
    }
    
    if actual_num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2
    
    logger.info(f"dataloader configuration: batch_size={batch_size}, num_workers={actual_num_workers}, shuffle={shuffle}")
    
    try:
        dataloader = DataLoader(**dataloader_kwargs)
        logger.info(f"{split} dataloader created successfully: {len(dataset)} samples, {len(dataloader)} batches")
        return dataloader
    except Exception as e:
        logger.error(f"failed to create dataloader: {e}")
        raise


def test_dataloader(config: FusionDataConfig, tokenizer, split: str = 'train', num_samples: int = 3):
    logger.info(f"testing dataloader ({split})")
    
    try:
        dataset = FusionDataset(config, tokenizer, split)
        logger.info(f"dataset created successfully: {len(dataset)} samples")
        
        for i in range(min(num_samples, len(dataset))):
            logger.info(f"testing sample {i+1}/{num_samples}")
            sample = dataset[i]
            
            if sample is None:
                logger.warning(f"sample {i} is None")
                continue
            
            logger.info(f"sample {i}:")
            logger.info(f"    - Input IDs: {sample['input_ids'].shape}")
            logger.info(f"    - Labels: {sample['labels'].shape}")
            logger.info(f"    - RGB: {sample['rgb_images'].shape}, mask: {sample['rgb_mask'].sum()}")
            logger.info(f"    - Depth: {sample['depth_images'].shape}, mask: {sample['depth_mask'].sum()}")
            logger.info(f"    - Event: {sample['event_images'].shape}, mask: {sample['event_mask'].sum()}")
            logger.info(f"    - Scene: {sample['scene_pointcloud'].shape}, mask: {sample['scene_mask']}")
        
        dataloader = create_fusion_dataloader(
            config, tokenizer, split, batch_size=2, num_workers=0, shuffle=False
        )
        
        logger.info("testing batch loading...")
        batch = next(iter(dataloader))
        
        if batch is None:
            logger.error("first batch is None")
        else:
            logger.info(f"batch loading successfully:")
            logger.info(f"  - Batch size: {batch['input_ids'].shape[0]}")
            logger.info(f"  - Sequence length: {batch['input_ids'].shape[1]}")
            logger.info(f"  - RGB shape: {batch['rgb_images'].shape}")
            logger.info(f"  - Depth shape: {batch['depth_images'].shape}")
            logger.info(f"  - Event shape: {batch['event_images'].shape}")
            logger.info(f"  - Scene shape: {batch['scene_pointcloud'].shape}")
            
        
    except Exception as e:
        logger.error(f"failed to test dataloader: {e}")
        import traceback
        traceback.print_exc()
        raise


def analyze_dataset_statistics(data_dir: str):
    logger.info(f"analyzing dataset statistics: {data_dir}")
    
    data_path = Path(data_dir)
    split_file = data_path / "split_index.json"
    
    if not split_file.exists():
        logger.error(f"split index file not found: {split_file}")
        return
    
    try:
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        train_batches = split_data.get('train_batches', [])
        val_batches = split_data.get('val_batches', [])
        train_samples = split_data.get('train_samples', 0)
        val_samples = split_data.get('val_samples', 0)
        total_samples = split_data.get('total_samples', 0)
        
        logger.info("dataset statistics:")
        logger.info(f"  training set: {len(train_batches)} batches, {train_samples} samples")
        logger.info(f"  validation set: {len(val_batches)} batches, {val_samples} samples")
        logger.info(f"  total: {total_samples} samples")
        
        data_packed_dir = data_path / "data_packed"
        if data_packed_dir.exists():
            total_size = 0
            file_count = 0
            
            for batch_file in data_packed_dir.glob("batch_*.json"):
                size = batch_file.stat().st_size
                total_size += size
                file_count += 1
            
            avg_size = total_size / max(file_count, 1) / 1024 / 1024  # MB
            total_size_gb = total_size / 1024 / 1024 / 1024  # GB
            
            logger.info(f"  data size: {total_size_gb:.2f} GB ({file_count} files)")
            logger.info(f"  average file size: {avg_size:.1f} MB")
        
        if train_batches:
            first_batch_file = data_packed_dir / train_batches[0]
            if first_batch_file.exists():
                try:
                    with open(first_batch_file, 'r') as f:
                        batch_data = json.load(f)
                    
                    if batch_data and len(batch_data) > 0:
                        sample = batch_data[0]
                        modality_data = sample.get('modality_data', {})
                        
                        logger.info("modal information (based on the first sample):")
                        
                        # RGB信息
                        rgb_data = modality_data.get('rgb', {})
                        rgb_views = len([v for v in rgb_data.values() if v])
                        logger.info(f"  RGB: {rgb_views}/6 views")
                        
                        # Depth信息
                        depth_data = modality_data.get('depth', {})
                        depth_views = len([v for v in depth_data.values() if v])
                        logger.info(f"  Depth: {depth_views}/6 views")
                        
                        # Event信息
                        event_data = modality_data.get('event', {})
                        event_views = len([v for v in event_data.values() if v])
                        logger.info(f"  Event: {event_views}/4 views")
                        
                        # LiDAR信息
                        lidar_data = modality_data.get('lidar')
                        logger.info(f"  LiDAR: {'yes' if lidar_data else 'no'}")
                        
                except Exception as e:
                    logger.warning(f"failed to analyze first batch: {e}")
        
    except Exception as e:
        logger.error(f"failed to analyze dataset statistics: {e}")


def check_data_integrity(data_dir: str, max_batches: int = 5):
    logger.info(f"checking data integrity: {data_dir} (checking first {max_batches} batches)")
    
    data_path = Path(data_dir)
    data_packed_dir = data_path / "data_packed"
    
    if not data_packed_dir.exists():
        logger.error(f"packed data directory not found: {data_packed_dir}")
        return False
    
    batch_files = list(data_packed_dir.glob("batch_*.json"))
    if not batch_files:
        logger.error("no batch files found")
        return False
    
    check_files = batch_files[:max_batches]
    issues = []
    
    for i, batch_file in enumerate(check_files):
        logger.info(f"checking {batch_file.name}...")
        
        try:
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            
            if not isinstance(batch_data, list):
                issues.append(f"{batch_file.name}: not a list format")
                continue
            
            if len(batch_data) == 0:
                issues.append(f"{batch_file.name}: empty file")
                continue
            
            sample = batch_data[0]
            required_keys = ['question', 'answer', 'modality_data']
            missing_keys = [key for key in required_keys if key not in sample]
            
            if missing_keys:
                issues.append(f"{batch_file.name}: missing keys {missing_keys}")
            
            modality_data = sample.get('modality_data', {})
            if not isinstance(modality_data, dict):
                issues.append(f"{batch_file.name}: modality_data is not a dictionary")
                continue
            
            has_data = any([
                modality_data.get('rgb'),
                modality_data.get('depth'),
                modality_data.get('event'),
                modality_data.get('lidar')
            ])
            
            if not has_data:
                issues.append(f"{batch_file.name}: no modality data")
            
            logger.info(f"    {len(batch_data)} samples")
            
        except json.JSONDecodeError as e:
            issues.append(f"{batch_file.name}: JSON decode error - {e}")
        except Exception as e:
            issues.append(f"{batch_file.name}: other error - {e}")
    
    if issues:
        logger.warning("found data integrity issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("data integrity check passed")
        return True


def validate_image_paths(data_dir: str, max_samples: int = 10):
    logger.info(f"validating image paths (checking first {max_samples} samples)")
    
    data_path = Path(data_dir)
    project_root = data_path.parent
    data_packed_dir = data_path / "data_packed"
    
    batch_files = list(data_packed_dir.glob("batch_*.json"))
    if not batch_files:
        logger.error("no batch files found")
        return False
    
    try:
        with open(batch_files[0], 'r') as f:
            batch_data = json.load(f)
        
        test_samples = batch_data[:max_samples]
        path_issues = []
        
        for i, sample in enumerate(test_samples):
            modality_data = sample.get('modality_data', {})
            
            # check RGB path
            rgb_data = modality_data.get('rgb', {})
            for view, path in rgb_data.items():
                if path:
                    full_path = project_root / path if not os.path.isabs(path) else Path(path)
                    if not full_path.exists():
                        path_issues.append(f"sample {i} RGB {view}: {path}")
            
            # check Depth path
            depth_data = modality_data.get('depth', {})
            for view, path in depth_data.items():
                if path:
                    full_path = project_root / path if not os.path.isabs(path) else Path(path)
                    if not full_path.exists():
                        path_issues.append(f"sample {i} Depth {view}: {path}")
            
            # check Event path
            event_data = modality_data.get('event', {})
            for view, path in event_data.items():
                if path:
                    full_path = project_root / path if not os.path.isabs(path) else Path(path)
                    if not full_path.exists():
                        path_issues.append(f"sample {i} Event {view}: {path}")
            
            # check LiDAR path
            lidar_path = modality_data.get('lidar')
            if lidar_path:
                full_path = project_root / lidar_path if not os.path.isabs(lidar_path) else Path(lidar_path)
                if not full_path.exists():
                    path_issues.append(f"sample {i} LiDAR: {lidar_path}")
        
        if path_issues:
            logger.warning(f"found {len(path_issues)} invalid paths:")
            for issue in path_issues[:5]:  
                logger.warning(f"  - {issue}")
            if len(path_issues) > 5:
                logger.warning(f"  ... there are {len(path_issues) - 5} more issues")
            return False
        else:
            logger.info("image paths validation passed")
            return True
            
    except Exception as e:
        logger.error(f"failed to validate image paths: {e}")
        return False


# main test function
def main():
    logger.info("starting data loader test")
    
    # basic configuration
    config = FusionDataConfig(data_dir="./data_rebuilt")
    
    # run various tests
    try:
        # 1. analyze dataset statistics
        analyze_dataset_statistics(config.data_dir)
        
        # 2. check data integrity
        check_data_integrity(config.data_dir)
        
        # 3. validate image paths
        validate_image_paths(config.data_dir)
        
        # 4. test data loader
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        
        test_dataloader(config, tokenizer, 'train', num_samples=3)
        
        logger.info("all tests completed")
        
    except Exception as e:
        logger.error(f"test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()