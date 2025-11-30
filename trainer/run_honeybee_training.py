#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LEOConfig:
    data_dir: str = './data_rebuilt'
    output_dir: str = './checkpoints_honeybee'
    llm_name_or_path: str = 'lmsys/vicuna-7b-v1.5'
    resume_from_checkpoint: Optional[str] = None
    
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_train_steps: Optional[int] = None
    num_workers: int = 4
    
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    train_file: str = 'train_multiview.json'
    validation_file: str = 'val_multiview.json'
    max_txt_len: int = 128
    image_size: int = 224
    point_cloud_size: int = 8192
    num_rgb_views: int = 4
    num_depth_views: int = 4
    num_event_views: int = 4
    max_obj_len: int = 0
    enabled_modalities: List[str] = field(default_factory=lambda: ['rgb', 'depth', 'event', 'pointcloud'])
    pointcloud_mode: str = 'scene'
    scene_pointcloud_size: int = 16384
    
 
    model_type: str = 'honeybee'
    honeybee_num_tokens: int = 64   
    honeybee_c_abs_layers: int = 2  
    honeybee_pooling_size: int = 2   
    
    enable_multiview_fusion: bool = False      
    multiview_fusion_views: int = 4            
    multiview_fusion_size: int = 448           
    
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 250
    max_eval_batches: int = 20
    log_grad_norm: bool = True
    max_eval_steps: Optional[int] = None

    def __post_init__(self):
        if self.model_type == 'honeybee':
            if self.honeybee_num_tokens <= 0:
                self.honeybee_num_tokens = 64
            if self.honeybee_c_abs_layers <= 0:
                self.honeybee_c_abs_layers = 2
        
        if self.enable_multiview_fusion:
            self.output_dir = './checkpoints_honeybee_fusion'
        else:
            self.output_dir = './checkpoints_honeybee'


def main():
    config = LEOConfig()
    
    import argparse
    parser = argparse.ArgumentParser(description='Honeybee multimodal Vicuna training')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data directory')
    parser.add_argument('--output_dir', type=str, default=config.output_dir, help='output directory')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=config.num_epochs, help='training epochs')
    parser.add_argument('--max_train_steps', type=int, default=config.max_train_steps, help='max train steps')
    parser.add_argument('--warmup_steps', type=int, default=config.warmup_steps, help='warmup steps')
    parser.add_argument('--save_steps', type=int, default=config.save_steps, help='save steps')
    parser.add_argument('--eval_steps', type=int, default=config.eval_steps, help='evaluation steps')
    parser.add_argument('--use_lora', action='store_true', help='use LoRA')
    parser.add_argument('--lora_rank', type=int, default=config.lora_rank, help='LoRA rank')
    
    parser.add_argument('--honeybee_num_tokens', type=int, default=config.honeybee_num_tokens, help='Honeybee token number')
    parser.add_argument('--honeybee_c_abs_layers', type=int, default=config.honeybee_c_abs_layers, help='C-Abstractor layers')
    parser.add_argument('--honeybee_pooling_size', type=int, default=config.honeybee_pooling_size, help='pooling size')
    
    parser.add_argument('--enable_multiview_fusion', action='store_true', help='enable multiview fusion')
    parser.add_argument('--multiview_fusion_views', type=int, default=config.multiview_fusion_views, help='number of views')
    
    args = parser.parse_args()
    
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.max_train_steps = args.max_train_steps
    config.warmup_steps = args.warmup_steps
    config.save_steps = args.save_steps
    config.eval_steps = args.eval_steps
    config.use_lora = args.use_lora
    config.lora_rank = args.lora_rank
    config.honeybee_num_tokens = args.honeybee_num_tokens
    config.honeybee_c_abs_layers = args.honeybee_c_abs_layers
    config.honeybee_pooling_size = args.honeybee_pooling_size
    config.enable_multiview_fusion = args.enable_multiview_fusion
    config.multiview_fusion_views = args.multiview_fusion_views
    
    logger.info("=" * 50)
    logger.info("starting Honeybee multimodal Vicuna training")
    logger.info("=" * 50)
    logger.info(f"model type: {config.model_type}")
    logger.info(f"data directory: {config.data_dir}")
    logger.info(f"output directory: {config.output_dir}")
    logger.info(f"enabled modalities: {config.enabled_modalities}")
    logger.info(f"pointcloud mode: {config.pointcloud_mode}")
    logger.info(f"batch size: {config.batch_size}")
    logger.info(f"learning rate: {config.learning_rate}")
    logger.info(f"training epochs: {config.num_epochs}")
    logger.info(f"max train steps: {config.max_train_steps}")
    logger.info(f"use LoRA: {config.use_lora}")
    if config.use_lora:
        logger.info(f"LoRA rank: {config.lora_rank}")
    logger.info("Honeybee特有配置:")
    logger.info(f"  - token number: {config.honeybee_num_tokens}")
    logger.info(f"  - C-Abstractor layers: {config.honeybee_c_abs_layers}")
    logger.info(f"  - pooling size: {config.honeybee_pooling_size}")

    logger.info("multiview fusion configuration:")
    logger.info(f"  - enable multiview fusion: {config.enable_multiview_fusion}")
    if config.enable_multiview_fusion:
        logger.info(f"  - number of views: {config.multiview_fusion_views}")
        logger.info(f"  - layout: [FRONT|BACK; LEFT|RIGHT]")
        logger.info(f"  - each modality is independently concatenated into a 2x2 grid")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    try:
        from leo_authentic_trainer import LEOTrainer
        from fusion_data_loader import create_fusion_dataloader, FusionDataConfig
        
        data_config = FusionDataConfig.from_leo_config(config)
        
        logger.info("creating trainer...")
        trainer = LEOTrainer(config)
        logger.info("trainer created successfully")
        
        logger.info("creating data loader...")
        train_loader = create_fusion_dataloader(
            config=data_config,
            tokenizer=trainer.tokenizer,
            split='train',
            batch_size=config.batch_size,
            num_workers=2,
            shuffle=True,
            model_type=config.model_type,
            enable_multiview_fusion=config.enable_multiview_fusion
        )
        logger.info(f"train data loader created successfully, batch size: {len(train_loader)}")
        
        eval_loader = create_fusion_dataloader(
            config=data_config,
            tokenizer=trainer.tokenizer,
            split='val',
            batch_size=config.batch_size,
            num_workers=2,
            shuffle=False,
            model_type=config.model_type,
            enable_multiview_fusion=config.enable_multiview_fusion
        )
        logger.info(f"eval data loader created successfully, batch size: {len(eval_loader)}")
        
        logger.info("starting training...")
        trainer.train(train_loader, eval_loader)
        
        logger.info("Honeybee training completed!")
        
    except Exception as e:
        logger.error(f"training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()