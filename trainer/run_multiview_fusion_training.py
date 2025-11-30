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
    output_dir: str = './checkpoints_multiview_fusion'
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

    model_type: str = 'multiview_fusion'
    multiview_fusion_views: int = 4           
    multiview_fusion_size: int = 448          
    multiview_spatial_dim: int = 49           
    multiview_channel_dim: int = 2304         
    multiview_hidden_dim: int = 512           
    multiview_spatial_heads: int = 8          
    multiview_channel_heads: int = 8          
    multiview_dropout: float = 0.1            
    fusion_variant: str = 'gap'               
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 250
    max_eval_batches: int = 20
    log_grad_norm: bool = True
    max_eval_steps: Optional[int] = None


def main():
    config = LEOConfig()
    
    import argparse
    parser = argparse.ArgumentParser(description='MultiView-Fusion model training')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data directory')
    parser.add_argument('--output_dir', type=str, default=config.output_dir, help='output directory')
    parser.add_argument('--llm_name_or_path', type=str, default=config.llm_name_or_path, help='LLM model path')
    parser.add_argument('--num_epochs', type=int, default=config.num_epochs, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=config.gradient_accumulation_steps, help='gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=config.warmup_steps, help='warmup steps')
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay, help='weight decay')
    parser.add_argument('--resume_from_checkpoint', type=str, default=config.resume_from_checkpoint, help='resume from checkpoint')
    parser.add_argument('--max_train_steps', type=int, default=config.max_train_steps, help='max training steps')
    parser.add_argument('--save_steps', type=int, default=config.save_steps, help='save steps')
    parser.add_argument('--eval_steps', type=int, default=config.eval_steps, help='evaluation steps')
    parser.add_argument('--logging_steps', type=int, default=config.logging_steps, help='logging steps')
    parser.add_argument('--max_eval_batches', type=int, default=config.max_eval_batches, help='max evaluation batches')
    parser.add_argument('--use_lora', action='store_true', help='use LoRA')
    parser.add_argument('--lora_rank', type=int, default=config.lora_rank, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=config.lora_alpha, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=config.lora_dropout, help='LoRA dropout')
    parser.add_argument('--multiview_hidden_dim', type=int, default=config.multiview_hidden_dim, help='Cross Attention hidden dimension')
    parser.add_argument('--multiview_spatial_heads', type=int, default=config.multiview_spatial_heads, help='Spatial attention heads')
    parser.add_argument('--multiview_channel_heads', type=int, default=config.multiview_channel_heads, help='Channel attention heads')
    parser.add_argument('--multiview_dropout', type=float, default=config.multiview_dropout, help='Dropout rate')
    parser.add_argument('--fusion_variant', type=str, default=config.fusion_variant,
                        choices=['gap', 'qattn', 'qattn_spectral', 'qattn_depthgate'],
                        help='MultiView-Fusion aggregation variant (output still single token)')
    parser.add_argument('--fusion_token_layout', type=str, default='single',
                        choices=['single','triple'],
                        help='Output fusion token layout: single(1) or triple(3)')
    
    args = parser.parse_args()
    
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.llm_name_or_path = args.llm_name_or_path
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.warmup_steps = args.warmup_steps
    config.weight_decay = args.weight_decay
    config.resume_from_checkpoint = args.resume_from_checkpoint
    config.max_train_steps = args.max_train_steps
    config.save_steps = args.save_steps
    config.eval_steps = args.eval_steps
    config.logging_steps = args.logging_steps
    config.max_eval_batches = args.max_eval_batches
    config.use_lora = args.use_lora
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    
    config.multiview_hidden_dim = args.multiview_hidden_dim
    config.multiview_spatial_heads = args.multiview_spatial_heads
    config.multiview_channel_heads = args.multiview_channel_heads
    config.multiview_dropout = args.multiview_dropout
    config.fusion_variant = args.fusion_variant
    config.fusion_token_layout = args.fusion_token_layout
    config.fusion_num_tokens = 3 if args.fusion_token_layout == 'triple' else 1
    
    logger.info("=" * 50)
    logger.info("starting MultiView-Fusion model training")
    logger.info("=" * 50)
    logger.info(f"model type: {config.model_type}")
    logger.info(f"data directory: {config.data_dir}")
    logger.info(f"output directory: {config.output_dir}")
    logger.info(f"LLM model: {config.llm_name_or_path}")
    logger.info(f"training epochs: {config.num_epochs}")
    logger.info(f"batch size: {config.batch_size}")
    logger.info(f"learning rate: {config.learning_rate}")
    logger.info(f"gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"warmup steps: {config.warmup_steps}")
    logger.info(f"weight decay: {config.weight_decay}")
    
    logger.info("MultiView-Fusion specific configuration:")
    logger.info(f"  - number of views: {config.multiview_fusion_views}")
    logger.info(f"  - fusion image size: {config.multiview_fusion_size}x{config.multiview_fusion_size}")
    logger.info(f"  - spatial dimension: {config.multiview_spatial_dim}")
    logger.info(f"  - channel dimension: {config.multiview_channel_dim}")
    logger.info(f"  - hidden dimension: {config.multiview_hidden_dim}")
    logger.info(f"  - spatial attention heads: {config.multiview_spatial_heads}")
    logger.info(f"  - channel attention heads: {config.multiview_channel_heads}")
    logger.info(f"  - dropout rate: {config.multiview_dropout}")
    logger.info(f"  - aggregation variant: {config.fusion_variant}")
    logger.info(f"  - output token layout: {config.fusion_token_layout} (num={config.fusion_num_tokens})")
    
    logger.info("LoRA configuration:")
    logger.info(f"  - use LoRA: {config.use_lora}")
    if config.use_lora:
        logger.info(f"  - LoRA rank: {config.lora_rank}")
        logger.info(f"  - LoRA alpha: {config.lora_alpha}")
        logger.info(f"  - LoRA dropout: {config.lora_dropout}")
    
    logger.info("modalities configuration:")
    logger.info(f"  - enable modalities: {config.enabled_modalities}")
    logger.info(f"  - RGB views: {config.num_rgb_views}")
    logger.info(f"  - Depth views: {config.num_depth_views}")
    logger.info(f"  - Event views: {config.num_event_views}")
    logger.info(f"  - point cloud mode: {config.pointcloud_mode}")
    logger.info(f"  - scene point cloud size: {config.scene_pointcloud_size}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"output directory created: {config.output_dir}")
    
    try:
        from leo_authentic_trainer import LEOTrainer
        from fusion_data_loader import create_fusion_dataloader, FusionDataConfig
        
        logger.info("initializing MultiView-Fusion trainer...")
        trainer = LEOTrainer(config)
        logger.info("trainer created successfully")
        
        data_config = FusionDataConfig.from_leo_config(config)
        
        logger.info("creating data loader...")
        
        train_loader = create_fusion_dataloader(
            config=data_config,
            tokenizer=trainer.tokenizer,
            split='train',
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            model_type=config.model_type,
            enable_multiview_fusion=True  
        )
        logger.info(f"train data loader created successfully, batch size: {len(train_loader)}")
        
        val_loader = create_fusion_dataloader(
            config=data_config,
            tokenizer=trainer.tokenizer,
            split='val',
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            model_type=config.model_type,
            enable_multiview_fusion=True  
        )
        logger.info(f"validation data loader created successfully, batch size: {len(val_loader)}")
        
        logger.info("starting training...")
        trainer.train(train_loader, val_loader)
        
        logger.info("MultiView-Fusion training completed!")
        
    except ImportError as e:
        logger.error(f"import error: {e}")
        logger.error("please ensure all dependencies are in the correct path")
        return 1
    except Exception as e:
        logger.error(f"error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 