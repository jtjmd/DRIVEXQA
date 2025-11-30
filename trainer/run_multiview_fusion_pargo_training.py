#!/usr/bin/env python3


import os
import sys
import argparse
import logging
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

from configs import get_multiview_fusion_pargo_config, FusionDataConfig
from leo_authentic_trainer import LEOTrainer
from fusion_data_loader import create_fusion_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='MultiView-Fusion-Pargo model training')
    
    parser.add_argument('--batch_size', type=int, default=4,
                       help='batch size (default: 4)')
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='learning rate (default: 2e-5)')

    parser.add_argument('--num_epochs', type=int, default=5,
                       help='training epochs (default: 5)')
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='gradient accumulation steps (default: 4)')
    
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='warmup steps (default: 100)')
    
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='save steps (default: 1000)')
    
    parser.add_argument('--eval_steps', type=int, default=250,
                       help='evaluation steps (default: 250)')
    
    # LoRA参数
    parser.add_argument('--use_lora', action='store_true',
                       help='use LoRA')
    
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank (default: 16)')
    
    # 路径参数
    parser.add_argument('--data_dir', type=str, default='./data_rebuilt',
                       help='data directory (default: ./data_rebuilt)')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='output directory (if not specified, will use the default value in the configuration)')
    
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='resume from checkpoint')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = get_multiview_fusion_pargo_config()
    
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.warmup_steps = args.warmup_steps
    config.save_steps = args.save_steps
    config.eval_steps = args.eval_steps
    config.use_lora = args.use_lora
    config.lora_rank = args.lora_rank
    config.resume_from_checkpoint = args.resume_from_checkpoint
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    logger.info(f"MultiView-Fusion-Pargo training configuration:")
    logger.info(f"   data directory: {config.data_dir}")
    logger.info(f"   output directory: {config.output_dir}")
    logger.info(f"   batch size: {config.batch_size}")
    logger.info(f"   learning rate: {config.learning_rate}")
    logger.info(f"   training epochs: {config.num_epochs}")
    logger.info(f"   ParGo token number: {config.pargo_num_tokens}")
    logger.info(f"   Partial layers: {config.pargo_partial_layers}")
    logger.info(f"   Global layers: {config.pargo_global_layers}")
    logger.info(f"   fusion dimension: {config.pargo_fusion_dim}")
    logger.info(f"   temperature: {config.pargo_temperature}")
    logger.info(f"   use LoRA: {config.use_lora}")
    if config.use_lora:
        logger.info(f"   LoRA rank: {config.lora_rank}")
    logger.info(f"initializing LEO trainer...")
    trainer = LEOTrainer(config)
    
    data_config = FusionDataConfig(
        data_dir=config.data_dir,
        enabled_modalities=config.enabled_modalities,
        image_size=config.image_size,
        num_rgb_views=config.num_rgb_views,
        num_depth_views=config.num_depth_views,
        num_event_views=config.num_event_views,
        pointcloud_mode=config.pointcloud_mode,
        scene_pointcloud_size=config.scene_pointcloud_size,
        object_pointcloud_size=config.point_cloud_size,
        max_length=config.max_txt_len
    )
    
    logger.info(f"creating data loader...")
    train_dataloader = create_fusion_dataloader(
        config=data_config,
        tokenizer=trainer.tokenizer,
        split='train',
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        model_type=config.model_type,
        enable_multiview_fusion=True  
    )
    
    val_dataloader = create_fusion_dataloader(
        config=data_config,
        tokenizer=trainer.tokenizer,
        split='val',
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        model_type=config.model_type,
        enable_multiview_fusion=True  
    )
    
    logger.info(f"data loader created successfully")
    logger.info(f"   train batch size: {len(train_dataloader)}")
    logger.info(f"   eval batch size: {len(val_dataloader)}")
    
    logger.info(f"starting MultiView-Fusion-Pargo model training...")
    trainer.train(train_dataloader, val_dataloader)
    
    logger.info(f"MultiView-Fusion-Pargo training completed!")
    logger.info(f"model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()