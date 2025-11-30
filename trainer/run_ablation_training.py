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

from configs import get_ablation_config, FusionDataConfig
from leo_authentic_trainer import LEOTrainer
from fusion_data_loader import create_fusion_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='MultiView-Ablation')
    
    parser.add_argument('--modalities', type=str, nargs='+', 
                       choices=['rgb', 'depth', 'event'],
                       default=['rgb', 'depth'],
                       help='modality list for 2D fusion (default: rgb depth)')
    
    parser.add_argument('--enable_lidar', action='store_true',
                       help='enable LiDAR independent token')
    
    parser.add_argument('--batch_size', type=int, default=4,
                       help='batch size (default: 4)')
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='learning rate (default: 2e-5)')
    
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='number of epochs (default: 5)')
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='gradient accumulation steps (default: 4)')
    
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='warmup steps (default: 100)')
    
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='save checkpoint steps (default: 1000)')
    
    parser.add_argument('--eval_steps', type=int, default=250,
                       help='evaluation steps (default: 250)')
    
    parser.add_argument('--use_lora', action='store_true',
                       help='use LoRA fine-tuning')
    
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank (default: 16)')
    
    parser.add_argument('--data_dir', type=str, default='./data_rebuilt',
                       help='data directory (default: ./data_rebuilt)')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='output directory (if not specified, will be generated automatically)')
    
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='resume training from checkpoint')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"creating ablation configuration...")
    logger.info(f"   2D modalities: {args.modalities}")
    logger.info(f"   LiDAR enabled: {args.enable_lidar}")
    
    config = get_ablation_config(
        modalities_2d=args.modalities,
        enable_lidar=args.enable_lidar
    )
    
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.warmup_steps = args.warmup_steps
    config.save_steps = args.save_steps
    config.eval_steps = args.eval_steps
    config.use_lora = args.use_lora
    config.lora_rank = args.lora_rank
    config.data_dir = args.data_dir
    config.resume_from_checkpoint = args.resume_from_checkpoint
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    logger.info(f"ablation configuration:")
    logger.info(f"   model type: {config.model_type}")
    logger.info(f"   2D modalities: {config.ablation_2d_modalities}")
    logger.info(f"   LiDAR enabled: {config.ablation_enable_lidar}")
    logger.info(f"   enabled modalities: {config.enabled_modalities}")
    logger.info(f"   output directory: {config.output_dir}")
    logger.info(f"   batch size: {config.batch_size}")
    logger.info(f"   learning rate: {config.learning_rate}")
    logger.info(f"   number of epochs: {config.num_epochs}")
    logger.info(f"   gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"   warmup steps: {config.warmup_steps}")
    logger.info(f"   save steps: {config.save_steps}")
    logger.info(f"   evaluation steps: {config.eval_steps}")
    logger.info(f"   LoRA: {config.use_lora} (rank: {config.lora_rank})")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info(f"initializing trainer...")
    trainer = LEOTrainer(config)
    
    logger.info(f"creating data configuration...")
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
        max_length=config.max_txt_len,
        ablation_enable_lidar=config.ablation_enable_lidar  
    )
    
    logger.info("data configuration:")
    logger.info(f"   enabled_modalities: {data_config.enabled_modalities}")
    logger.info(f"   pointcloud_mode: {data_config.pointcloud_mode}")
    logger.info(f"   scene_pointcloud_size: {data_config.scene_pointcloud_size}")
    logger.info(f"   ablation_enable_lidar: {data_config.ablation_enable_lidar}")  
    
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
    
    logger.info(f"starting training...")
    trainer.train(train_dataloader, val_dataloader)
    
    logger.info(f"training completed!")
    logger.info(f"checkpoints saved to: {config.output_dir}")

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        logger.info("training interrupted by user")
    except Exception as e:
        logger.error(f"training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 