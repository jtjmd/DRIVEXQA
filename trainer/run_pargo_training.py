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
    output_dir: str = './checkpoints_pargo'
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
    
    model_type: str = 'pargo'
    pargo_num_tokens: int = 64      
    pargo_partial_layers: int = 2    
    pargo_global_layers: int = 2     
    pargo_fusion_dim: int = 512      
    pargo_temperature: float = 0.1   
    
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 250
    max_eval_batches: int = 20
    log_grad_norm: bool = True
    max_eval_steps: Optional[int] = None

    def __post_init__(self):
        if self.model_type == 'pargo':
            if self.pargo_num_tokens <= 0:
                self.pargo_num_tokens = 64
            if self.pargo_partial_layers <= 0:
                self.pargo_partial_layers = 2
            if self.pargo_global_layers <= 0:
                self.pargo_global_layers = 2
            if self.pargo_fusion_dim <= 0:
                self.pargo_fusion_dim = 512
            if self.pargo_temperature <= 0:
                self.pargo_temperature = 0.1

        if self.enable_multiview_fusion:
            self.output_dir = './checkpoints_pargo_fusion'
        else:
            self.output_dir = './checkpoints_pargo'


def main():
    config = LEOConfig()
    
    import argparse
    parser = argparse.ArgumentParser(description='ParGo model training')
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
    parser.add_argument('--pargo_num_tokens', type=int, default=config.pargo_num_tokens, help='ParGo token数量')
    parser.add_argument('--pargo_partial_layers', type=int, default=config.pargo_partial_layers, help='Partial encoder layers')
    parser.add_argument('--pargo_global_layers', type=int, default=config.pargo_global_layers, help='Global encoder layers')
    parser.add_argument('--pargo_fusion_dim', type=int, default=config.pargo_fusion_dim, help='fusion dimension')
    parser.add_argument('--pargo_temperature', type=float, default=config.pargo_temperature, help='attention temperature')
    
    parser.add_argument('--enable_multiview_fusion', action='store_true', help='enable multiview fusion')
    parser.add_argument('--multiview_fusion_views', type=int, default=config.multiview_fusion_views, help='multiview fusion views')
    
    parser.add_argument('--use_lora', action='store_true', default=config.use_lora, help='use LoRA')
    parser.add_argument('--lora_rank', type=int, default=config.lora_rank, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=config.lora_alpha, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=config.lora_dropout, help='lora dropout')
    
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
    config.pargo_num_tokens = args.pargo_num_tokens
    config.pargo_partial_layers = args.pargo_partial_layers
    config.pargo_global_layers = args.pargo_global_layers
    config.pargo_fusion_dim = args.pargo_fusion_dim
    config.pargo_temperature = args.pargo_temperature
    config.enable_multiview_fusion = args.enable_multiview_fusion
    config.multiview_fusion_views = args.multiview_fusion_views
    config.use_lora = args.use_lora
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout

    logger.info("=" * 50)
    logger.info("starting ParGo model training")
    logger.info("=" * 50)
    logger.info(f"model type: {config.model_type}")
    logger.info(f"data directory: {config.data_dir}")
    logger.info(f"output directory: {config.output_dir}")
    logger.info(f"LLM model: {config.llm_name_or_path}")
    logger.info(f"training epochs: {config.num_epochs}")
    logger.info(f"batch size: {config.batch_size}")
    logger.info(f"learning rate: {config.learning_rate}")
    logger.info("ParGo specific configuration:")
    logger.info(f"  - token number: {config.pargo_num_tokens}")
    logger.info(f"  - Partial encoder layers: {config.pargo_partial_layers}")
    logger.info(f"  - Global encoder layers: {config.pargo_global_layers}")
    logger.info(f"  - fusion dimension: {config.pargo_fusion_dim}")
    logger.info(f"  - attention temperature: {config.pargo_temperature}")
    
    logger.info("multiview fusion configuration:")
    logger.info(f"  - enable multiview fusion: {config.enable_multiview_fusion}")
    if config.enable_multiview_fusion:
        logger.info(f"  - number of views: {config.multiview_fusion_views}")
        logger.info(f"  - layout: [FRONT|BACK; LEFT|RIGHT]")
        logger.info(f"  - each modality is independently concatenated into a 2x2 grid")
    
    logger.info("LoRA configuration:")
    logger.info(f"  - use LoRA: {config.use_lora}")
    if config.use_lora:
        logger.info(f"  - lora rank: {config.lora_rank}")
        logger.info(f"  - lora alpha: {config.lora_alpha}")
        logger.info(f"  - lora dropout: {config.lora_dropout}")
    
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
        
        data_config = FusionDataConfig.from_leo_config(config)
        
        logger.info("initializing ParGo trainer...")
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
        
        val_loader = create_fusion_dataloader(
            config=data_config,
            tokenizer=trainer.tokenizer,
            split='val',
            batch_size=config.batch_size,
            num_workers=2,
            shuffle=False,
            model_type=config.model_type,
            enable_multiview_fusion=config.enable_multiview_fusion
        )
        logger.info(f"validation data loader created successfully, batch size: {len(val_loader)}")
        
        logger.info("starting ParGo model training...")
        trainer.train(train_loader, val_loader)
        
        logger.info("ParGo model training completed!")
        
        logger.info("saving final model...")
        trainer.save_checkpoint(0, is_final=True, epoch_suffix='final')
        logger.info(f"final model saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"error during training: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    logger.info("ParGo model training completed!")



if __name__ == "__main__":
    main()