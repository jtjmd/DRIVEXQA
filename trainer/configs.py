from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LEOConfig:
    

    data_dir: str = './data_rebuilt'  
    output_dir: str = './checkpoints'
    llm_name_or_path: str = 'lmsys/vicuna-7b-v1.5'
    resume_from_checkpoint: Optional[str] = None
    

    num_epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
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
    
    model_type: str = 'baseline'  
    

    use_sq_hub: bool = False  
    use_ppx: bool = False     
    aux_modality_weight: float = 0.3  
    

    honeybee_num_tokens: int = 64    
    honeybee_c_abs_layers: int = 2   
    honeybee_pooling_size: int = 2   
    

    pargo_num_tokens: int = 64       
    pargo_partial_layers: int = 2   
    pargo_global_layers: int = 2     
    pargo_fusion_dim: int = 512      
    pargo_temperature: float = 0.1   
    

    multiview_fusion_views: int = 4           
    multiview_fusion_size: int = 448         
    multiview_spatial_dim: int = 49          
    multiview_channel_dim: int = 2304        
    multiview_hidden_dim: int = 512           
    multiview_spatial_heads: int = 8          
    multiview_channel_heads: int = 8          
    multiview_dropout: float = 0.1            
    fusion_variant: str = 'gap'
    fusion_token_layout: str = 'single'
    fusion_num_tokens: int = 1

    ablation_2d_modalities: List[str] = field(default_factory=lambda: ['rgb', 'depth', 'event'])  
    ablation_enable_lidar: bool = True        
    ablation_query_priority: List[str] = field(default_factory=lambda: ['rgb', 'depth', 'event'])  
    
    enable_multiview_fusion: bool = False     
    
    logging_steps: int = 100    
    save_steps: int = 1000      
    eval_steps: int = 500       
    max_eval_batches: int = 20  
    log_grad_norm: bool = True
    max_eval_steps: Optional[int] = None

    def __post_init__(self):
        if self.model_type == 'cmnext':
            self.use_sq_hub = True
            self.use_ppx = True
            self.output_dir = './checkpoints_cmnext'
        elif self.model_type == 'honeybee':
            self.output_dir = './checkpoints_honeybee'
            if self.honeybee_num_tokens <= 0:
                self.honeybee_num_tokens = 64
            if self.honeybee_c_abs_layers <= 0:
                self.honeybee_c_abs_layers = 2
        elif self.model_type == 'pargo':
            self.output_dir = './checkpoints_pargo'
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
        elif self.model_type in ['multiview_fusion', 'multiview_fusion_honeybee', 'multiview_fusion_pargo']:
            self.output_dir = './checkpoints_multiview_fusion'
            self.enable_multiview_fusion = True
            if self.multiview_fusion_views != 4:
                self.multiview_fusion_views = 4  
            if self.multiview_fusion_size <= 0:
                self.multiview_fusion_size = 448
            if self.multiview_spatial_dim <= 0:
                self.multiview_spatial_dim = 49   
            if self.multiview_channel_dim <= 0:
                self.multiview_channel_dim = 2304  
            if self.multiview_hidden_dim <= 0:
                self.multiview_hidden_dim = 512
        elif self.model_type == 'ablation':
            modality_str = '_'.join(sorted(self.ablation_2d_modalities))
            lidar_str = '_lidar' if self.ablation_enable_lidar else ''
            self.output_dir = f'./checkpoints_ablation_{modality_str}{lidar_str}'
            self.enable_multiview_fusion = True
            if self.multiview_fusion_views != 4:
                self.multiview_fusion_views = 4
            if self.multiview_fusion_size <= 0:
                self.multiview_fusion_size = 448
            if self.multiview_spatial_dim <= 0:
                self.multiview_spatial_dim = 49
            if self.multiview_channel_dim <= 0:
                self.multiview_channel_dim = 2304
            if self.multiview_hidden_dim <= 0:
                self.multiview_hidden_dim = 512


@dataclass
class FusionDataConfig:
    data_dir: str = "./data_rebuilt"  
    enabled_modalities: List[str] = field(default_factory=lambda: ['rgb', 'depth', 'event', 'pointcloud'])
    
    image_size: int = 224
    num_rgb_views: int = 4    
    num_depth_views: int = 4  
    num_event_views: int = 4  
    pointcloud_mode: str = 'scene'  
    object_pointcloud_size: int = 1024  
    scene_pointcloud_size: int = 16384  
    max_objects_per_sample: int = 20    
    
    max_length: int = 128
    
    ablation_enable_lidar: bool = False  
    
    fusion_token_layout: str = 'single'  
    fusion_num_tokens: int = 1

    @classmethod
    def from_leo_config(cls, leo_config: LEOConfig) -> 'FusionDataConfig':
        return cls(
            data_dir=leo_config.data_dir,
            enabled_modalities=leo_config.enabled_modalities,
            image_size=leo_config.image_size,
            num_rgb_views=leo_config.num_rgb_views,
            num_depth_views=leo_config.num_depth_views,
            num_event_views=leo_config.num_event_views,
            pointcloud_mode=leo_config.pointcloud_mode,
            scene_pointcloud_size=leo_config.scene_pointcloud_size,
            object_pointcloud_size=leo_config.point_cloud_size,  
            max_length=leo_config.max_txt_len,
            fusion_token_layout=getattr(leo_config, 'fusion_token_layout', 'single'),
            fusion_num_tokens=getattr(leo_config, 'fusion_num_tokens', 1)
        )

def get_baseline_config() -> LEOConfig:
    return LEOConfig(
        model_type='baseline',
        use_sq_hub=False,
        use_ppx=False
    )

def get_cmnext_config() -> LEOConfig:
    return LEOConfig(
        model_type='cmnext',
        use_sq_hub=True,
        use_ppx=True,
        aux_modality_weight=0.3,
        output_dir='./checkpoints_cmnext'
    )

def get_honeybee_config() -> LEOConfig:
    return LEOConfig(
        model_type='honeybee',
        honeybee_num_tokens=64,
        honeybee_c_abs_layers=2,
        honeybee_pooling_size=2,
        output_dir='./checkpoints_honeybee'
    )

def get_pargo_config() -> LEOConfig:
    return LEOConfig(
        model_type='pargo',
        pargo_num_tokens=64,
        pargo_partial_layers=2,
        pargo_global_layers=2,
        pargo_fusion_dim=512,
        pargo_temperature=0.1,
        output_dir='./checkpoints_pargo'
    )

def get_multiview_fusion_config() -> LEOConfig:
    return LEOConfig(
        model_type='multiview_fusion',
        multiview_fusion_views=4,
        multiview_fusion_size=448,
        multiview_spatial_dim=49,
        multiview_channel_dim=2304,
        multiview_hidden_dim=512,
        multiview_spatial_heads=8,
        multiview_channel_heads=8,
        multiview_dropout=0.1,
        output_dir='./checkpoints_multiview_fusion'
    )

def get_multiview_fusion_honeybee_config() -> LEOConfig:
    return LEOConfig(
        model_type='multiview_fusion_honeybee',
        multiview_fusion_views=4,
        multiview_fusion_size=448,
        multiview_spatial_dim=49,
        multiview_channel_dim=2304,
        multiview_hidden_dim=512,
        multiview_spatial_heads=8,
        multiview_channel_heads=8,
        multiview_dropout=0.1,
        honeybee_num_tokens=64,
        honeybee_c_abs_layers=2,
        honeybee_pooling_size=2,
        output_dir='./checkpoints_multiview_fusion_honeybee'
    )

def get_multiview_fusion_pargo_config() -> LEOConfig:
    return LEOConfig(
        model_type='multiview_fusion_pargo',
        multiview_fusion_views=4,
        multiview_fusion_size=448,
        multiview_spatial_dim=49,
        multiview_channel_dim=2304,
        multiview_hidden_dim=512,
        multiview_spatial_heads=8,
        multiview_channel_heads=8,
        multiview_dropout=0.1,
        pargo_num_tokens=64,
        pargo_partial_layers=2,
        pargo_global_layers=2,
        pargo_fusion_dim=512,
        pargo_temperature=0.1,
        output_dir='./checkpoints_multiview_fusion_pargo'
    )

def get_ablation_config(modalities_2d: List[str], enable_lidar: bool = True) -> LEOConfig:

    valid_modalities = {'rgb', 'depth', 'event'}
    if not all(m in valid_modalities for m in modalities_2d):
        raise ValueError(f"Invalid modalities: {modalities_2d}. Must be subset of {valid_modalities}")
    
    if len(modalities_2d) == 0:
        raise ValueError("At least one 2D modality must be specified")
    
    enabled_modalities = list(modalities_2d)
    if enable_lidar:
        enabled_modalities.append('pointcloud')
    
    return LEOConfig(
        model_type='ablation',
        ablation_2d_modalities=modalities_2d,
        ablation_enable_lidar=enable_lidar,
        ablation_query_priority=['rgb', 'depth', 'event'],  
        enabled_modalities=enabled_modalities,
        multiview_fusion_views=4,
        multiview_fusion_size=448,
        multiview_spatial_dim=49,
        multiview_channel_dim=2304,
        multiview_hidden_dim=512,
        multiview_spatial_heads=8,
        multiview_channel_heads=8,
        multiview_dropout=0.1,
        pointcloud_mode='scene',
        scene_pointcloud_size=16384,
        point_cloud_size=8192
    )