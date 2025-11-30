#!/bin/bash
# 统一的多模态Vicuna训练脚本
# 支持五种模式: baseline, cmnext, honeybee, pargo, multiview_fusion
# 修复了所有导入问题和训练卡住问题

set -e


# 检查参数
if [ $# -eq 0 ]; then
    echo "error: please specify the model type"
    echo "usage: bash trainer/train_multimodal_unified.sh [baseline|cmnext|honeybee|pargo|multiview_fusion|multiview_fusion_honeybee|multiview_fusion_pargo|ablation] [options]"
    echo ""
    echo "standard models example:"
    echo "  bash trainer/train_multimodal_unified.sh baseline"
    echo "  bash trainer/train_multimodal_unified.sh honeybee --batch_size 2"
    echo "  bash trainer/train_multimodal_unified.sh pargo --num_epochs 3"
    echo "  bash trainer/train_multimodal_unified.sh multiview_fusion --batch_size 4 --fusion_variant qattn"
    echo "  bash trainer/train_multimodal_unified.sh multiview_fusion_honeybee --batch_size 4"
    echo "  bash trainer/train_multimodal_unified.sh multiview_fusion_pargo --batch_size 4"
    echo ""
    echo "ablation experiment example:"
    echo "  bash trainer/train_multimodal_unified.sh ablation --modalities rgb"
    echo "  bash trainer/train_multimodal_unified.sh ablation --modalities rgb depth"
    echo "  bash trainer/train_multimodal_unified.sh ablation --modalities rgb depth event --enable_lidar"
    echo "  bash trainer/train_multimodal_unified.sh ablation --modalities depth event --batch_size 2"
    exit 1
fi

MODEL_TYPE=$1
shift  

if [ "$MODEL_TYPE" != "baseline" ] && [ "$MODEL_TYPE" != "cmnext" ] && [ "$MODEL_TYPE" != "honeybee" ] && [ "$MODEL_TYPE" != "pargo" ] && [ "$MODEL_TYPE" != "multiview_fusion" ] && [ "$MODEL_TYPE" != "ablation" ] && [ "$MODEL_TYPE" != "multiview_fusion_honeybee" ] && [ "$MODEL_TYPE" != "multiview_fusion_pargo" ]; then
    echo "error: model type must be 'baseline', 'cmnext', 'honeybee', 'pargo', 'multiview_fusion', 'ablation', 'multiview_fusion_honeybee' or 'multiview_fusion_pargo'"
    exit 1
fi

DATA_DIR="./data_rebuilt"
BATCH_SIZE=4                
LEARNING_RATE=2e-5
NUM_EPOCHS=5
WARMUP_STEPS=100
SAVE_STEPS=1000
EVAL_STEPS=250
USE_LORA=true
LORA_RANK=16
NUM_WORKERS=2               

FUSION_ENABLED=false
FUSION_VARIANT=""
FUSION_TOKEN_LAYOUT=""
if [[ "$*" == *"--enable_multiview_fusion"* ]]; then
    FUSION_ENABLED=true
fi

if [ "$MODEL_TYPE" == "multiview_fusion" ]; then
    args_array=("$@")
    for ((i=0; i<${#args_array[@]}; i++)); do
        if [ "${args_array[$i]}" == "--fusion_variant" ]; then
            next_index=$((i+1))
            if [ $next_index -lt ${#args_array[@]} ]; then
                FUSION_VARIANT="${args_array[$next_index]}"
            fi
            break
        fi
    done
    for ((i=0; i<${#args_array[@]}; i++)); do
        if [ "${args_array[$i]}" == "--fusion_token_layout" ]; then
            next_index=$((i+1))
            if [ $next_index -lt ${#args_array[@]} ]; then
                FUSION_TOKEN_LAYOUT="${args_array[$next_index]}"
            fi
            break
        fi
    done
fi

ABLATION_MODALITIES=""
ABLATION_ENABLE_LIDAR="false"

if [ "$MODEL_TYPE" == "ablation" ]; then
    echo "detecting ablation experiment mode, parsing parameters..."
    
    i=1
    for arg in "$@"; do
        if [ "$arg" == "--modalities" ]; then
            shift_count=$((i + 1))
            modality_args=""
            j=$shift_count
            for remaining_arg in "${@:$j}"; do
                if [[ "$remaining_arg" == --* ]]; then
                    break
                fi
                if [ "$remaining_arg" == "rgb" ] || [ "$remaining_arg" == "depth" ] || [ "$remaining_arg" == "event" ]; then
                    if [ -z "$modality_args" ]; then
                        modality_args="$remaining_arg"
                    else
                        modality_args="$modality_args $remaining_arg"
                    fi
                fi
                j=$((j + 1))
            done
            ABLATION_MODALITIES="$modality_args"
        elif [ "$arg" == "--enable_lidar" ]; then
            ABLATION_ENABLE_LIDAR="true"
        fi
        i=$((i + 1))
    done
    
    if [ -z "$ABLATION_MODALITIES" ]; then
        echo "error: ablation experiment mode must specify --modalities parameter"
        echo "example: --modalities rgb depth"
        exit 1
    fi
    
    modality_str=$(echo "$ABLATION_MODALITIES" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    lidar_suffix=""
    if [ "$ABLATION_ENABLE_LIDAR" == "true" ]; then
        lidar_suffix="_lidar"
    fi
    OUTPUT_DIR="./checkpoints_ablation_${modality_str}${lidar_suffix}"
    
    echo "ablation experiment configuration:"
    echo "  2D modalities: $ABLATION_MODALITIES"
    echo "  LiDAR enabled: $ABLATION_ENABLE_LIDAR"
    echo "  output directory: $OUTPUT_DIR"
else
    if [ "$FUSION_ENABLED" == "true" ]; then
        OUTPUT_DIR="./checkpoints_${MODEL_TYPE}_fusion"
    else
        OUTPUT_DIR="./checkpoints_${MODEL_TYPE}"
    fi
    if [ "$MODEL_TYPE" == "multiview_fusion" ] && [ -n "$FUSION_VARIANT" ]; then
        OUTPUT_DIR="${OUTPUT_DIR}_${FUSION_VARIANT}"
    fi
    if [ "$MODEL_TYPE" == "multiview_fusion" ] && [ -n "$FUSION_TOKEN_LAYOUT" ]; then
        OUTPUT_DIR="${OUTPUT_DIR}_${FUSION_TOKEN_LAYOUT}"
    fi
fi

echo "basic training configuration:"
echo "  model type: $MODEL_TYPE"
echo "  data directory: $DATA_DIR"
echo "  output directory: $OUTPUT_DIR"
if [ "$MODEL_TYPE" == "multiview_fusion" ] && [ -n "$FUSION_VARIANT" ]; then
  echo "  fusion variant: $FUSION_VARIANT"
fi
if [ "$MODEL_TYPE" == "multiview_fusion" ] && [ -n "$FUSION_TOKEN_LAYOUT" ]; then
  echo "  token layout: $FUSION_TOKEN_LAYOUT"
fi
echo "  batch size: $BATCH_SIZE"
echo "  learning rate: $LEARNING_RATE"
echo "  training epochs: $NUM_EPOCHS"
echo "  worker number: $NUM_WORKERS (avoid deadlock)"
echo "  LoRA: $USE_LORA (rank: $LORA_RANK)"

if [ ! -d "$DATA_DIR" ]; then
    echo "error: data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -f "$DATA_DIR/split_index.json" ]; then
    echo "error: split index file does not exist: $DATA_DIR/split_index.json"
    exit 1
fi

if [ ! -d "$DATA_DIR/data_packed" ]; then
    echo "error: packed data directory does not exist: $DATA_DIR/data_packed"
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    echo "error: no GPU detected"
    exit 1
elif [ "$AVAILABLE_GPUS" -lt 4 ]; then
    echo "warning: detected $AVAILABLE_GPUS GPUs, less than 4"
    NUM_GPUS=$AVAILABLE_GPUS
else
    echo "detected $AVAILABLE_GPUS GPUs, using 4 for training"
    NUM_GPUS=4
fi

echo ""
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits | head -4

mkdir -p "$OUTPUT_DIR"

export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo ""
echo "environment optimization completed"

case "$MODEL_TYPE" in
    "baseline")
        TRAIN_SCRIPT="trainer/run_baseline_training.py"
        echo "starting Baseline model training..."
        ;;
    "cmnext")
        TRAIN_SCRIPT="trainer/run_cmnext_training.py"
        echo "starting CMNeXt model training..."
        ;;
    "honeybee")
        TRAIN_SCRIPT="trainer/run_honeybee_training.py"
        echo "starting Honeybee model training..."
        ;;
    "pargo")
        TRAIN_SCRIPT="trainer/run_pargo_training.py"
        echo "starting ParGo model training..."
        ;;
    "multiview_fusion")
        TRAIN_SCRIPT="trainer/run_multiview_fusion_training.py"
        echo "starting MultiView-Fusion model training..."
        ;;
    "ablation")
        TRAIN_SCRIPT="trainer/run_ablation_training.py"
        echo "starting ablation experiment training..."
        echo "  modalities: $ABLATION_MODALITIES"
        echo "  LiDAR: $ABLATION_ENABLE_LIDAR"
        ;;
    "multiview_fusion_honeybee")
        TRAIN_SCRIPT="trainer/run_multiview_fusion_honeybee_training.py"
        echo "starting MultiView-Fusion-Honeybee model training..."
        echo "  using Honeybee C-Abstractor projection layers"
        ;;
    "multiview_fusion_pargo")
        TRAIN_SCRIPT="trainer/run_multiview_fusion_pargo_training.py"
        echo "starting MultiView-Fusion-Pargo model training..."
        echo "  using ParGo Partial-Global projection layers"
        ;;
esac

# 检查训练脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "error: training script does not exist: $TRAIN_SCRIPT"
    echo "please ensure all training scripts are placed in the trainer directory"
    exit 1
fi

TRAIN_ARGS="--data_dir $DATA_DIR"
TRAIN_ARGS="$TRAIN_ARGS --output_dir $OUTPUT_DIR"
TRAIN_ARGS="$TRAIN_ARGS --batch_size $BATCH_SIZE"
TRAIN_ARGS="$TRAIN_ARGS --learning_rate $LEARNING_RATE"
TRAIN_ARGS="$TRAIN_ARGS --num_epochs $NUM_EPOCHS"
TRAIN_ARGS="$TRAIN_ARGS --warmup_steps $WARMUP_STEPS"
TRAIN_ARGS="$TRAIN_ARGS --save_steps $SAVE_STEPS"
TRAIN_ARGS="$TRAIN_ARGS --eval_steps $EVAL_STEPS"
TRAIN_ARGS="$TRAIN_ARGS --use_lora --lora_rank $LORA_RANK"

if [ "$MODEL_TYPE" == "ablation" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --modalities $ABLATION_MODALITIES"
    if [ "$ABLATION_ENABLE_LIDAR" == "true" ]; then
        TRAIN_ARGS="$TRAIN_ARGS --enable_lidar"
    fi
    
    filtered_args=""
    skip_modalities=false
    for arg in "$@"; do
        if [ "$skip_modalities" == "true" ]; then
            if [ "$arg" == "rgb" ] || [ "$arg" == "depth" ] || [ "$arg" == "event" ]; then
                continue
            else
                skip_modalities=false
            fi
        fi
        
        if [ "$arg" == "--modalities" ]; then
            skip_modalities=true
            continue
        elif [ "$arg" == "--enable_lidar" ]; then
            continue
        elif [ "$skip_modalities" == "false" ]; then
            filtered_args="$filtered_args $arg"
        fi
    done
    TRAIN_ARGS="$TRAIN_ARGS $filtered_args"
else
TRAIN_ARGS="$TRAIN_ARGS $@"

if [ "$MODEL_TYPE" == "multiview_fusion" ]; then
    for arg in "$@"; do
        if [ "$arg" == "--fusion_variant" ]; then
            echo "using custom fusion_variant (see final parameters)"
            break
        fi
    done
fi
fi

if [ "$FUSION_ENABLED" == "true" ]; then
    echo "four-view fusion: enabled"
else
    echo "four-view fusion: disabled (add --enable_multiview_fusion to enable)"
fi

echo ""
echo "final training parameters: $TRAIN_ARGS"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/training.log"

echo ""
echo "training log will be saved to: $TRAIN_LOG"

# 训练前最后确认
echo ""
echo "training pre-check..."
echo "  data directory: $(ls -la $DATA_DIR | head -3)"
echo "  output directory: $OUTPUT_DIR"
echo "  training script: $TRAIN_SCRIPT"
echo "  GPU number: $NUM_GPUS"
echo ""

read -p "confirm to start training? (y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "training cancelled"
    exit 0
fi

echo ""
echo "starting distributed training..."
echo "start time: $(date)"
echo ""

accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --main_process_port=29500 \
    $TRAIN_SCRIPT \
    $TRAIN_ARGS 2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "end time: $(date)"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "training completed successfully!"
    echo ""
    echo "training results:"
    echo "  output directory: $OUTPUT_DIR"
    echo "  checkpoint number: $(ls -1 $OUTPUT_DIR 2>/dev/null | grep -E "(epoch-|final)" | wc -l)"
    echo "  training log: $TRAIN_LOG"
    echo ""
    echo "available checkpoints:"
    ls -la "$OUTPUT_DIR"/ 2>/dev/null | grep -E "(epoch-|final)" || echo "  (no checkpoint files)"
    echo "$MODEL_TYPE model training completed!"
    
else
    echo ""
    echo "training failed (exit code: $TRAIN_EXIT_CODE)"
    echo ""
    echo ""
    
    if [ -f "$TRAIN_LOG" ]; then
        echo "latest log output:"
        echo "=================="
        tail -20 "$TRAIN_LOG"
        echo "=================="
    fi
    


    exit 1
fi

