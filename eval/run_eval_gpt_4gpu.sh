#!/bin/bash

set -e


print_usage() {
    echo "  checkpoint_path    "
    echo "  --model_type       multiview_fusion"
    echo "                     baseline, honeybee, pargo, cmnext,"
    echo "                             multiview_fusion, multiview_fusion_honeybee, multiview_fusion_pargo,"
    echo "                             ablation"
    echo "  --modalities       rgb, depth, event, pointcloud"
    echo "  --enable_lidar     LiDAR"
    echo "  --attention_type   spatial, channel, both "
    echo "  --fusion_variant   gap, qattn "
    echo "                     multiview_fusion: gap, qattn, qattn_spectral, qattn_depthgate"
    echo "  --enable_multiview_fusion 2x2"
    echo "  --data_dir         path: ./data_rebuilt"
    echo "  --fusion_token_layout  single|triple "
    echo "  --eval_split       "
    echo "  --batch_size       "
    echo "  --max_samples      "
    echo "  --output_dir       "
    echo "  --openai_api_key   "
    echo "  --disable_gpt_eval "
    echo ""
    echo "  OPENAI_API_KEY     OpenAI API"
}


if [ $# -eq 0 ]; then
    echo "missing key parameters"
    print_usage
    exit 1
fi



if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi


CHECKPOINT_PATH=$1
shift


MODEL_TYPE="multiview_fusion"
MODALITIES=()
ENABLE_LIDAR=false
ATTENTION_TYPE="both"
ENABLE_MULTIVIEW_FUSION=false
FUSION_VARIANT=""
FUSION_TOKEN_LAYOUT=""
DATA_DIR="./data_rebuilt"
EVAL_SPLIT="val"
BATCH_SIZE=4
MAX_SAMPLES=""
OUTPUT_DIR=""
OPENAI_API_KEY=""
DISABLE_GPT_EVAL=false


while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            if [[ "$2" =~ ^(baseline|honeybee|pargo|cmnext|multiview_fusion|multiview_fusion_honeybee|multiview_fusion_pargo|ablation)$ ]]; then
                MODEL_TYPE="$2"
            else
                echo " '$2'"
                echo "    baseline, honeybee, pargo, cmnext, multiview_fusion, multiview_fusion_honeybee, multiview_fusion_pargo, ablation"
                exit 1
            fi
            shift 2
            ;;
        --modalities)
            MODALITIES=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do

                if [[ "$1" =~ ^(rgb|depth|event|pointcloud)$ ]]; then
                    MODALITIES+=("$1")
                else
                    echo " '$1'"
                    echo "   rgb, depth, event, pointcloud"
                    exit 1
                fi
                shift
            done

            if [ ${#MODALITIES[@]} -eq 0 ]; then
                echo "at least one modal"
                exit 1
            fi
            ;;
        --enable_lidar)
            ENABLE_LIDAR=true
            shift
            ;;
        --attention_type)
            if [[ "$2" =~ ^(spatial|channel|both)$ ]]; then
                ATTENTION_TYPE="$2"
            else
                echo "'$2'"
                echo "   spatial, channel, both"
                exit 1
            fi
            shift 2
            ;;
        --enable_multiview_fusion)
            ENABLE_MULTIVIEW_FUSION=true
            shift
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --fusion_variant)
            FUSION_VARIANT="$2"
            shift 2
            ;;
        --fusion_token_layout)
            FUSION_TOKEN_LAYOUT="$2"
            shift 2
            ;;
        --eval_split)
            if [[ "$2" =~ ^(val|test)$ ]]; then
                EVAL_SPLIT="$2"
            else
                echo " '$2'"
                echo "    val, test"
                exit 1
            fi
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --openai_api_key)
            OPENAI_API_KEY="$2"
            shift 2
            ;;
        --disable_gpt_eval)
            DISABLE_GPT_EVAL=true
            shift
            ;;
        *)
            echo "'$1'"
            print_usage
            exit 1
            ;;
    esac
done


if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "not exist: $CHECKPOINT_PATH"
    exit 1
fi


if [ ! -d "$DATA_DIR" ]; then
    echo "no data: $DATA_DIR"
    exit 1
fi


if [ ${#MODALITIES[@]} -gt 0 ] && [ "$MODEL_TYPE" != "ablation" ]; then

    MODALITIES=()
fi


if [ "$MODEL_TYPE" = "ablation" ]; then

    true  
elif [ "$MODEL_TYPE" = "multiview_fusion" ]; then

    ENABLE_LIDAR=true
else

    ENABLE_LIDAR=true
fi


if [ -z "$OUTPUT_DIR" ]; then

    OUTPUT_DIR="gpt_eval_results_${MODEL_TYPE}"
    

    if [ "$MODEL_TYPE" = "ablation" ] && [ ${#MODALITIES[@]} -gt 0 ]; then
        MODALITIES_STR=$(IFS=_; echo "${MODALITIES[*]}")
        OUTPUT_DIR="${OUTPUT_DIR}_${MODALITIES_STR}"
    fi
    

    if [ "$MODEL_TYPE" = "ablation" ]; then
        if [ "$ENABLE_LIDAR" = true ]; then
            OUTPUT_DIR="${OUTPUT_DIR}_lidar"
        else
            OUTPUT_DIR="${OUTPUT_DIR}_nolidar"
        fi
 
        if [ "$ATTENTION_TYPE" != "both" ]; then
            OUTPUT_DIR="${OUTPUT_DIR}_${ATTENTION_TYPE}"
        fi
  
        if [ -n "$FUSION_VARIANT" ] && [ "$FUSION_VARIANT" != "qattn" ]; then
            OUTPUT_DIR="${OUTPUT_DIR}_${FUSION_VARIANT}"
        fi
    fi
    
 
    if [ "$MODEL_TYPE" != "ablation" ]; then
        if [ "$ENABLE_MULTIVIEW_FUSION" = true ]; then
            OUTPUT_DIR="${OUTPUT_DIR}_fusion"
        else
            OUTPUT_DIR="${OUTPUT_DIR}_nofusion"
        fi
    fi

    if [ "$MODEL_TYPE" = "multiview_fusion" ]; then
        if [ -n "$FUSION_VARIANT" ]; then
            OUTPUT_DIR="${OUTPUT_DIR}_${FUSION_VARIANT}"
        fi
        if [ -n "$FUSION_TOKEN_LAYOUT" ]; then
            OUTPUT_DIR="${OUTPUT_DIR}_${FUSION_TOKEN_LAYOUT}"
        fi
    fi
    

    OUTPUT_DIR="${OUTPUT_DIR}_${EVAL_SPLIT}"
fi


if [ -z "$OPENAI_API_KEY" ]; then
    OPENAI_API_KEY="${OPENAI_API_KEY:-}"
fi


if [ "$DISABLE_GPT_EVAL" = false ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "no api key"
    DISABLE_GPT_EVAL=true
fi



if [ ${#MODALITIES[@]} -gt 0 ]; then
    echo "   activ modalities: ${MODALITIES[*]}"
fi
echo "   LiDAR: $ENABLE_LIDAR"
if [ "$MODEL_TYPE" = "ablation" ]; then
    echo "   attn: $ATTENTION_TYPE"
fi
echo "   fusion: $ENABLE_MULTIVIEW_FUSION"
echo "   fusion_variant: ${FUSION_VARIANT:-default}"
echo "   token_layout: ${FUSION_TOKEN_LAYOUT:-default}"
echo "   split: $EVAL_SPLIT"
echo "   batch_size: $BATCH_SIZE"
echo "   max_samples: $([ -n "$MAX_SAMPLES" ] && echo "$MAX_SAMPLES" || echo "no limit")"
echo "   output: $OUTPUT_DIR"
echo "   GPT: $([ "$DISABLE_GPT_EVAL" = true ] && echo "false" || echo "true")"
if [ "$DISABLE_GPT_EVAL" = false ]; then
    echo "   APIkey: $([ -n "$OPENAI_API_KEY" ] && echo "true" || echo "false")"
fi


if ! command -v nvidia-smi &> /dev/null; then
    echo "no nvidia-smi"
    exit 1
fi


GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ $GPU_COUNT -eq 0 ]; then
    echo "no GPU"
    exit 1
fi




if [ $GPU_COUNT -gt 4 ]; then
    GPU_COUNT=4
fi




EVAL_ARGS=(
    "--checkpoint_path" "$CHECKPOINT_PATH"
    "--data_dir" "$DATA_DIR"
    "--output_dir" "$OUTPUT_DIR"
    "--model_type" "$MODEL_TYPE"
    "--eval_split" "$EVAL_SPLIT"
    "--batch_size" "$BATCH_SIZE"
    "--num_workers" "0"
)


if [ -n "$MAX_SAMPLES" ]; then
    EVAL_ARGS+=("--max_samples" "$MAX_SAMPLES")
fi


if [ ${#MODALITIES[@]} -gt 0 ]; then
    EVAL_ARGS+=("--modalities" "${MODALITIES[@]}")
fi

if [ "$ENABLE_LIDAR" = true ]; then
    EVAL_ARGS+=("--enable_lidar")
fi

if [ "$MODEL_TYPE" = "ablation" ] && [ "$ATTENTION_TYPE" != "both" ]; then
    EVAL_ARGS+=("--attention_type" "$ATTENTION_TYPE")
fi

if [ "$ENABLE_MULTIVIEW_FUSION" = true ]; then
    EVAL_ARGS+=("--enable_multiview_fusion")
fi

if [ -n "$FUSION_VARIANT" ]; then
    EVAL_ARGS+=("--fusion_variant" "$FUSION_VARIANT")
fi
if [ -n "$FUSION_TOKEN_LAYOUT" ]; then
    EVAL_ARGS+=("--fusion_token_layout" "$FUSION_TOKEN_LAYOUT")
fi

if [ "$DISABLE_GPT_EVAL" = true ]; then
    EVAL_ARGS+=("--disable_gpt_eval")
fi

if [ -n "$OPENAI_API_KEY" ]; then
    EVAL_ARGS+=("--openai_api_key" "$OPENAI_API_KEY")
fi


export PYTHONPATH="${PYTHONPATH}:$(pwd)/trainer:$(pwd)/eval"
export TOKENIZERS_PARALLELISM=false


if [ -n "$OPENAI_API_KEY" ]; then
    export OPENAI_API_KEY="$OPENAI_API_KEY"
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_with_gpt.py"


if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "no script: $EVAL_SCRIPT"
    exit 1
fi


echo "   torchrun --nproc_per_node=$GPU_COUNT python $EVAL_SCRIPT ${EVAL_ARGS[*]}"
echo ""


LOG_FILE="${OUTPUT_DIR}/evaluation.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "log saved to: $LOG_FILE"
echo ""


if [ $GPU_COUNT -gt 1 ]; then

    torchrun \
        --nproc_per_node=$GPU_COUNT \
        --master_port=29501 \
        "$EVAL_SCRIPT" \
        "${EVAL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
else

    CUDA_VISIBLE_DEVICES=0 python "$EVAL_SCRIPT" "${EVAL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
fi


if [ $? -eq 0 ]; then
   
    echo "  finished"
else
    echo ""
    echo "error"
    exit 1
fi