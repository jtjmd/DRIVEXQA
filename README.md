# DRIVEXQA: Cross-modal Visual Question Answering for Adverse Driving Scene Understanding

[![Dataset](https://img.shields.io/badge/Dataset-Download-blue)](https://1drv.ms/u/c/61920c3ee2305751/IQAtTCkMDPZKSaM9JbMK9NoUAZ9gBhT2m1lRMd-av1eYFFE?e=bKCYbK)


A comprehensive multi-modal dataset and framework for visual question answering in adverse driving conditions. DRIVEXQA addresses the critical challenge of maintaining robust scene understanding when autonomous vehicles face challenging weather conditions and sensor degradation scenarios.

**Key Statistics:**
- 📊 **7,885 frames** with **102,505 QA pairs**
- 🌦️ **5 weather conditions**: Sunny, Cloudy, Rainy, Foggy, Night
- ⚠️ **5 sensor failure types**: Motion Blur, Overexposure, Underexposure, LiDAR Jitter, Event Camera Low-resolution
- 📷 **4 sensor modalities**: RGB, Depth, Event Camera, LiDAR
- 🔄 **4-view coverage**: Front, Back, Left, Right
- 📝 **3 question hierarchies**: Global Scene, Allocentric, Ego-Vehicle Centric

---

## 🚗 DRIVEXQA Highlights

- **🌧️ Adverse Condition Focus**: Systematic coverage of 5 weather conditions and 5 sensor failure types (35.5% corner cases)
- **📷 Multi-Modal Sensor Fusion**: RGB + Depth + Event Camera + LiDAR for complementary perception
- **👁️ Multi-View Understanding**: complete scene coverage with 4-view camera system (Front, Back, Left, Right)
- **🎯 Safety-Critical VQA**: Three-level hierarchical questions (Global Scene, Allocentric, Ego-Vehicle Centric)
- **⚡ MVX-LLM Framework**: Novel Dual Cross-Attention (DCA) mechanism for robust multi-modal fusion
- **📊 GPT-4o Evaluation**: Semantic correctness scoring with strong human correlation (Spearman ρ > 0.83)

---

## 📑 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Supported Modalities](#supported-modalities-for-adverse-driving-scenes)
- [Quick Start](#quick-start)
  - [Installation](#prerequisites)
  - [Data Preparation](#data-preparation)
  - [Training](#basic-training)
  - [Evaluation](#quick-start-evaluation)
- [Model Architectures](#model-architectures)
- [Training Configuration](#training-configuration)
- [Evaluation Framework](#evaluation)
- [Citation](#citation)


---

## Overview

DRIVEXQA addresses the critical gap in multi-modal large language models (MLLMs) for autonomous driving: existing MLLMs cannot accept multiple complementary sensor modalities as input, making them inadequate for safety-critical scenarios requiring robust performance under sensor failures and adverse weather conditions.

### The Challenge

Unlike general visual understanding tasks, autonomous driving demands:
- **Comprehensive spatial understanding** across multiple viewpoints
- **Environmental robustness** under adverse conditions (fog, rain, night)
- **Sensor reliability assessment** during hardware failures
- **Cross-modal compensation** when individual sensors degrade

### Our Solution: MVX-LLM

We propose **MVX-LLM** (Multi-View Cross-modal Large Language Model), a token-efficient architecture featuring:

1. **Dual Cross-Attention (DCA) Mechanism**: Fuses RGB, Depth, and Event camera data through complementary spatial and channel attention pathways
2. **Independent LiDAR Processing**: Point cloud features processed via PointNet++ for geometric understanding
3. **Query Attention Aggregation**: Learnable query-based token compression for efficient LLM integration
4. **LoRA Fine-tuning**: Parameter-efficient adaptation with rank-16 for Vicuna-7B-v1.5

### Supported Model Architectures

- **MVX-LLM (Ours)**: Dual Cross-Attention with Query Attention aggregation
- **Token Fusion Baselines**: 
  - Prepend: Simple token concatenation
  - Self-Query: Abnormal condition modeling without cross-modal fusion
  - GAP: Global Average Pooling
  - QAttn(Spectral): Query Attention with frequency domain processing
  - QAttn(DepthGate): Depth-guided confidence estimation
- **Projector Baselines**:
  - Honeybee: Locality-enhanced projectors
  - ParGo: Partial-Global projectors

---

## Directory Structure

```
.
├── trainer/                               # Training scripts and modules
│   ├── run_baseline_training.py           # Prepend model training
│   ├── run_cmnext_training.py             # Self-Query model training
│   ├── run_honeybee_training.py           # Honeybee model training
│   ├── run_pargo_training.py              # ParGo model training
│   ├── run_multiview_fusion_training.py   # MVX-LLM training
│   ├── run_multiview_fusion_honeybee_training.py
│   ├── run_multiview_fusion_pargo_training.py
│   ├── run_ablation_training.py           # Ablation experiments
│   ├── train_multimodal_unified.sh        # Unified training script
│   ├── configs.py                         # Configuration definitions
│   ├── leo_authentic_trainer.py           # Core trainer implementation
│   ├── fusion_data_loader.py              # Multi-modal data loader
│   ├── vision_encoder.py                  # Vision encoders
│   └── pointnet_util.py                   # Point cloud processing utilities
│
└── eval/                                  # Evaluation scripts and modules
    ├── evaluate_with_gpt.py               # Main evaluation script
    ├── evaluate_ablation.py               # Ablation study evaluation
    ├── enhanced_evaluator.py              # Comprehensive evaluator
    ├── evaluation_metrics.py              # Traditional NLG metrics
    ├── gpt_evaluator.py                   # GPT-4o semantic evaluation
    ├── metadata_extractor.py              # Scene/question metadata
    └── run_eval_gpt_4gpu.sh               # Unified evaluation script

```

---

## Supported Modalities for Adverse Driving Scenes

DRIVEXQA leverages complementary sensor modalities to ensure robust scene understanding even when individual sensors fail or degrade:

- **RGB**: Multi-view RGB cameras (4 views: Front, Back, Left, Right)
- **Depth**: Multi-view depth maps (4 views)
- **Event**: Multi-view event camera data (4 views)
- **LiDAR Point Cloud**: Scene-level 3D spatial understanding (16,384 points)

### Multi-View Fusion Layout

For 2D modalities (RGB/Depth/Event), views are arranged in a 2×2 grid:

```
┌─────────┬─────────┐
│  FRONT  │  BACK   │
├─────────┼─────────┤
│  LEFT   │  RIGHT  │
└─────────┴─────────┘
```

Each modality is independently concatenated into 448×448 fusion images.

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch torchvision transformers accelerate
pip install peft  # For LoRA training
```

### Data Preparation

1. Download the DRIVEXQA dataset and place it in `./data_rebuilt/`
2. Ensure the following files exist:
   - `data_rebuilt/split_index.json` - 80%/10%/10% train/val/test split indices
   - `data_rebuilt/data_packed/` - Packed multi-modal sensor data

**Dataset Composition:**

| Category | Distribution | Details |
|----------|--------------|---------|
| **Weather Conditions** | Balanced 20% each | Rainy, Night, Sunny, Cloudy, Foggy |
| **Sensor Failures** | 35.5% of total | Motion Blur (42.7%), Underexposure (14.2%), Overexposure (14.3%), LiDAR Jitter (14.2%), Event Low-res (14.6%) |
| **Question Types** | 13 per scene | Global Scene (2), Allocentric (8), Ego-Centric (3) |
| **QA Pairs** | 102,505 total | Avg question: 11.4 words, Avg answer: 12.9 words |

**Hierarchical Question Structure:**

1. **Global Scene Level** (2 questions/scene): Weather conditions, traffic density, overall environmental assessment
2. **Allocentric Level** (8 questions/scene): Spatial relationships, distance measurements, object categorization, traffic signs
3. **Ego-Vehicle Centric Level** (3 questions/scene): Lane positioning, sensor health, surrounding vehicle behavior

### Basic Training

Train the MVX-LLM model with Dual Cross-Attention:

```bash
# Train MVX-LLM (our proposed model)
bash trainer/train_multimodal_unified.sh multiview_fusion

# With custom aggregation variant
bash trainer/train_multimodal_unified.sh multiview_fusion \
  --fusion_variant qattn \
  --batch_size 4

# Train baseline models for comparison
bash trainer/train_multimodal_unified.sh baseline
bash trainer/train_multimodal_unified.sh honeybee
bash trainer/train_multimodal_unified.sh pargo
```

**Important Note**: The "multiview_fusion" model type in our codebase implements the MVX-LLM architecture described in the paper, featuring:
- Dual Cross-Attention (DCA) for RGB/Depth/Event fusion
- Independent PointNet++ encoder for LiDAR
- Query Attention (QAttn) aggregation mechanism
- LoRA fine-tuning for Vicuna-7B-v1.5

### Advanced Training Options

#### MultiView-Fusion with Different Aggregation Variants

```bash
# Global Average Pooling (default)
bash trainer/train_multimodal_unified.sh multiview_fusion --fusion_variant gap

# Query-based Attention
bash trainer/train_multimodal_unified.sh multiview_fusion --fusion_variant qattn

# Spectral Query Attention
bash trainer/train_multimodal_unified.sh multiview_fusion --fusion_variant qattn_spectral

# Depth-Gated Query Attention
bash trainer/train_multimodal_unified.sh multiview_fusion --fusion_variant qattn_depthgate
```

#### MultiView-Fusion with Projection Layers

```bash
# Honeybee C-Abstractor projection
bash trainer/train_multimodal_unified.sh multiview_fusion_honeybee

# ParGo Partial-Global projection
bash trainer/train_multimodal_unified.sh multiview_fusion_pargo
```

---

## Ablation Experiments

### Modality Ablation

Test different modality combinations:

```bash
# RGB only
bash trainer/train_multimodal_unified.sh ablation --modalities rgb

# RGB + Depth
bash trainer/train_multimodal_unified.sh ablation --modalities rgb depth

# RGB + Depth + Event
bash trainer/train_multimodal_unified.sh ablation --modalities rgb depth event

# Depth + Event (no RGB)
bash trainer/train_multimodal_unified.sh ablation --modalities depth event

# Add LiDAR to any combination
bash trainer/train_multimodal_unified.sh ablation --modalities rgb depth --enable_lidar
```

**Output Naming Convention:**

- `checkpoints_ablation_rgb/` - RGB only
- `checkpoints_ablation_depth_rgb/` - Depth + RGB
- `checkpoints_ablation_depth_event_rgb_lidar/` - All modalities

---

## Model Architectures

### MVX-LLM (Our Proposed Model)

Our primary contribution addressing the limitation that existing MLLMs cannot process multiple complementary sensor modalities.

**Architecture Overview:**

```
RGB (4 views) ──┐
Depth (4 views) ├──> CLIP ViT ──> Spatial Cross-Attention ──┐
Event (4 views) ┘                  Channel Cross-Attention ──┼──> QAttn Aggregation ──> LLM
                                                               │
LiDAR Points ────────> PointNet++ ────────────────────────────┘
```

**Key Components:**

1. **Dual Cross-Attention (DCA)**:
   - **Spatial Cross-Attention**: Establishes spatial correspondences between modalities (8 heads, 64 dims each)
   - **Channel Cross-Attention**: Enables cross-channel feature enhancement (4 heads, 12 dims each)
   - Operates on 48-token aligned representations (512-dimensional)

2. **Token Aggregation Variants**:
   - **GAP**: Global Average Pooling (baseline, no parameters)
   - **QAttn**: Learnable query-based aggregation (best performance: GPTScore 55.5)
   - **Spectral QAttn**: Frequency domain processing via depthwise convolution
   - **DepthGate QAttn**: Depth-guided confidence estimation

3. **LiDAR Processing**:
   - PointNet++ hierarchical set abstraction
   - Independent processing (not participating in cross-modal fusion)
   - 512-dimensional projection for LLM integration

**Training Configuration:**
- **Optimizer**: Adam with learning rate 1e-4
- **Batch Size**: 16
- **LoRA**: Rank 16 for parameter-efficient fine-tuning
- **GPUs**: 4x NVIDIA A100-40GB with mixed precision
- **LLM Backbone**: Vicuna-7B-v1.5

**Performance (GPT-4o-mini Score):**
- Overall: 55.5
- Foggy: 53.5 
- Camera Overexposure: 51.3
- LiDAR Jitter: 57.4

---

### Baseline Comparison Models

#### 1. Baseline (Prepend)

Standard multi-modal fusion with simple token concatenation.

**Key Features:**
- Independent vision encoders (CLIP-based for 2D, PointNet++ for 3D)
- Simple concatenation of modality tokens
- Direct projection to LLM embedding space

**Performance**: GPTScore 55.1 

---

#### 2. Self-Query

Self-query mechanism with abnormal condition modeling but lacking cross-modal vision-language integration.

**Key Features:**
- Abnormal scenario consideration
- No cross-modal fusion mechanism
- Self-attention based feature refinement

**Performance**: GPTScore 40.6 (overall), demonstrates importance of cross-modal fusion

---

#### 3. Honeybee

C-Abstractor based projection for efficient token compression.

**Key Features:**
- Deformable attention with C-Abstractor layers
- Configurable token number (default: 64)
- Pooling-based spatial reduction

**Configuration**:
- `honeybee_num_tokens`: 64
- `honeybee_c_abs_layers`: 2
- `honeybee_pooling_size`: 2

**Performance**: GPTScore 48.2 (with GAP aggregation)

---

#### 4. ParGo

Partial-Global encoder with hierarchical feature processing.

**Key Features:**
- Partial encoder for local features
- Global encoder for holistic understanding
- Temperature-controlled attention fusion

**Configuration**:
- `pargo_num_tokens`: 64
- `pargo_partial_layers`: 2
- `pargo_global_layers`: 2
- `pargo_fusion_dim`: 512
- `pargo_temperature`: 0.1

**Performance**: GPTScore 33.2 (with GAP aggregation)

---

## 🔬 MVX-LLM Technical Details

### Dual Cross-Attention (DCA) Mechanism

The core innovation of MVX-LLM is the Dual Cross-Attention mechanism that fuses RGB, Depth, and Event camera data through complementary pathways:

#### 1. Spatial Alignment

All modalities are aligned to 48-token representation (from native 49 tokens):

```
F'_rgb = F_rgb × W_rgb_align ∈ R^(B×48×512)
F'_depth = F_depth × W_depth_align ∈ R^(B×48×512)  
F'_event = F_event × W_event_align ∈ R^(B×48×512)
```

#### 2. Spatial Cross-Attention

Establishes spatial correspondences (8 heads, 64 dims each):

```
F_s = MultiHeadAttn_s(Q_rgb, K_multi, V_multi)
where:
  Q_rgb = F'_rgb ∈ R^(B×48×512)
  K_multi = V_multi = [F'_depth; F'_event] ∈ R^(B×96×512)
```

#### 3. Channel Cross-Attention

Cross-channel enhancement (4 heads, 12 dims each):

```
F_c = MultiHeadAttn_c(Q^T_rgb, K'_multi, V'_multi)
where:
  Q^T_rgb = (F'_rgb)^T ∈ R^(B×512×48)
  K'_multi = V'_multi = [F'_depth; F'_event]^T × W_align
```

#### 4. Fusion & Aggregation

```
F_fused = (F_s + F^T_c) / 2
F_final = MultiHeadAttn(Q_learn, F_fused, F_fused)  # QAttn
```

### Complete Architecture Pipeline

```
Input: RGB(4 views) + Depth(4 views) + Event(4 views) + LiDAR(16K points)
         ↓
CLIP ViT (shared) for 2D modalities → 49 tokens → Align to 48 tokens
         ↓
Dual Cross-Attention (Spatial + Channel) → F_fused (48×512)
         ↓
Query Attention Aggregation → F_2d (1×512)
         
PointNet++ for LiDAR → F_lidar (1×512)
         ↓
Concatenate [F_2d; F_lidar] → (2×512)
         ↓
Linear Projection → Vicuna-7B-v1.5 (LoRA rank-16)
         ↓
Answer Generation
```

---

## Training Configuration

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./data_rebuilt` | Dataset directory |
| `--output_dir` | Auto-generated | Checkpoint output directory |
| `--batch_size` | 4 | Training batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--num_epochs` | 5 | Number of training epochs |
| `--warmup_steps` | 100 | Warmup steps |
| `--save_steps` | 1000 | Save checkpoint interval |
| `--eval_steps` | 250 | Evaluation interval |

### LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_lora` | True | Enable LoRA fine-tuning |
| `--lora_rank` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.1 | LoRA dropout rate |

### Example: Custom Training

```bash
bash trainer/train_multimodal_unified.sh multiview_fusion \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --warmup_steps 200 \
  --save_steps 500 \
  --eval_steps 100 \
  --use_lora \
  --lora_rank 32 \
  --fusion_variant qattn_spectral
```

---

## Output Structure

After training, checkpoints are saved with the following structure:

```
checkpoints_<model_type>/
├── logs/
│   └── training.log              # Training logs
├── epoch-1/                      # Epoch checkpoints
├── epoch-2/
├── ...
└── final/                        # Final model
```
---

## Evaluation

### Evaluation Framework

The repository includes a comprehensive evaluation framework supporting both traditional NLG metrics and GPT-based semantic evaluation.

#### Available Evaluation Scripts

```
eval/
├── evaluate_with_gpt.py           # Main evaluation script with GPT support
├── evaluate_ablation.py           # Ablation study evaluation
├── enhanced_evaluator.py          # Comprehensive evaluator class
├── evaluation_metrics.py          # Traditional NLG metrics (BLEU, ROUGE, CIDEr, METEOR)
├── gpt_evaluator.py              # GPT-4o based semantic scoring
├── metadata_extractor.py         # Scene and question metadata extraction
└── run_eval_gpt_4gpu.sh          # Unified evaluation bash script
```

### Supported Metrics

#### Traditional NLG Metrics

- **BLEU-1/2/3/4**: N-gram overlap precision
- **ROUGE-L**: Longest common subsequence F-measure
- **CIDEr**: Consensus-based image description evaluation
- **METEOR**: Semantic similarity with WordNet
- **Sentence Similarity**: Embedding-based cosine similarity

#### GPT-based Semantic Evaluation for Driving Scenes

GPT-4o evaluates answer correctness considering the safety-critical nature of driving scenarios:

- **Scoring Range**: 1-5 scale
  - 5: Perfect match or correct in meaning (safety-critical info accurate)
  - 4: Key information correct, minor flaws
  - 3: Partially correct (some safety-relevant details missing)
  - 2: Mostly wrong but some relevance
  - 1: Completely wrong or nonsense (potentially dangerous)

- **DRIVEXQA Analysis Dimensions**:
  - Overall performance across all scenarios
  - **By Weather Condition**: sunny, foggy, night, rainy
  - **By Question Type**: Global (scene understanding), Local (object detection), Ego (walker perspective)
  - **By Sensor Condition**: 
    - Normal operation
    - Camera faults (overexposure, underexposure, motion blur)
    - LiDAR degradation (jitter)
    - Event camera degradation (low resolution)
    - Multi-sensor failures

### Quick Start: Evaluation

#### Basic Evaluation

```bash
# Evaluate a trained model with GPT scoring
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_multiview_fusion/final \
  --model_type multiview_fusion \
  --eval_split val \
  --openai_api_key YOUR_API_KEY
```

#### Evaluation Options

```bash
# Disable GPT evaluation (traditional metrics only)
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_baseline/final \
  --model_type baseline \
  --disable_gpt_eval

# Evaluate specific number of samples
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_honeybee/final \
  --model_type honeybee \
  --max_samples 500 \
  --openai_api_key YOUR_API_KEY

# Custom output directory
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_pargo/final \
  --model_type pargo \
  --output_dir my_evaluation_results
```

#### Ablation Study Evaluation

```bash
# Evaluate modality ablation
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_ablation_rgb_depth/final \
  --model_type ablation \
  --modalities rgb depth \
  --enable_lidar \
  --openai_api_key YOUR_API_KEY

# Evaluate without LiDAR
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_ablation_rgb/final \
  --model_type ablation \
  --modalities rgb \
  --openai_api_key YOUR_API_KEY

# Evaluate attention type ablation
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_ablation_rgb_depth/final \
  --model_type ablation \
  --modalities rgb depth \
  --attention_type spatial \
  --openai_api_key YOUR_API_KEY
```

#### MultiView-Fusion Variant Evaluation

```bash
# Evaluate different fusion variants
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_multiview_fusion_qattn/final \
  --model_type multiview_fusion \
  --fusion_variant qattn \
  --openai_api_key YOUR_API_KEY

# Evaluate with different token layouts
bash eval/run_eval_gpt_4gpu.sh \
  ./checkpoints_multiview_fusion_triple/final \
  --model_type multiview_fusion \
  --fusion_token_layout triple \
  --openai_api_key YOUR_API_KEY
```

### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `checkpoint_path` | Path to trained model checkpoint | Required |
| `--model_type` | Model architecture type | `multiview_fusion` |
| `--eval_split` | Evaluation split (val/test) | `val` |
| `--batch_size` | Inference batch size | 4 |
| `--max_samples` | Maximum samples to evaluate | All |
| `--output_dir` | Evaluation results directory | Auto-generated |
| `--openai_api_key` | OpenAI API key for GPT evaluation | `$OPENAI_API_KEY` |
| `--disable_gpt_eval` | Skip GPT evaluation | False |
| `--data_dir` | Dataset directory | `./data_rebuilt` |

#### Model-Specific Parameters

**For ablation experiments:**
- `--modalities`: Modality list (rgb, depth, event, pointcloud)
- `--enable_lidar`: Include LiDAR point cloud
- `--attention_type`: Attention mechanism (spatial/channel/both)

**For MultiView-Fusion:**
- `--fusion_variant`: Aggregation variant (gap/qattn/qattn_spectral/qattn_depthgate)
- `--fusion_token_layout`: Token layout (single/triple)

### Evaluation Output Structure

After evaluation, results are organized as:

```
gpt_eval_results_<model_type>_<config>_<split>/
├── evaluation.log                          # Detailed execution log
├── predictions.json                        # All model predictions
├── traditional_metrics.json                # BLEU, ROUGE, CIDEr, METEOR scores
├── gpt_evaluation_results_<timestamp>.jsonl  # Per-sample GPT scores
├── gpt_evaluation_analysis_<timestamp>.json  # GPT score analysis
└── comprehensive_evaluation_report.json    # Combined report
```

### Evaluation Results Format

#### Traditional Metrics (`traditional_metrics.json`)

```json
{
  "bleu_1": 0.45,
  "bleu_2": 0.32,
  "bleu_3": 0.24,
  "bleu_4": 0.18,
  "rouge_l": 0.52,
  "cider": 1.23,
  "meteor": 0.38,
  "sentence_similarity": 0.76,
  "avg_prediction_length": 15.2,
  "avg_groundtruth_length": 14.8,
  "prediction_length_ratio": 1.03
}
```

#### GPT Evaluation Analysis (`gpt_evaluation_analysis_<timestamp>.json`)

```json
{
  "total_samples": 1000,
  "successful_evaluations": 998,
  "overall_score": {
    "raw_avg": 4.15,
    "standardized": 78.75,
    "distribution": {
      "1": 12,
      "2": 45,
      "3": 123,
      "4": 456,
      "5": 362
    }
  },
  "by_weather": {
    "sunny": {
      "count": 250,
      "raw_avg": 4.32,
      "standardized": 83.00
    },
    "foggy": {
      "count": 250,
      "raw_avg": 3.98,
      "standardized": 74.50
    }
  },
  "by_question_type": {
    "Global": {
      "count": 333,
      "raw_avg": 4.25,
      "standardized": 81.25
    },
    "Local": {
      "count": 334,
      "raw_avg": 4.10,
      "standardized": 77.50
    }
  }
}
```
### Environment Setup for Evaluation

```bash
# Install evaluation dependencies
pip install openai>=1.0.0
pip install sentence-transformers
pip install rouge-score
pip install pycocoevalcap
pip install nltk

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Set OpenAI API key (optional)
export OPENAI_API_KEY="your-api-key-here"
```
---

## 🙏 Acknowledgments

We thank the following:
- Vision encoders based on CLIP (Radford et al., 2021) and PointNet++ (Qi et al., 2017)
- LLM backbone: Vicuna-7B-v1.5 from LMSYS
- Training framework built with PyTorch and HuggingFace Transformers
- Evaluation framework uses OpenAI GPT-4o-mini for semantic scoring
- **LEO (embodied-generalist)**: Our training pipeline is inspired by and builds upon the [LEO framework](https://github.com/embodied-generalist/embodied-generalist) for embodied AI
