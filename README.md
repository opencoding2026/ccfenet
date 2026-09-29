# Cross-Modal Closed-LoopFeature Enhancement Network forLanguage-Guided Remote Sensing lmage Target Recognition

This repository contains the open-source code for CCFENet.

## Preparation

Use Linux with a CUDA-enabled PyTorch installation compatible with the local GPU. Install PyTorch and torchvision for that CUDA version first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Provide the following files or set the corresponding environment variables:

```text
pretrained/internlm-xcomposer2d5-7b/          MODEL: base model and tokenizer
pretrained/internlm-xcomposer2d5-clip/        IXC_CLIP_PATH: vision tower
pretrained/sam2/sam2_hiera_large.pt           SAM2_CKPT: SAM2 weights
data/LaSeRS/train/images/                     LASERS_BASE_PATH
data/LaSeRS/train/annotations/train_data.json
data/LaSeRS/test/images/
data/LaSeRS/test/annotations/*.json
```

LaSeRS annotations are JSON lists. Each item needs `id`, `image_name`, `description`, `answer`, and `mask` (one COCO RLE per target in answer order). The answer's `[SEG]` count must match the number of target masks for teacher-forced evaluation. `data_lasers.py` uses the published train/test directory layout directly; no conversion step is needed.

## Train

```bash
bash train_lasers.sh
```

The default configuration uses one GPU, BF16, LoRA, DeepSpeed ZeRO-2, microbatch size 2 and gradient accumulation 10. Override paths without editing the launcher:

```bash
MODEL=/path/to/base-model \
IXC_CLIP_PATH=/path/to/vision-tower \
SAM2_CKPT=/path/to/sam2_hiera_large.pt \
LASERS_BASE_PATH=/path/to/LaSeRS \
OUTPUT_DIR=/path/to/output \
bash train_lasers.sh
```

## Evaluate

Training writes a DeepSpeed checkpoint and a LoRA adapter under `checkpoint-last`. The evaluator also loads a full model state to restore non-LoRA modules. Export it once after training:

```bash
python export_lasers_fp32.py output_lasers_train/checkpoint-last
bash eval_lasers_export.sh
```

The exporter materializes the full state in CPU memory and may need substantial RAM and disk space. Set `CHECKPOINT_PATH` and `FULL_STATE_DICT_PATH` when evaluating another run. The default evaluation mode is `generate`; to evaluate with fixed answer tokens, set `INFERENCE_MODE=teacher_forced`. `ANNOTATION_SPLIT` selects one test JSON file; when unset, all JSON splits under `test/annotations` are evaluated. Results go to `lasers_export/` by default, including per-target masks, combined multi-target overlays, JSONL records and `lasers_export_summary.json`.

The evaluation reports ordered IoU metrics: target masks are paired by their sequence order, without rematching predicted objects to ground truth. `AverageOrderedMetrics` averages split-level metrics equally; `OverallOrderedMetrics` recomputes metrics over all test records.

## Acknowledgement 
We appreciate GeoPixel for making their models and code available as open-source contributions.

