# 🔥 DeepThinkVLA Zero-Shot Evaluation on LIBERO+ 🔥

This repository is **specifically for zero-shot evaluation on LIBERO+**:

- **Training**: DeepThinkVLA is trained **only on the standard LIBERO dataset** (no LIBERO+ fine-tuning).
- **Evaluation**: the trained model is **directly evaluated on LIBERO+** (zero-shot transfer) using the scripts here.

For the full DeepThinkVLA project (paper, released checkpoints, training/RL pipeline), refer to the official implementation:
- [OpenBMB/DeepThinkVLA](https://github.com/OpenBMB/DeepThinkVLA)

## 🔗 Quick Links

- **Official DeepThinkVLA**: [OpenBMB/DeepThinkVLA](https://github.com/OpenBMB/DeepThinkVLA)
- **Zero-shot eval entrypoint**: `experiments/run_libero_plus_eval.py`
- **Model wrapper**: `sft/modeling_deepthinkvla.py` (class `DeepThinkVLA`)

## 🛠️ Setup

Tested with Python >= 3.10 on Linux + NVIDIA GPUs.

```bash
conda create -n deepthinkvla python=3.10 -y
conda activate deepthinkvla
pip install -r requirements.txt
```

## 💾 Checkpoint (trained on LIBERO only)

`--pretrained_checkpoint` should point to a **DeepThinkVLA checkpoint trained on LIBERO**.

- **Released SFT checkpoint (LIBERO)**: you can download the LIBERO SFT model from [yinchenghust/deepthinkvla_libero_cot_sft](https://huggingface.co/yinchenghust/deepthinkvla_libero_cot_sft).

- **Local path (recommended)**: a folder with Hugging Face-style files (`config.json`, tokenizer files, weights, etc.)
- **Hugging Face model id**: supported as well (use `huggingface-cli login` if private)

- **Security**: never commit tokens; use `huggingface-cli login` or environment variables locally.

## 🧪 Zero-shot Evaluation on LIBERO+

Run evaluation directly:

```bash
python experiments/run_libero_plus_eval.py \
  --pretrained_checkpoint /path/to/deepthinkvla_libero_checkpoint \
  --num_images_in_input 2 \
  --task_suite_name libero_10 \
  --max_new_tokens 2048 \
  --swanlab_mode disabled
```

Or use the wrapper script:

```bash
bash eval.sh
```

### Outputs

- **Logs**: `experiments/logs/`
- **Rollout videos** (if enabled by the script): `rollouts/`

### Optional logging (SwanLab)

By default, logging is disabled. To enable SwanLab logging:

- **Set**: `SWANLAB_API_KEY` in your environment
- **Set**: `SWANLAB_MODE` to `cloud-only` or `local`

## 📊 Zero-shot Results (LIBERO+)

The following numbers are **zero-shot success rates (SR)** on **LIBERO+**, evaluated with a DeepThinkVLA model **trained only on LIBERO** (no LIBERO+ fine-tuning).

### Breakdown by shift type

| Objects Layout | Language Instructions | Light Conditions | Camera Viewpoints | Robot Initial States | Background Textures | Sensor Noise | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7993 | 0.845 | 0.900 | 0.885 | 0.405 | 0.753 | 0.944 | 0.790 |

### Breakdown by task suite

| object | spatial | goal | 10 | Total |
|---:|---:|---:|---:|---:|
| 0.840 | 0.879 | 0.697 | 0.746 | 0.790 |

## 📁 Repository Structure

```text
DeepThinkVLA_libero_plus/
├── experiments/
│   ├── run_libero_plus_eval.py   # zero-shot LIBERO+ evaluation entrypoint
│   ├── deepthinkvla_utils.py     # model loading + decoding helpers
│   └── libero_utils.py           # env/image/video helpers
├── libero/                       # LIBERO simulator (assets + tasks + benchmark)
├── sft/                          # minimal model wrapper used by eval
├── data/                         # dataset helpers + action tokenizer + normalization
├── eval.sh                       # example eval launcher
├── requirements.txt
└── README.md
```

## 🙏 Acknowledgements

This repo builds on the DeepThinkVLA project and the LIBERO simulator ecosystem.
See the full acknowledgements in the official repo:
- [OpenBMB/DeepThinkVLA](https://github.com/OpenBMB/DeepThinkVLA)