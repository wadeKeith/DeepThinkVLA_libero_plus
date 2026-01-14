# openpi fast oft

## Installation

```bash
conda create -n openpi_fast_oft python=3.10 -y

pip install -r requirements.txt

cd RoboTwin

bash script/_install.sh

bash script/_download_assets.sh

```

Download Datasets
```bash
huggingface-cli download --repo-type dataset --token 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 --resume-download yinchenghust/libero_cot --local-dir data/datasets/physical-intelligence/libero_cot
```

Upload local model to huggingface

```bash
# Upload base model

huggingface-cli upload --repo-type model --token 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 yinchenghust/openpi_fast_oft_base physical-intelligence/pi0fast_base

# Upload trained model

huggingface-cli upload --repo-type model --token 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 --private yinchenghust/openpi_fast_oft_libero_cot physical-intelligence/sft_cot


# Download base model
huggingface-cli download --repo-type model --token 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 --resume-download yinchenghust/openpi_fast_oft_base --local-dir physical-intelligence/pi0fast_base/

# Download trained model
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --repo-type model --token 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 --resume-download yinchenghust/openpi_fast_oft_libero_cot --local-dir physical-intelligence/sft_cot/
```

## EVAL LIBERO

```bash
nohup bash eval.sh >libero_object.log 2>&1 &

nohup bash eval.sh >libero_spatial.log 2>&1 &

nohup bash eval.sh >libero_goal.log 2>&1 &

nohup bash eval.sh >libero_10.log 2>&1 &

```

# Download RL trained model
```bash
huggingface-cli download --repo-type model --token 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 --resume-download yinchenghust/fast_oft_rl --local-dir physical-intelligence/rl_cot/
```

## EVAL ROBOTWIN

```bash
nohup bash eval_robotwin.sh >lift_pot.log 2>&1 &

nohup bash eval_robotwin.sh >beat_block_hammer.log 2>&1 &

nohup bash eval_robotwin.sh >pick_dual_bottles.log 2>&1 &

nohup bash eval_robotwin.sh >place_phone_stand.log 2>&1 &

nohup bash eval_robotwin.sh >move_can_pot.log 2>&1 &

nohup bash eval_robotwin.sh >place_a2b_left.log 2>&1 &

nohup bash eval_robotwin.sh >place_empty_cup.log 2>&1 &

nohup bash eval_robotwin.sh >handover_mic.log 2>&1 &

nohup bash eval_robotwin.sh >handover_block.log 2>&1 &

nohup bash eval_robotwin.sh >stack_bowls_two.log 2>&1 &

nohup bash eval_robotwin.sh >blocks_ranking_rgb.log 2>&1 &

nohup bash eval_robotwin.sh >put_bottles_dustbin.log 2>&1 &

```