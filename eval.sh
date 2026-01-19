set -x

export TOKENIZERS_PARALLELISM=false
export SWANLAB_PROJECT_NAME='deepthinkvla'
export SWANLAB_API_KEY='YOUR_API_KEY'
# For open-source safety, do NOT hardcode credentials here.
# If you want SwanLab logging, set SWANLAB_API_KEY in your environment and set SWANLAB_MODE to 'cloud-only' or 'local'.
export SWANLAB_MODE='disabled' # cloud-only, local, disabled
export CUDA_VISIBLE_DEVICES=0


# libero_object, libero_spatial, libero_goal, libero_10, libero_90

python experiments/run_libero_plus_eval.py \
    --pretrained_checkpoint yinchenghust/sft_cot \
    --num_images_in_input 2 \
    --task_suite_name libero_object \
    --max_new_tokens 2048 \
    --project_name $SWANLAB_PROJECT_NAME \
    --swanlab_api_key $SWANLAB_API_KEY \
    --swanlab_mode $SWANLAB_MODE \
    --seed 429 \
    --panel_width_px 812 \
