set -x

export TOKENIZERS_PARALLELISM=false
export SWANLAB_PROJECT_NAME='openpi_fast_oft'
export SWANLAB_API_KEY='5WXb6nm31cT5JlXCRyAcG'
export SWANLAB_MODE='disabled' # cloud-only, local, disabled
export CUDA_VISIBLE_DEVICES=3


# libero_object, libero_spatial, libero_goal, libero_10, libero_90

python experiments/run_libero_eval.py \
    --pretrained_checkpoint physical-intelligence/rl_cot_best \
    --num_images_in_input 2 \
    --task_suite_name libero_10 \
    --max_new_tokens 2048 \
    --project_name $SWANLAB_PROJECT_NAME \
    --swanlab_api_key $SWANLAB_API_KEY \
    --swanlab_mode $SWANLAB_MODE \
    --seed 429 \
    --panel_width_px 812 \
    --scale_down_value 2.0 \
