ENVIRONMENT=/share/miniconda3/envs/4dvla_diff
source /share/miniconda3/bin/activate $ENVIRONMENT
cd /share/code/4D_VLA/CogACT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/share/huggingface
export PYTHONPATH=/share/code/4D_VLA/CogACT/vlm:$PYTHONPATH

FUTURE_ACTION_STEPS=0
SETTING=cogact_pretrain_freeze_vit_window${FUTURE_ACTION_STEPS}_diff+ar_boi_eoi
FREEZE_VISON=true
FREEZE_LLM=false
LOAD_DIT=false
ACTION_TOKENIZER_EXIST=true
USE_DIFF=true
AR_DIFF_LOSS=true
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0

TASK=6tasks_selected_keyframe_state
NUM_GPUS=8
NODES=1
BATCH_SIZE=32
EPOCHS=200
LEARNING_RATE=2e-5

DATA_ROOT=/share/rlds_data
EXP_ROOT=/share/code/4D_VLA/exp


torchrun --standalone --nnodes ${NODES} --nproc-per-node ${NUM_GPUS} scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix rlbench \
  --vla.base_vlm phi-2+3b \
  --vla.expected_world_size $((${NUM_GPUS} * ${NODES})) \
  --vla.per_device_batch_size ${BATCH_SIZE} \
  --vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE})) \
  --vla.learning_rate ${LEARNING_RATE} \
  --vla.epochs ${EPOCHS} \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir ${DATA_ROOT}/${TASK} \
  --run_root_dir ${EXP_ROOT} \
  --run_id exp_${TASK}_${SETTING}_phi3b \
  --image_aug false \
  --wandb_project cogact \
  --wandb_entity 1162737898-the-chinese-university-of-hong-kong \
  --save_interval 100 \
  --action_dim 7 \
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --load_dit ${LOAD_DIT} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --ar_diff_loss ${AR_DIFF_LOSS} \
  --action_model_type DiT-B \
  --is_resume False \
  # --pretrained_checkpoint "/share/huggingface/hub/models--CogACT--CogACT-Base/snapshots/6550bf0992f162fc5d74f14ffee30771a9433363/checkpoints/CogACT-Base.pt"
  # --pretrained_checkpoint "/share/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0" \
  # --pretrained_checkpoint "/share/code/CogACT/exp/exp_rtx_dataset_clean_freeze_none_window15/checkpoints/step-028459-epoch-01-loss=0.0434.pt"
  # --pretrained_checkpoint '/share/code/4D_VLA/CogACT/exp/exp_rtx_dataset_clean_our_pretrain_clean_freeze_none_window0/checkpoints/step-056917-epoch-01-loss=0.1391.pt' \
