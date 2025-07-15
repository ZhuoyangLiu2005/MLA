# 每次开始训练记得检查视角！！！

# ENVIRONMENT=/media/miniconda3/envs/efvla
# conda activate $ENVIRONMENT
cd /media/liuzhuoyang/new_vla/5D_VLA_beta/CogACT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/media/huggingface
export PYTHONPATH=/media/liuzhuoyang/new_vla/5D_VLA_beta/CogACT/vlm:$PYTHONPATH

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=3600

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIMM_OFFLINE=1

# for debug
# export WANDB_MODE=offline

# training settings
FUTURE_ACTION_STEPS=0
FREEZE_VISON=false
FREEZE_LLM=false
LOAD_DIT=false
ACTION_TOKENIZER_EXIST=true
USE_DIFF=false
AR_DIFF_LOSS=false
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0
PRETRAIN=openvla
LLM_VISION_LAYERS=8

SETTING=Pretrain${PRETRAIN}_FreezeVis${FREEZE_VISON}_Window${FUTURE_ACTION_STEPS}_Contrastive_256_Vislayer${LLM_VISION_LAYERS}_5dft_1024_0403_0622

DATA_MIX=rtx_dataset
TASK=rtx_dataset_4
BATCH_SIZE=8
EPOCHS=10
LEARNING_RATE=2e-5

NUM_GPUS=8
NODES=1
MASTER_ADDR="10.200.64.219"
NODE_RANK=0
LOG_DIR="./logs_pretrain_0623"


DATA_ROOT=/media/rlds_data
EXP_ROOT=/media/liuzhuoyang/new_vla/5D_VLA_beta/pretrain_exp


torchrun --nnodes $NODES --nproc-per-node $NUM_GPUS --node_rank=$NODE_RANK --master_addr=${MASTER_ADDR} --master_port=29500 scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${DATA_MIX} \
  --vla.base_vlm prism-dinosiglip-224px+7b \
  --vla.expected_world_size $((${NUM_GPUS} * ${NODES})) \
  --vla.per_device_batch_size ${BATCH_SIZE} \
  --vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE})) \
  --vla.learning_rate ${LEARNING_RATE} \
  --vla.epochs ${EPOCHS} \
  --vla.freeze_vision_tower ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir ${DATA_ROOT}/${TASK} \
  --run_root_dir ${EXP_ROOT} \
  --run_id exp_${TASK}_${SETTING} \
  --image_aug false \
  --wandb_project one_model_vla_pretrain \
  --wandb_entity liumail2023-peking-university \
  --save_interval 1 \
  --action_dim 7 \
  --ar_zoom 1.0 \
  --diff_zoom 1.0 \
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --load_dit ${LOAD_DIT} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --ar_diff_loss ${AR_DIFF_LOSS} \
  --action_model_type DiT-B \
  --is_resume False \
  --pretrained_checkpoint "/media/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"
  > $LOG_FILE 2>&1
  # --pretrained_checkpoint "/media/liuzhuoyang/new_vla/5D_VLA_beta/exp/exp_4tasks_selected_keyframe_pointcloud_4608_0619_Pretrainvlm_FreezeVistrue_Window0_Contrastive_576_Vislayer8_5dft_1024_0403_0619/checkpoints/step-015930-epoch-300-loss=1.5462.pt"
  # --pretrained_checkpoint "/media/liuzhuoyang/new_vla/2D_VLA_beta/pretrain_exp/exp_rtx_dataset_4_freeze_vit_window0_huoshan_eve_pretrain/checkpoints/step-201072-epoch-01-loss=1.7223.pt"
  # --pretrained_checkpoint "/media/huggingface/hub/models--CogACT--CogACT-Base/snapshots/6550bf0992f162fc5d74f14ffee30771a9433363/checkpoints/CogACT-Base.pt"
  # --pretrained_checkpoint "/home/liuzhuoyang/4D_VLA-hao-beta/pretrain-exp/ours1/checkpoints/step-053620-epoch-02-loss=1.0470.pt"
  # --pretrained_checkpoint "/home/liuzhuoyang/4D_VLA-hao/exp/exp_15tasks_selected_keyframe_cogact_pretrain_freeze_vit_window0_ar/checkpoints/step-006601-epoch-300-loss=0.0000.pt" \
  # --pretrained_checkpoint "/media/code/CogACT/exp/exp_rtx_dataset_clean_freeze_none_window15/checkpoints/step-028459-epoch-01-loss=0.0434.pt"
  # --pretrained_checkpoint '/media/code/4D_VLA/CogACT/exp/exp_rtx_dataset_clean_our_pretrain_clean_freeze_none_window0/checkpoints/step-056917-epoch-01-loss=0.1391.pt' \