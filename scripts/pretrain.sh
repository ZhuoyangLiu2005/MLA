
cd /media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/media/huggingface

export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy:$PYTHONPATH
export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vlm:$PYTHONPATH
export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/transformers:$PYTHONPATH

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIMM_OFFLINE=1

# for debug
# export WANDB_MODE=offline

# training settings
FUTURE_ACTION_STEPS=0
FREEZE_VISON=false
FREEZE_LLM=false
ACTION_TOKENIZER_EXIST=false
USE_DIFF=true
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0
PRETRAIN=Openvla
USE_POINTCLOUD=false
USE_CONTRASTIVE=false
LLM_VISION_LAYERS=8
USE_REC=false
RECON_IMG=false
USE_ROI=false
RECON_PC=false

SETTING=Pretrain${PRETRAIN}_FreezeVis${FREEZE_VISON}_Window${FUTURE_ACTION_STEPS}_Diff${USE_DIFF}_Rec${USE_REC}2d_Contrastive_Vislayer${LLM_VISION_LAYERS}_1024_0403_0818

TASK=rtx_0812
BATCH_SIZE=8
EPOCHS=10
LEARNING_RATE=2e-5

NUM_GPUS=8
NODES=4
MASTER_ADDR="10.200.64.126" # 122: 10.200.64.219, 241: 10.200.64.126
NODE_RANK=3
LOG_DIR="/media/liuzhuoyang/new_vla/Rec_Diff_beta/pretrain-exp/exp_${TASK}_${SETTING}/logs_pretrain"
mkdir -p $LOG_DIR
if [ $NODE_RANK -eq 0 ]; then
  LOG_FILE="$LOG_DIR/main.log"
else
  LOG_FILE="$LOG_DIR/node_${NODE_RANK}.log"
fi

DATA_ROOT=/media/liuzhuoyang/data/rtx/rlds
EXP_ROOT=/media/liuzhuoyang/new_vla/Rec_Diff_beta/pretrain-exp

torchrun --nnodes $NODES --nproc-per-node $NUM_GPUS --node_rank=$NODE_RANK --master_addr=${MASTER_ADDR} --master_port=29501 scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix rtx_dataset \
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
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --use_pointcloud ${USE_POINTCLOUD} \
  --use_contrastive ${USE_CONTRASTIVE} \
  --use_reconstruction ${USE_REC} \
  --recon_image ${RECON_IMG} \
  --use_roi ${USE_ROI} \
  --recon_pointcloud ${RECON_PC} \
  --is_resume False \
  --pretrained_checkpoint "/media/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0" \
  > $LOG_FILE 2>&1