
cd /path/to/MLA

export PYTHONPATH=/path/to/MLA:$PYTHONPATH

# for debug
export WANDB_MODE=offline

# training settings
FUTURE_ACTION_STEPS=0
FREEZE_VISON=true
FREEZE_LLM=false
ACTION_TOKENIZER_EXIST=false
ACTION_DIM=7
USE_DIFF=true
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0
PRETRAIN=vlm

USE_POINTCLOUD=false # no point cloud in RTX dataset
USE_TAC=false # no tactile in simulator
USE_CONTRASTIVE=false
LLM_VISION_LAYERS=8
USE_GEN=false
GEN_IMG=false
USE_ROI=false
GEN_PC=false
GEN_TAC=false

SETTING=Pretrain${PRETRAIN}_FreezeVis${FREEZE_VISON}_Window${FUTURE_ACTION_STEPS}_Diff${USE_DIFF}_PC${USE_POINTCLOUD}_Contrastive${USE_CONTRASTIVE}_Gen${USE_GEN}_Vislayer${LLM_VISION_LAYERS}

TASK=name_of_your_task
BATCH_SIZE=8
EPOCHS=300
LEARNING_RATE=2e-5

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8 # gpus per machine
NODES=1
MASTER_ADDR="" # use $ifconfig$ to get
NODE_RANK=0

LOG_DIR="/path/to/MLA/logs_pretrain"
mkdir -p $LOG_DIR
if [ $NODE_RANK -eq 0 ]; then
  LOG_FILE="$LOG_DIR/main.log"
else
  LOG_FILE="$LOG_DIR/node_${NODE_RANK}.log"
fi

DATA_ROOT=/path/to/your/datasets
EXP_ROOT=/path/to/MLA/exp

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
  --wandb_project mla \
  --wandb_entity <your-w&b-account> \
  --save_interval 1 \
  --action_dim ${ACTION_DIM} \
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
  > $LOG_FILE 2>&1
# --pretrained_checkpoint "/media/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0" \
  