
cd /media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy

export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy:$PYTHONPATH
export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vlm:$PYTHONPATH
export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/transformers:$PYTHONPATH

export HF_HOME=/media/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIMM_OFFLINE=1

# for debug
export WANDB_MODE=offline

# training settings
FUTURE_ACTION_STEPS=0
FREEZE_VISON=true
FREEZE_LLM=false
ACTION_TOKENIZER_EXIST=false
USE_DIFF=true
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0
PRETRAIN=pre0815
USE_POINTCLOUD=true
USE_CONTRASTIVE=true
LLM_VISION_LAYERS=8
USE_REC=false
RECON_IMG=false
USE_ROI=false
RECON_PC=false

SETTING=Pretrain${PRETRAIN}_FreezeVis${FREEZE_VISON}_Window${FUTURE_ACTION_STEPS}_Diff${USE_DIFF}_Rec${USE_REC}ALL_Contrastive_Vislayer${LLM_VISION_LAYERS}_1024_0403_0818

TASK=6tasks_selected_keyframe_nextpc_0806
BATCH_SIZE=8
EPOCHS=300
LEARNING_RATE=2e-5

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8 # gpus per machine
NODES=1
MASTER_ADDR="10.200.64.222" # ifconfig
NODE_RANK=0

DATA_ROOT=/media/liuzhuoyang/data/rlbench/rlds
EXP_ROOT=/media/liuzhuoyang/new_vla/Rec_Diff_beta/exp

torchrun --standalone --nnodes ${NODES} --nproc-per-node ${NUM_GPUS} scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix rlbench \
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
  --wandb_project one_model_vla_sft \
  --wandb_entity liumail2023-peking-university \
  --save_interval 100 \
  --action_dim 7 \
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --use_contrastive ${USE_CONTRASTIVE} \
  --use_pointcloud ${USE_POINTCLOUD} \
  --use_reconstruction ${USE_REC} \
  --recon_image ${RECON_IMG} \
  --use_roi ${USE_ROI} \
  --recon_pointcloud ${RECON_PC} \
  --is_resume False \
  --pretrained_checkpoint "/media/liuzhuoyang/new_vla/Rec_Diff_beta/pretrain-exp/exp_rtx_0812_Pretrainopenvla_FreezeVisfalse_Window0_Difftrue_Rectrue2d_Contrastive_Vislayer8_1024_0403_0815/checkpoints/step-022717-epoch-01-loss=1.4848.pt"
  # --pretrained_checkpoint "/media/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"
  # --pretrained_checkpoint "/media/liuzhuoyang/new_vla/2D_VLA_beta/pretrain_exp/exp_rtx_dataset_4_freeze_vit_window0_huoshan_eve_pretrain/checkpoints/step-201072-epoch-01-loss=1.7223.pt"
  # --pretrained_checkpoint "/media/huggingface/hub/models--CogACT--CogACT-Base/snapshots/6550bf0992f162fc5d74f14ffee30771a9433363/checkpoints/CogACT-Base.pt"
  # --pretrained_checkpoint "/home/liuzhuoyang/4D_VLA-hao-beta/pretrain-exp/ours1/checkpoints/step-053620-epoch-02-loss=1.0470.pt"
  # --pretrained_checkpoint "/home/liuzhuoyang/4D_VLA-hao/exp/exp_15tasks_selected_keyframe_cogact_pretrain_freeze_vit_window0_ar/checkpoints/step-006601-epoch-300-loss=0.0000.pt" \
  # --pretrained_checkpoint "/media/code/CogACT/exp/exp_rtx_dataset_clean_freeze_none_window15/checkpoints/step-028459-epoch-01-loss=0.0434.pt"
  # --pretrained_checkpoint '/media/code/4D_VLA/CogACT/exp/exp_rtx_dataset_clean_our_pretrain_clean_freeze_none_window0/checkpoints/step-056917-epoch-01-loss=0.1391.pt' \