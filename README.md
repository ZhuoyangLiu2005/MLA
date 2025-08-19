# LLM-Policy

## Fine-tuning on custom datasets (beta)

We applied the policy on RLbench and Metaworld benchmark. And it's convenient to finetune on these datasets.

For RLBench and Metaworld, you should firstly generate the training dataset in RLDS dataset form. And custom the keys which match those in LLM_policy/vlm/prismatic/vla/datasets/rlds/oxe/configs.py`

Then you can fine-tune the model on these datasets:

```bash
# rlbench
bash LLM_policy/scripts/train_rlbench.sh
# metaworld
bash LLM_policy/scripts/train_metaworld.sh
```

Take `LLM_policy/scripts/train_rlbench.sh` as an example:

```bash
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

torchrun --standalone --nnodes ${NODES} --nproc-per-node ${NUM_GPUS} scripts/train.py
--vla.type prism-dinosiglip-224px+oxe+diffusion
--vla.data_mix rlbench
--vla.base_vlm prism-dinosiglip-224px+7b
--vla.expected_world_size $((${NUM_GPUS} * ${NODES}))
--vla.per_device_batch_size ${BATCH_SIZE}
--vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE}))
--vla.learning_rate ${LEARNING_RATE}
--vla.epochs ${EPOCHS}
--vla.freeze_vision_tower ${FREEZE_VISON}
--vla.freeze_llm_backbone ${FREEZE_LLM}
--data_root_dir ${DATA_ROOT}/${TASK}
--run_root_dir ${EXP_ROOT}
--run_id exp_${TASK}_${SETTING}
--image_aug false
--wandb_project one_model_vla_sft
--wandb_entity liumail2023-peking-university
--save_interval 100
--action_dim 7
--repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS}
--action_tokenizer_exist ${ACTION_TOKENIZER_EXIST}
--future_action_window_size ${FUTURE_ACTION_STEPS}
--class_dropout_prob ${CLASS_DROPOUT_PROB}
--use_diff ${USE_DIFF}
--use_contrastive ${USE_CONTRASTIVE}
--use_pointcloud ${USE_POINTCLOUD}
--use_reconstruction ${USE_REC}
--recon_image ${RECON_IMG}
--use_roi ${USE_ROI}
--recon_pointcloud ${RECON_PC}
--is_resume False
--pretrained_checkpoint "`/media/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0`"
```

The hyperparameters can be set as follows:

|                              |pretrain|ft-stage1| ft-stage2 | 
|:-----------|:-----------:|:-----------:|:-----------:|
| use_pointcloud  | false | true | false |
| use_contrastive  | false | true | true |
| use_rec                | false | false | true |
| recon_img           | false | false | true |
| use_roi                 | false | false | true/false |
| recon_pc              | false | false | true |


## Evaluation (beta)

Run the following scripts to evaluate on the RLBench benchmark:

```bash
bash LLM_policy/scripts/test_rlbench.sh
```

If the model is trained with point cloud, set the argument `use_pointcloud=1`

For Metaworld benchmark, run the script below:

```bash
bash LLM_policy/scripts/test_metaworld.sh
```

## Pre-Training on RTX-dataset (beta)

You can start the trainging from the weights of [OpenVLA](https://github.com/openvla/openvla) for greater efficiency. Please follow the instruction of [OpenVLA](https://github.com/openvla/openvla) to download their weights:

```
# From OpenVLA repo
# Change directory to your base model checkpoints folder
cd <PATH TO BASE MODEL CHECKPOINTS DIR>
# Download checkpoint (30 GB) -- may take a few minutes
git clone git@hf.co:openvla/openvla-7b-prismatic
# If the command above did not download the full checkpoint,
# manually fetch it via git Large File Storage (LFS)
# Note: You may have to configure an SSH key for this to work
cd openvla-7b-prismatic
git lfs fetch --all
```


The data of [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) can be download following [OXE](https://robotics-transformer-x.github.io/) and [OpenVLA](https://github.com/openvla/openvla). Then launch the training script. We provide the following training scripts:

```bash
bash LLM_policy/scripts/pretrain.sh
```

You can also start training from PrismaticVLM and simply ignore the ``--pretrained_checkpoint``. However, it will take longer to converge.

## License

All the code, model weights, and data are licensed under [MIT license](./LICENSE).

