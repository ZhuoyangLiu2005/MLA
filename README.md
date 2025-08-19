# LLM_policy

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
# ...
# training settings

USE_POINTCLOUD=true
USE_CONTRASTIVE=true
LLM_VISION_LAYERS=8
USE_REC=false
RECON_IMG=false
USE_ROI=false
RECON_PC=false

# ...

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

