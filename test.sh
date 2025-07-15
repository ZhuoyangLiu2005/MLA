ENVIRONMENT=/share/miniconda3/envs/4dvla_diff
source /share/miniconda3/bin/activate $ENVIRONMENT
cd /share/code/4D_VLA/CogACT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/share/huggingface

export PYTHONPATH=/share/code/4D_VLA/CogACT:/share/code/4D_VLA/CogACT/vla:$PYTHONPATH
export PYTHONPATH=/share/code/4D_VLA/CogACT/vlm:$PYTHONPATH

python /share/code/4D_VLA/CogACT/test.py
