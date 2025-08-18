
cd /media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy
export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy:/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vla:$PYTHONPATH
export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vlm:$PYTHONPATH
export HF_HOME=/media/huggingface
export COPPELIASIM_ROOT=/media/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIMM_OFFLINE=1

CUDA=7
N=0
Xvfb :$N -screen 0 1024x768x24 &  
export DISPLAY=:$N

models=("/media/liuzhuoyang/new_vla/Rec_Diff_beta/exp/exp_metaworld_assembly_pointcloud_0814_PretrainDiff_300_FreezeVistrue_Window0_Difftrue_Recfalse_Contrastive_Vislayer8_1024_0403_0814/checkpoints/step-017000-epoch-100-loss=3.3380.pt")

# tasks=("assembly" "button-press" "box-close" "drawer-open" "bin-picking" "dial-turn" "hammer" "hand-insert" "lever-pull#" "peg-unplug-side" "push-wall" "reach" "shelf-place" "sweep-into")
tasks=("assembly")

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'[/=]' '{print $(NF-1)}')
  for task in "${tasks[@]}"; do
    python scripts/test_metaworld.py \
      --replay_or_predict 'predict' \
      --camera_name "corner" \
      --model_path ${model} \
      --task_name ${task} \
      --save_dir /media/liuzhuoyang/new_vla/Rec_Diff_beta/metaworld/metaworld_test_0718/${exp_name} \
      --cuda ${CUDA} \
      --use_diff 1 \
      --use_robot_state 1 \
      --use_pointcloud 1 \
      --episode_max_length 200 \
      --num_episodes 20 \
      --action_dim 4
  done
done
