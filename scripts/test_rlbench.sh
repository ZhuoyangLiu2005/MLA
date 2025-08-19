
cd /media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy
export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy:/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vla:$PYTHONPATH
export PYTHONPATH=/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vlm:$PYTHONPATH
export HF_HOME=/media/huggingface
export COPPELIASIM_ROOT=/media/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIMM_OFFLINE=1

Xvfb :0 -screen 0 1024x768x24 &  
export DISPLAY=:0

models=("/media/liuzhuoyang/new_vla/Rec_Diff_beta/exp/exp_6tasks_selected_keyframe_nextpc_0806_Pretrainpre0815_FreezeVistrue_Window0_Difftrue_RecfalseALL_Contrastive_Vislayer8_1024_0403_0818/checkpoints/step-007800-epoch-200-loss=1.3488.pt") 

CUDA=3
tasks=("sweep_to_dustpan") 
# "close_box" "close_laptop_lid" "toilet_seat_down" "sweep_to_dustpan" 
# "close_fridge" "phone_on_base" "take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger" 
# "lamp_on" "place_wine_at_rack_location" "unplug_charger" "water_plants"

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'/' '{print $(NF-2)}')
  action_steps=$(echo ${exp_name} | grep -oP 'Window\K[0-9]+')
  for task in "${tasks[@]}"; do
    python scripts/test_rlbench.py \
      --exp_name ${exp_name}_200_rec_diff \
      --task_name ${task} \
      --model_path ${model} \
      --model_action_steps ${action_steps} \
      --num_episodes 20 \
      --cfg_scale 0.0 \
      --use_diff 1 \
      --use_pointcloud 1 \
      --use_wrist 0 \
      --use_robot_state 1 \
      --result_dir /media/liuzhuoyang/new_vla/Rec_Diff_beta/rlbench/rlbench_test_0819/4tasks_0815e1_concat_256_freezetrue_difftrue_recfalse1_visionlayer8_1024_0403_0818 \
      --cuda ${CUDA} \
      --hf_token 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
  done
done

