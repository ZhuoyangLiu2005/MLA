
cd /media/liuzhuoyang/new_vla/MLA_beta/MLA

export PYTHONPATH=/path/to/MLA:$PYTHONPATH

export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

Xvfb :0 -screen 0 1024x768x24 &  
export DISPLAY=:0

models=("/ckpt/path") 

CUDA=0
tasks=("close_box") 
# "close_box" "close_laptop_lid" "toilet_seat_down" "sweep_to_dustpan" 
# "close_fridge" "phone_on_base" "take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger" 
# "lamp_on" "place_wine_at_rack_location" "unplug_charger" "water_plants"

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'/' '{print $(NF-2)}')
  action_steps=$(echo ${exp_name} | grep -oP 'Window\K[0-9]+')
  for task in "${tasks[@]}"; do
    python scripts/test_rlbench.py \
      --exp_name ${exp_name} \
      --task_name ${task} \
      --model_path ${model} \
      --model_action_steps ${action_steps} \
      --num_episodes 20 \
      --cfg_scale 0.0 \
      --use_diff 1 \
      --use_pointcloud 1 \
      --use_wrist 0 \
      --use_robot_state 1 \
      --result_dir /path/to/MLA/rlbench \
      --cuda ${CUDA} \
      --hf_token your_huggingface_token
  done
done

