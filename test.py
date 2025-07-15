from PIL import Image
from vla import load_vla
import torch
import numpy as np
def model():
    model = load_vla(
            '/home/lzy/4D_VLA-hao-beta/exp/exp_10tasks_selected_keyframe_mulview_ours2_2_pretrain_freeze_vit_window0_diff+ar_boi_eoi_state_mlp_500_new/checkpoints/step-001715-epoch-50-loss=0.0203.pt',
            load_for_training=False,
            action_model_type='DiT-B',
            future_action_window_size=0,
            load_dit = False,
            hf_token="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",
            use_diff=True
            )
    model.to('cuda:4').eval()
    image: Image.Image = Image.open('/home/lzy/4D_VLA-hao-beta/CogACT/000.png') 
    prompt = "close the lid on the box"
    action, action1, conf, infer_time = model.predict_action_diff_ar(
            front_image=image,
            instruction=prompt,
            unnorm_key = 'rlbench',
            cfg_scale = 0.0, 
            use_ddim = True,
            num_ddim_steps = 10,
            action_dim = 7,
            cur_robot_state = np.array([ 0.27849028, -0.00815899,  1.47193933, -3.14159094,  0.24234043,  3.14158629,  1.        ]),
            )
    print(infer_time[0])
    print(infer_time[1])

model()



####### infer注意: 尝试两种infer，一种是直接读actions[0]，另一种参考如下
'''
action计算方法: 论文公式(4),K设置为2:

t = 0 的时候 就是action_t0_output[0]
t = 1 的时候 w0*action_t0_output[1]+w1*action_t1_output[0]
t = 2 的时候 w0*action_t0_output[2]+w1*action_t1_output[1]+w2*action_t2_output[0]
t = 3 的时候 w0*action_t1_output[2]+w1*action_t2_output[1]+w2*action_t3_output[0]
t = 3 的时候 w0*action_t2_output[2]+w1*action_t3_output[1]+w2*action_t4_output[0]
.
.
.

w的计算方法: 论文公式(5)

'''