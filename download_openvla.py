# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from safetensors.torch import load_file

import torch

weights_path_0 = "/share/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0/model-00001-of-00003.safetensors"
weights_path_1 = "/share/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0/model-00002-of-00003.safetensors"
weights_path_2 = "/share/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0/model-00003-of-00003.safetensors"
state_dict0 = load_file(weights_path_0)
state_dict1 = load_file(weights_path_1)
state_dict2 = load_file(weights_path_2)
merged_state_dict = {}
for state_dict in [state_dict0, state_dict1, state_dict2]:
    merged_state_dict.update(state_dict)

print(merged_state_dict)

'''
# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

image: Image.Image = Image.open('/share/ch_collect/ch_collect_keypoints_rlbench_0121/predict_results/close_box/exp_cb_cll_prib_tsd_4_freeze_vit_window0_AR_diff/images/episode1/000.png') 
prompt = "In: What action should the robot take to close the lid on the box?\nOut: "

inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(action)

'''