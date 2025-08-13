import torch
from collections import defaultdict

def print_state_dict_keys(state_dict, indent=0):
    """递归打印 state_dict 的 keys，并显示层次结构"""
    key_tree = defaultdict(list)
    
    for key in state_dict.keys():
        parts = key.split('.')
        current_level = key_tree
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
    
    def print_tree(tree, indent=0):
        for key, subtree in tree.items():
            print("  " * indent + f"└─ {key}")
            if isinstance(subtree, dict):
                print_tree(subtree, indent + 1)
    
    print("\nCheckpoint Key Hierarchy:")
    print_tree(key_tree)

if __name__ == "__main__":
    checkpoint_path = "/media/liuzhuoyang/new_vla/Rec_Diff_beta/exp/exp_4tasks_selected_keyframe_nextpc_0806_tttPretrainDiff_300_FreezeVistrue_Window0_Difftrue_Rectrue3dpointmae_Contrastive_Vislayer8_1024_0403_0812/checkpoints/step-000027-epoch-01-loss=2.2063.pt"
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = checkpoint["model"]["reconstruction_manager"]
    
    # 打印所有 keys
    print("\nAll keys in checkpoint:")
    for key in state_dict.keys():
        print(key)
    
    # 打印层次化结构
    print_state_dict_keys(state_dict)