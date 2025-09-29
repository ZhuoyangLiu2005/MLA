import torch
from dataclasses import dataclass
from .contrastive import project_3d_to_2d_672_rlbench, project_3d_to_2d_672_franka_right, project_3d_to_2d_672_franka_front

@dataclass
class CameraParams:
    K: torch.Tensor 
    R: torch.Tensor  
    t: torch.Tensor  

# used camera parameters
CAMERA_CONFIGS = {
    "rlbench_front": CameraParams(
        K=torch.tensor([
            [-307.7174807,    0.0,         112.0],
            [   0.0,        -307.7174807,  112.0],
            [   0.0,           0.0,          1.0]
        ], dtype=torch.float32),
        R=torch.tensor([
            [ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01],
            [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07],
            [-5.66244125e-07,  9.06307936e-01, -4.22617912e-01]
        ], dtype=torch.float32),
        t=torch.tensor([1.34999919e+00, 3.71546562e-08, 1.57999933e+00], dtype=torch.float32)
    ),
    "franka_right": CameraParams(
        K=torch.tensor([
            [387.414794921875, 0.0, 319.47052001953125],   
            [0.0, 386.8714904785156, 241.13287353515625],  
            [0.0, 0.0, 1.0]                             
        ],dtype=torch.float32),
        R=torch.tensor([
            [ 0.91300858,  0.26157042, -0.31304353],
            [ 0.39730357, -0.7442472,   0.53688545],
            [-0.09254842, -0.61455433, -0.78342694]
        ], dtype=torch.float32),
        t=torch.tensor([0.8591219242556176, -0.5851783639922448, 0.7535876808722389], dtype=torch.float32)
    ),
    "franka_front": CameraParams(
        K=torch.tensor([
            [388.2638244628906, 0.0, 328.3757019042969],
            [0.0, 387.84130859375, 240.24295043945312],
            [0.0, 0.0, 1.0]                         
        ],dtype=torch.float32),
        R=torch.tensor([
            [-0.01750229,  0.95018522, -0.31119403],
            [ 0.99984609,  0.01625676, -0.00659609],
            [-0.0012085,  -0.31126158, -0.95032351],
        ], dtype=torch.float32),
        t=torch.tensor([0.8545415959817313, 0.5748472977587156, 1.0411478820663598], dtype=torch.float32)
    ),
}

def get_camera_params(config_name="default", device=None):
    if config_name not in CAMERA_CONFIGS:
        raise ValueError(f"Unknown camera config: {config_name}. Available configs: {list(CAMERA_CONFIGS.keys())}")
    
    params = CAMERA_CONFIGS[config_name]
    if device is not None:
        params.K = params.K.to(device)
        params.R = params.R.to(device)
        params.t = params.t.to(device)
    
    return params


PROJECT_FUNCS = {
    "rlbench_front": project_3d_to_2d_672_rlbench,
    "franka_right": project_3d_to_2d_672_franka_right,
    "franka_front": project_3d_to_2d_672_franka_front,
}

def get_projection_func(camera_name: str):
    if camera_name not in PROJECT_FUNCS:
        raise ValueError(
            f"Unknown projection func for camera {camera_name}. "
            f"Available: {list(PROJECT_FUNCS.keys())}"
        )
    return PROJECT_FUNCS[camera_name]
