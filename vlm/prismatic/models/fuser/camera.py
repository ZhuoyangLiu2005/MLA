import torch
from dataclasses import dataclass

@dataclass
class CameraParams:
    K: torch.Tensor 
    R: torch.Tensor  
    t: torch.Tensor  

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
    "metaworld_corner": CameraParams(
        K=torch.tensor([
            [270.39191899,   0.0,         112.0],
            [  0.0,         270.39191899, 112.0],
            [  0.0,          0.0,           1.0]
        ],dtype=torch.float32),
        R=torch.tensor([
            [-0.70710678,  0.19245009,  0.68041382],
            [ 0.70710678,  0.19245009,  0.68041382],
            [ 0.0,         0.96225045, -0.27216553]
        ], dtype=torch.float32),
        t=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
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

