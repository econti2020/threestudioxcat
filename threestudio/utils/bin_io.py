import numpy as np

def load_bin_voxel(filepath, shape=(128, 128, 128)):
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(shape)

def save_bin_voxel(filepath, voxel_tensor):
    voxel_np = voxel_tensor.cpu().numpy().astype(np.float32)
    voxel_np.tofile(filepath)
