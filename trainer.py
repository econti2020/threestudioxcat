# insert in Trainer.__init__ or on_fit_start:
if hasattr(self.cfg, "input_bin") and self.cfg.input_bin is not None:
    from threestudio.utils.bin_io import load_bin_voxel
    voxel_np = load_bin_voxel(self.cfg.input_bin)
    import torch
    voxel_tensor = torch.from_numpy(voxel_np).unsqueeze(0).cuda() if torch.cuda.is_available() else torch.from_numpy(voxel_np).unsqueeze(0)
    # self.model.initialize_voxel(voxel_tensor)

# and on_fit_end:
if hasattr(self.cfg, "output_bin") and self.cfg.output_bin is not None:
    from threestudio.utils.bin_io import save_bin_voxel
    # output_voxel_tensor = self.model.get_voxel()  # Retrieve your tensor here
    output_voxel_tensor = voxel_tensor  # For test pipeline
    save_bin_voxel(self.cfg.output_bin, output_voxel_tensor)
