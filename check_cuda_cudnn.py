import torch

print("==========================================")
if torch.cuda.is_available():
    print("CUDA is available.") 
    print("Device count:", torch.cuda.device_count())
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDNN enabled:", torch.backends.cudnn.enabled)
    print("CUDNN version:", torch.backends.cudnn.version())
else:
    print("CUDA is not available.")
print("==========================================")
