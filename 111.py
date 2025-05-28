import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version (used by PyTorch): {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Please check your installation.")

try:
    import torchvision
    print(f"Torchvision Version: {torchvision.__version__}")
except ImportError:
    print("Torchvision is not installed.")
try:
    import torchaudio
    print(f"Torchaudio Version: {torchaudio.__version__}")
except ImportError:
    print("Torchaudio is not installed.")
