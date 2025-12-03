import subprocess
import sys
import shutil

def install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

# Function to check for NVIDIA GPU
def has_nvidia_gpu():
    return shutil.which("nvidia-smi") is not None

print("Checking system for NVIDIA GPU...")

# Install PyTorch based on GPU availability
if has_nvidia_gpu():
    print("NVIDIA GPU detected! Installing CUDA 12.6 PyTorch...")
    install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu126"
    )
    print("CUDA 12.6 PyTorch installed successfully!")
else:
    print("No NVIDIA GPU detected. Installing CPU-only PyTorch...")
    install("torch")
    print("CPU-only PyTorch installed successfully!")

print("Installing additional packages...")
install("numpy")
install("colorama")
print("\nDone! All dependencies installed successfully!")
input("Press Enter to exit...")
