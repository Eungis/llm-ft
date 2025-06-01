# modality
Multimodal self learning w. lectures, papers, etc.


# trouble shooting

- problem: command `torch.cuda.is_available()` outputs UserWarning:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 10010). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
```

- solution: command `nvidia-smi` to check cuda information and install the appropriate torch version
ex. `pip install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html`