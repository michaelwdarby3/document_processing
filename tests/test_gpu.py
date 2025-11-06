import ctypes
lib = ctypes.CDLL('libcuda.so.1')
res = lib.cuInit(0)
print('cuInit result:', res)
if res == 0:
    import torch
    print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Device 0:', torch.cuda.get_device_name(0))

