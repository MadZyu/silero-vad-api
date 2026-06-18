"""RNNoise Python wrapper - 基于 ctypes 调用 librnnoise.so"""
import ctypes
import numpy
import os

# 查找 librnnoise.so:
# 1. 环境变量 RNNOISE_LIB 指定的路径
# 2. 工程目录下的 lib/librnnoise.so
# 3. /tmp/rnnoise/.libs/librnnoise.so (旧路径兼容)
_SEARCH_PATHS = [
    os.environ.get("RNNOISE_LIB", ""),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "librnnoise.so"),
    "/tmp/rnnoise/.libs/librnnoise.so",
]

lib = None
for _path in _SEARCH_PATHS:
    if _path and os.path.exists(_path):
        lib = ctypes.cdll.LoadLibrary(_path)
        break

if lib is None:
    raise FileNotFoundError(
        "找不到 librnnoise.so，请设置 RNNOISE_LIB 环境变量或将库放到 lib/ 目录"
    )

lib.rnnoise_process_frame.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.rnnoise_process_frame.restype = ctypes.c_float
lib.rnnoise_create.argtypes = [ctypes.c_void_p]
lib.rnnoise_create.restype = ctypes.c_void_p
lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]


class RNNoise(object):
    def __init__(self):
        self.obj = lib.rnnoise_create(None)

    def process_frame(self, inbuf):
        outbuf = numpy.ndarray((480,), "h", inbuf).astype(ctypes.c_float)
        outbuf_ptr = outbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        VadProb = lib.rnnoise_process_frame(self.obj, outbuf_ptr, outbuf_ptr)
        return (VadProb, outbuf.astype(ctypes.c_short).tobytes())

    def destroy(self):
        lib.rnnoise_destroy(self.obj)