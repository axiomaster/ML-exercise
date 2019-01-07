# -*- coding: utf-8 -*-

import pynvml

pynvml.nvmlInit()

handler = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler) # type:c_nvmlMemory_t
print(meminfo.used)
