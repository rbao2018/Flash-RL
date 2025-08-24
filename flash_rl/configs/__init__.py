from .fp8 import FP8TensorConfig, FP8ChannelConfig, FP8vLLMConfig, FP8vLLMFastConfig
from .int8 import Int8Config, Int8PruneConfig, Int8FastConfig
from .bf16 import BF16Config

def get_default_config(fn):
    return {
        'fp8': FP8vLLMConfig(),
        'fp8_vllm': FP8vLLMConfig(),
        'fp8_vllm_fast': FP8vLLMFastConfig(),
        'fp8_fast': FP8vLLMFastConfig(),
        'fp8_channel': FP8ChannelConfig(),
        'fp8_tensor': FP8TensorConfig(),
        'int8': Int8Config(),
        'int8_fast': Int8FastConfig(),
        'int8_wo_prune': Int8Config(),
        'int8_prune': Int8PruneConfig(),
        'bf16': BF16Config(),
    }[fn]
