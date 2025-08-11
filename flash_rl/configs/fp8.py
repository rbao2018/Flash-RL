from dataclasses import dataclass, field
from typing import List, Any, Optional

@dataclass
class FP8TensorConfig:
    fn: str = 'fp8_tensor'
    load_format: str = 'dummy'
    distributed_executor_backend: str = 'external_launcher'
    module_attribute_to_preserve: List[str] = field(default_factory=lambda: ['workspace'])

@dataclass
class FP8ChannelConfig:
    fn: str = 'fp8_channel'
    load_format: str = 'dummy'
    distributed_executor_backend: str = 'external_launcher'
    module_attribute_to_preserve: List[str] = field(default_factory=lambda: ['workspace'])

@dataclass
class FP8vLLMConfig:
    fn: str = 'fp8_vllm'
    load_format: str = 'auto'
    distributed_executor_backend: str = 'external_launcher'
    module_attribute_to_preserve: List[str] = field(default_factory=lambda: ['workspace'])
    quantization: str = "fp8"
