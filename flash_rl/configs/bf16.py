from dataclasses import dataclass

@dataclass
class BF16Config:
    fn: str = 'bf16'
    load_format: str = 'dummy'
    distributed_executor_backend: str = 'external_launcher'
    