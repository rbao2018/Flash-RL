from dataclasses import dataclass, field
from typing import List, Any, Optional

@dataclass
class Int8Config:
    fn: str = 'int8'
    load_format: str = 'auto'
    distributed_executor_backend: str = 'external_launcher'
    
@dataclass
class Int8PruneConfig:
    fn: str = 'int8_prune'
    load_format: str = 'auto'
    distributed_executor_backend: str = 'external_launcher'
