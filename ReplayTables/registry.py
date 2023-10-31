import numpy as np
from typing import Any, Dict
from ReplayTables.ReplayBuffer import ReplayBuffer

from ReplayTables.BackwardsReplay import BackwardsReplay, BackwardsReplayConfig
from ReplayTables.PER import PrioritizedReplay, PERConfig
from ReplayTables.PSER import PrioritizedSequenceReplay, PSERConfig

def build_buffer(buffer_type: str, max_size: int, lag: int, rng: np.random.Generator, config: Dict[str, Any]) -> ReplayBuffer:
    if buffer_type == 'uniform' or buffer_type == 'standard':
        return ReplayBuffer(max_size, lag, rng)

    elif buffer_type == 'backwards':
        c = BackwardsReplayConfig(**config)
        return BackwardsReplay(max_size, lag, rng, c)

    elif buffer_type == 'PER':
        c = PERConfig(**config)
        return PrioritizedReplay(max_size, lag, rng, c)

    elif buffer_type == 'PSER':
        c = PSERConfig(**config)
        return PrioritizedSequenceReplay(max_size, lag, rng, c)

    raise Exception('Unable to determine type of buffer')
