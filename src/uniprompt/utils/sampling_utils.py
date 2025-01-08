import random
from typing import List, Tuple


@staticmethod
def random_batch(data: List, batch_size: int) -> List:
    return random.sample(data, min(batch_size, len(data)))

@staticmethod
def sequential_batch(data: List, batch_size: int, offset: int = 0) -> List:
    return data[offset:offset + batch_size]

@staticmethod
def mini_batch_indices(batch_size: int, mini_batch_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + mini_batch_size, batch_size))
            for i in range(0, batch_size, mini_batch_size)]
