# from typing import Callable
from typing import Callable, overload, TypeVar

T = TypeVar('T')

@overload
def njit(cache: bool) -> Callable[[T], T]: ...
@overload
def njit(f: T) -> T: ...

def jit(cache: bool, forceobj: bool) -> Callable[[T], T]: ...
