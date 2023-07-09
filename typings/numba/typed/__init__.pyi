from typing import Any
from typing import List as List
from typing import Dict as ODict

class Dict(ODict):
    @staticmethod
    def empty(t1: Any, t2: Any) -> ODict: ...
