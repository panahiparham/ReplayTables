from typing import Any
from typing import Dict as ODict

List = list

class Dict(ODict):
    @staticmethod
    def empty(t1: Any, t2: Any) -> ODict: ...
