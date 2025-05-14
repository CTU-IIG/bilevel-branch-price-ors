from enum import Enum
from dataclasses import dataclass, field
from typing import List


@dataclass
class Patient:
    id: int
    duration: int
    priority: int
    leader_priority: int
    surgeon_id: int

    def __hash__(self):
        return hash(self.id)


@dataclass
class Surgeon:
    id: int
    patients: List[Patient] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)


@dataclass
class Block:
    id: int
    day: int
    start: int
    end: int
    or_id: int

    def duration(self):
        return self.end - self.start

    def __hash__(self):
        return hash(self.id)

    def __post_init__(self):
        self.duration = self.end - self.start


@dataclass
class OR:
    start: int
    end: int
    start_times: List[int]  # possible start times
    end_times: List[int]  # possible end times
    blocks: List[Block]  # OR blocks
    conflicts: dict

    def __post_init__(self):
        self.duration = self.end - self.start


class LazyConstraint(str, Enum):
    OLC = "objective-based"
    ALC = "assignment-based"


if __name__ == "__main__":
    ...
