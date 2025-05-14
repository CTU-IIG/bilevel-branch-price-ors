# python built-ins
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Any

# external packages
import numpy as np
from matplotlib import pyplot as plt

from src.classes import *
from src.parsers import parse_instance_input, parse_paramerters_str, parse_input


class ProblemResult:
    def __init__(self):
        # Solution info
        self.proved_optimal = False
        self.obj: float = np.inf                        # F
        self.relax_obj: float = np.inf                  # F^{LPR}

        self.num_LCs: int = 0                           # LC
        self.num_callbacks: int = 0                     # CB
        self.num_CG_iterations: Optional[int] = None    # CGIter
        self.num_columns: Optional[int] = None          # Col
        self.num_nodes: Optional[int] = None            # nodes

        self.overall_time: float = 0.                   # TimeT
        self.callback_time: float = 0.                  # TimeCB
        self.subproblem_time: Optional[float] = None    # TimeSP
        self.mp_time: Optional[float] = None            # TimeMP
        self.inh_time: Optional[float] = None           # TimeInH
        self.ilp_time: Optional[float] = None           # TimeILP

        # Solution data
        self.surgeon_schedule: dict[Surgeon, list[Block]] = defaultdict(list)
        self.patient_schedule: dict[Patient, Block] = defaultdict()

        # Instance attributes
        self.instance: Optional[str] = None
        self.parameters: Optional[str] = None
        self.parameters_data: Optional[dict] = None
        self.patients: Optional[List[Patient]] = None
        self.capacity: Optional[int] = None
        self.num_days: Optional[int] = None
        self.num_ORs: Optional[int] = None
        self.num_surgeons: Optional[int] = None

    @property
    def gap(self):
        if self.obj and self.relax_obj:
            return (self.obj - self.relax_obj) / self.relax_obj

    @property
    def computed(self):
        return self.obj != np.inf

    @property
    def obj_computed(self):
        assert self.capacity
        a = self.parameters_data["alpha"] * (self.capacity - sum((patient.duration for patient in self.patient_schedule.keys())))
        b = self.parameters_data["beta"] * sum((patient.leader_priority * (1 - int(patient in self.patient_schedule.keys())) for patient in self.patients))
        return a + b

    @property
    def timeouted(self):
        return not self.proved_optimal

    @property
    def feasibility(self):
        return self.obj is not None and self.obj != np.inf

    @property
    def utilization(self) -> float:
        if not self.capacity:
            return 0
        utilized = sum((patient.duration for patient in self.patient_schedule.keys()))
        assert utilized <= self.capacity
        return utilized / self.capacity

    @property
    def leader_plan_priority(self) -> float:
        return sum((patient.leader_priority for patient in self.patient_schedule.keys()))

    @property
    def followers_plan_priority(self) -> float:
        return sum((patient.priority for patient in self.patient_schedule.keys()))

    @property
    def followers_objs(self):
        return tuple(sum(patient.priority for patient in surgeon.patients if patient in self.patient_schedule.keys()) for surgeon in self.surgeons)

    def _to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if not callable(v) and not k.startswith('__')}
        data['gap'] = self.gap
        return data

    def _from_dict(self, data: dict) -> Any:
        for k in self.__dict__.keys():
            try:
                setattr(self, k, data[k])
            except KeyError:
                continue
        return self

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self._to_dict(), f, indent=4)

    def visualize(self, filename: str = None) -> None:
        assert self.num_ORs == 1

        cmap = (plt.get_cmap("tab10"))(np.arange(self.num_surgeons))

        fig, ax = plt.subplots()
        for surgeon, blocks in self.surgeon_schedule.items():
            for k, block in enumerate(blocks):
                barh = ax.barh(
                    y=block.day*self.num_ORs+block.or_id,
                    width=block.duration,
                    left=block.start,
                    color=cmap[surgeon.id],
                    label=surgeon.id if k == 0 else None,
                )
                ax.bar_label(
                    barh,
                    labels=[f"S{surgeon.id}"],
                    label_type="center",
                )

        ax.grid(True, linestyle='dashed')
        plt.title(f"{self.instance}")
        plt.legend(loc="center left", bbox_to_anchor=(0.95, 0.5), ncol=1)
        plt.show()
        # plt.savefig(f'outputs/visualizations/{filename}.png', bbox_inches='tight')

    def _compute_obj(self):
        assert self.capacity
        utilized = sum((patient.duration for patient in self.patient_schedule.keys()))
        penalty = sum([patient.leader_priority for patient in self.patients if patient not in self.patient_schedule.keys()])
        obj = int(round(self.parameters_data["alpha"] * (self.capacity - utilized) + self.parameters_data["beta"] * penalty))
        return obj

    def validate(self):
        if self.overall_time - (self.mp_time + self.subproblem_time) > 10:
            raise AssertionError(f"T_all: {self.overall_time}, T_MP: {self.mp_time}, T_SP: {self.subproblem_time}")
        # obj = self._compute_obj()
        # if not obj == self.obj:
        #     raise AssertionError(f"{obj} != {self.obj}")

    @classmethod
    def load(cls, dirname: str, filename: str) -> Any:
        with open(os.path.join(dirname, filename), "rb") as f:
            data = json.load(f)
        result = cls()._from_dict(data)
        if "__" not in filename:
            return result
        instance, parameters, name = filename.split("__")
        if instance and parameters:
            result.parameters = parameters
            result.instance = instance
            result.parameters_data = parse_paramerters_str(parameters)
            scenario = result.parameters_data["scenario"]
            instance_data = parse_instance_input(instance, scenario)
            if result.overall_time > result.parameters_data["timeout"]: result.proved_optimal = False
            result.overall_time = min(result.parameters_data["timeout"] + 10e-3, result.overall_time)

            surgeons: dict = {surgeon.id: surgeon for surgeon in instance_data["surgeons"]}
            blocks: dict = {block.id: block for block in instance_data["blocks"]}
            patients: dict = {patient.id: patient for patient in instance_data["patients"]}
            result.surgeon_schedule = {
                surgeons[int(s)]: list(map(lambda id: blocks[id], schedule))
                for s, schedule in result.surgeon_schedule.items()
            }
            result.patient_schedule = {
                patients[int(patient)]: blocks[block]
                for patient, block in result.patient_schedule.items()
            }
            result.capacity = sum((
                instance_data["ORs"][d].duration * instance_data["num_ORs"]
                for d in range(instance_data["num_days"])
            ))
            result.num_days = instance_data["num_days"]
            result.num_ORs = instance_data["num_ORs"]
            result.num_surgeons = instance_data["num_surgeons"]
            result.patients = instance_data["patients"]
            if result.obj != result.obj_computed:
                raise AssertionError(f"{result.obj} != {result.obj_computed} ({instance}__{parameters})")
        return result

    def __str__(self):
        return f"{self.__class__.__name__}<{self._to_dict().__str__()}>"

    def __repr__(self):
        return f"{self.__class__.__name__}<proved_optimal: {self.proved_optimal}, obj: {self.obj}>"


class Problem(ABC):
    def __init__(self, instance: str, parameters: str, name: Optional[str] = None):
        self.instance: str = instance
        self.parameters: str = parameters
        self.name = name if name else self.__class__.__name__[:-8]

        # --- LOAD INPUT

        parameters_data, instance_data = parse_input(parameters, instance, alg_name=self.name)

        self.m: int = parameters_data["m"]
        self.alpha: int = parameters_data["alpha"]
        self.beta: int = parameters_data["beta"]
        self.M: int = parameters_data["M"]
        self.branching_threshold: float = parameters_data["branching_threshold"]
        self.theta: int = parameters_data["theta"]
        self.epsilon: float = parameters_data["epsilon"]
        self.is_callback: bool = parameters_data["is_callback"]
        self.is_initial_heuristics: bool = parameters_data["is_initial_heuristics"]
        self.multiple_patterns_per_iteration: bool = parameters_data["multiple_patterns_per_iteration"]
        self.lc_remembering: bool = parameters_data["lc_remembering"]
        self.timeout: bool = parameters_data["timeout"]
        self.max_depth: int = parameters_data["max_depth"]
        self.LC_type: str = parameters_data["LC_type"]
        self.scenario: str = parameters_data["scenario"]
        self.decentralized: str = parameters_data["decentralized"]

        self.surgeons: List[Surgeon] = instance_data["surgeons"]  # 1D array of Surgeon classes
        self.patients: List[Patient] = instance_data["patients"]  # 1D array of Patient classes
        self.ORs: np.ndarray[OR] = instance_data["ORs"]           # 2D array (shape=(num_days, num_ORs)) of OR classes
        self.blocks: List[Block] = instance_data["blocks"]        # 1D array of all OR blocks
        self.num_surgeons: int = instance_data["num_surgeons"]
        self.num_patients: int = instance_data["num_patients"]
        self.num_ORs: int = instance_data["num_ORs"]  # NOTE: this is number of parallel operating rooms
        self.num_days: int = instance_data["num_days"]
        self.start_times: List[int] = instance_data["start_times"]
        self.end_times: List[int] = instance_data["end_times"]

        # --- COMPUTE PARAMETERS

        self.capacity = sum((self.ORs[d].duration * self.num_ORs for d in range(self.num_days)))    # a.k.a. C - overall capacity

        self.block_durations: List[int] = sorted(list(set([block.duration for block in self.blocks])))
        self.L: int = len(self.block_durations)

        self.blocks_of_length: defaultdict = defaultdict(list)  # Blocks having length l a.k.a. B_{l}
        self.K: defaultdict = defaultdict(int)
        for block in self.blocks:
            self.K[block.duration] += 1
            self.blocks_of_length[block.duration].append(block)

        self.max_priority: int = max([patient.priority for patient in self.patients])

    def _get_overlapping_blocks(self, day: int, time: int):
        """
        :param day:
        :param time:
        :return: list of Blocks that overlap on the given day and time
        """
        return self.ORs[day].conflicts[time]

    def _get_day_blocks(self, day: int):
        """
        :param day:
        :return: list of Blocks, that are in the given day
        """
        return self.ORs[day].blocks

    # @abstractmethod
    # def save_timeline(self) -> None:
    #     ...

    # @abstractmethod
    # def print_solution(self): pass
    #
    # @abstractmethod
    # def save_and_visualize_graph(self):
    #     ...

    @abstractmethod
    def get_result(self) -> ProblemResult:
        ...

    @abstractmethod
    def run(self):
        ...

    def save(self, filepath: str, result: Optional[ProblemResult] = None):
        if result:
            result.save(filepath)


if __name__ == "__main__":
    filename = "51_ordays_5_load_1,05surgeons_10__s1_a1_b1_InH1_MuP1_LCR1_LC2__BnP"
    result = ProblemResult.load(f"./outputs/results_optim/", filename)
    print(result)
    result.visualize()
    # print("51_ordays_5_load_1,05surgeons_10__s1_a1_b1_InH1_MuP1_LCR1_LC2__BnP".split("__"))
