import random
import os
import numpy as np

from src.classes import *
from src.lib import *


def parse_instance_input(filename: str, scenario: str, instances_dir: str = "instances/") -> dict:
    """
    :param filename:        Name of the instance file, excluding `.txt` suffix
    :param instances_dir:   Path to the directory where the instance file is located
    :param scenario:        A scenario for setting patient priorities
    """
    # Ensure consistenly generated priorities
    random.seed(10)

    f = open(os.path.join(f"{instances_dir}", f"{filename}.txt"), 'r')

    # Read first line - parameters
    parameters_line = f.readline()
    num_surgeons, num_patients, num_ORs, num_days = list(map(int, parameters_line.split()))

    # Ignore second line - only comment
    f.readline()

    # Init Surgeons and Patients
    patients = np.ndarray(num_patients, dtype=Patient)
    surgeons = np.ndarray(num_surgeons, dtype=Surgeon)
    for i in range(num_surgeons):
        surgeons[i] = Surgeon(id=i)

    # Read Patients
    for i in range(num_patients):
        # Read from file
        line = f.readline()
        row = list(map(int, line.split()))
        if len(row) == 3:
            id_, surgeon_id, duration = row
            priority, leader_priority = None, None
        elif len(row) == 5:
            id_, surgeon_id, duration, priority, leader_priority = row
        else:
            raise AssertionError()
        # Apply scenario
        if scenario == "scenario_1":
            priority = 1
            leader_priority = 1
        elif scenario == "scenario_2":
            priority = priority if priority is not None else random.randint(1, 4)
            leader_priority = priority
        elif scenario == "scenario_3":
            priority = 1
            leader_priority = leader_priority if leader_priority is not None else random.randint(1, 4)
        elif scenario == "scenario_4":
            priority = priority if priority is not None else random.randint(1, 4)
            leader_priority = 1
        elif scenario == "scenario_5":
            priority = priority if priority is not None else random.randint(1, 4)
            leader_priority = leader_priority if leader_priority is not None else random.randint(1, 4)
        elif scenario == 'default':
            priority = priority if priority is not None else random.randint(1, 4)
            leader_priority = duration
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        # Init Patient
        patient = Patient(
            id=id_,
            duration=duration,
            priority=priority,
            leader_priority=leader_priority,
            surgeon_id=surgeon_id
        )
        patients[i] = patient
        surgeons[surgeon_id].patients.append(patient)

    # Ignore OR comment line
    f.readline()
    # Read OR lines
    lines = list()
    for i in range(num_days * num_ORs):
        line = f.readline()
        lines.append(line)

    # Ignore start times comment line
    f.readline()
    # Read start times
    line = f.readline()
    start_times = list(map(int, line.split()))

    # Ignore end times comment line
    f.readline()
    # Read end times
    line = f.readline()
    end_times = list(map(int, line.split()))

    # Init ORs: day OR start end
    ORs = np.ndarray(num_days, dtype=object)
    blocks = list()
    block_id = 0
    for line in lines[::num_ORs]:  # read every num_of_ORs-th line
        day, or_id, start, end = list(map(int, line.split()))
        OR_blocks = list()
        conflicts_list = [[] for _ in range(len(start_times))]
        for start_ in start_times:
            if start_ < start:
                continue
            for end_ in end_times:
                if start_ >= end_ or end_ > end:
                    continue
                b = Block(id=block_id, day=day, start=start_, end=end_, or_id=or_id)
                OR_blocks.append(b)
                for i in range(len(start_times)):
                    if start_ <= start_times[i] and end_ >= end_times[i]:
                        conflicts_list[i].append(b)
                block_id += 1
        blocks.extend(OR_blocks)
        conflicts = dict(zip(start_times, conflicts_list))
        ORs[day] = OR(start=start, end=end, start_times=start_times, end_times=end_times,
                      blocks=OR_blocks, conflicts=conflicts)
    blocks = np.array(blocks)

    return {
        "patients": patients,
        "surgeons": surgeons,
        "ORs": ORs,
        "blocks": blocks,
        "num_surgeons": num_surgeons,
        "num_patients": num_patients,
        "num_ORs": num_ORs,
        "num_days": num_days,
        "start_times": start_times,
        "end_times": end_times,
        "load": parse_instance_name(filename)["load"],
        "load_real": sum([patient.duration for patient in patients]) / sum([ORs[d].duration * num_ORs for d in range(num_days)])
    }


def parse_instance_name(instance_name: str) -> dict:
    chunks = instance_name.split("_")
    return {
        "num_days": 5,
        "num_ORs": int(chunks[2]) / 5,
        "num_surgeons": int(chunks[6]),
        "load": float(chunks[4])
    }


def parse_paramerters_str(paramerters_str: str) -> dict:
    """
    Decodes parameters in the form such as "s5_a1_b1_t1200_InH1_MuP1_LCR1_LC2".
    """

    parameters = paramerters_str.split("_")
    scenario: int = int(parameters[0][1:])
    alpha: float = float(parameters[1][1:])
    beta: float = float(parameters[2][1:])
    timeout: float = float(parameters[3][1:])
    try:    # BnP
        initial_heuristics: bool = bool(int(parameters[4][-1]))
        multiple_patterns_per_iteration: bool = bool(int(parameters[5][-1]))
        lc_remembering: bool = bool(int(parameters[6][-1]))
        idx = 7
    except (IndexError, ValueError):  # ILP
        initial_heuristics: bool = True
        multiple_patterns_per_iteration: bool = True
        lc_remembering: bool = True
        idx = 4
    lc_type = LazyConstraint.ALC if parameters[idx][-1] == "2" else LazyConstraint.OLC
    is_callback = False if (len(parameters) == idx + 2 and parameters[idx+1] == "nocallback") else True
    decentralized = True if (len(parameters) == idx + 2 and parameters[idx+1] == "decentralized") else False

    return {
        "scenario": f"scenario_{scenario}",
        "alpha": alpha,
        "beta": beta,
        "is_initial_heuristics": initial_heuristics,
        "multiple_patterns_per_iteration": multiple_patterns_per_iteration,
        "lc_remembering": lc_remembering,
        "LC_type": lc_type,
        "is_callback": is_callback,
        "decentralized": decentralized,
        "m": 0,
        "M": 1000,
        "branching_threshold": 0.5,
        "theta": 3,
        "epsilon": 0.001,
        "timeout": timeout,
        "max_depth": 40,
    }


def parse_input(parameters: str, instance: Optional[str] = None, alg_name: str = "BnP"):
    parameters_data = parse_paramerters_str(parameters)
    if instance is None:
        return parameters_data, None
    scenario = parameters_data["scenario"]
    instance_data = parse_instance_input(instance, scenario=scenario)
    return parameters_data, instance_data


def to_paramerters_str(
        scenario: int,
        alpha: float, beta: float, timeout,
        InH: bool, MuP: bool, LCR: bool,
        lc_type: LazyConstraint,
        alg_name: str = "BnP",
        is_callback: bool = True,
        decentralized: bool = False) -> str:
    """
    Encodes parameters into the standardizes string format
    """
    lc_type = 1 if lc_type == LazyConstraint.OLC else 2
    is_callback = "" if is_callback else "_nocallback"
    decentralized = "" if not decentralized else "_decentralized"
    if alg_name == "BnP":
        return f"s{scenario}_a{alpha}_b{beta}_t{timeout}_InH{int(InH)}_MuP{int(MuP)}_LCR{int(LCR)}_LC{lc_type}{is_callback}{decentralized}"
    else:
        return f"s{scenario}_a{alpha}_b{beta}_t{timeout}_LC{lc_type}{is_callback}{decentralized}"


if __name__ == "__main__":
    ...
