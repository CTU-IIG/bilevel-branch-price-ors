import datetime
import logging
import os.path
import subprocess
import sys
from enum import Enum, auto
from typing import Optional

from src.OR_scheduling import ProblemResult
from src.parsers import parse_paramerters_str, parse_instance_input

OUT_DIR: str = ...
IN_DIR: str = ...


class IgnoreFlag(Enum):
    COMPUTED = auto()       # ignore computed instances
    TIMEOUT = auto()        # ignore timeouted instances
    NON_TIMEOUT = auto()    # ignore non-timeouted instances
    NONE = auto()           # compute all

    def ignore(self, result: ProblemResult) -> bool:
        if self == self.NONE:
            return False
        elif self == self.COMPUTED:
            return result.computed
        elif self == self.TIMEOUT:
            return result.proved_optimal
        elif self == self.NON_TIMEOUT:
            return result.proved_optimal
        else:
            raise ValueError(f"Flag {self} not recognised.")


def run_batch(parameters: str, cls: str, name: str = "",
              num_days: Optional[int] = None, num_ORs: Optional[int] = None, num_surgeons: Optional[int] = None, num_patients: Optional[int] = None, load: Optional[float] = None,
              ignore_flag=IgnoreFlag.COMPUTED):
    logging.info(f"BATCH {parameters} {cls} num_days={num_days}, num_ORs={num_ORs}, num_surgeons={num_surgeons}, num_patients={num_patients}")
    scenario = parse_paramerters_str(parameters)["scenario"]
    dirname = IN_DIR

    for filename in sorted(filter(lambda s: s.endswith("txt"), os.listdir(dirname))):
        instance = parse_instance_input(filename[:-4], scenario)
        if ((num_days is None or instance["num_days"] == num_days)
                and (num_ORs is None or instance["num_ORs"] == num_ORs)
                and (num_surgeons is None or instance["num_surgeons"] == num_surgeons)
                and (num_patients is None or instance["num_patients"] == num_patients)
                and (load is None or instance["load"] == load)
        ):
            logging.info(f"{datetime.datetime.now().strftime('%d.%m. %H:%M:%S')}: {filename[:-4]} {parameters}")
            result = subprocess.run([
                sys.executable,
                "-m",
                "run_instance",
                OUT_DIR,
                cls,
                filename[:-4],
                parameters,
                *["--ignore_flag", str(ignore_flag)],
                *["--name", name]
            ], cwd=os.getcwd(), capture_output=True, encoding="UTF-8")
            logging.info(result.stdout)
            if result.returncode != 0:
                logging.info(result.stderr)


if __name__ == "__main__":
    for s, ors, load in ((10, 1, 2.0), (14, 1, 2.0), (12, 1, 1.5), (12, 1, 2.5), (20, 2, 2.0), (20, 4, 2.0)):
        run_batch("s5_a1_b1_t1200_InH1_MuP1_LCR1_LC2", "BnP_Problem", num_surgeons=s, num_ORs=ors, load=load)
        run_batch("s5_a1_b1_t1200_LC2", "ILP_Problem", num_surgeons=s, num_ORs=ors, load=load)
