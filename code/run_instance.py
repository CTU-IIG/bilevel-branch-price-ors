import os
import datetime
import logging
import argparse
from collections.abc import Callable
from fileinput import filename

from src.OR_scheduling import Problem, ProblemResult
from src.OR_scheduling_ILP import ILP_Problem
from src.OR_scheduling_BnP import BnP_Problem
from src.OR_scheduling_BnP_decentralized import BnP_Problem_decentralized
from run_testsuite import IgnoreFlag


def run_instance(out_dir: str, cls: Callable[...], instance: str, parameters: str, ignore_flag: IgnoreFlag, name=None, save=True) -> (Problem, ProblemResult):
    problem: Problem = cls(instance, parameters, name=name)
    name = problem.name

    filename = f"{instance}__{parameters}__{name}.json"
    filepath = os.path.join(out_dir, filename)
    if os.path.isfile(filepath):
        try:
            result = ProblemResult.load(out_dir, filename)
            if ignore_flag.ignore(result):
                print(f"\tIgnoring {name} instance ({ignore_flag}): {instance} ({parameters})")
                return problem, result
        except:
            pass

    problem.run()
    result = problem.get_result()
    print(f"\tRan in {result.overall_time} (optimality {result.proved_optimal}, obj {result.obj})")
    if save:
        print(f"\t\t→ {filepath}")
        problem.save(filepath, result)
    return problem, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("cls")
    parser.add_argument("instance")
    parser.add_argument("parameters")
    parser.add_argument("--name", default=None)
    parser.add_argument("--ignore_flag", default="IgnoreFlag.COMPUTED")
    args = parser.parse_args()

    run_instance(args.out_dir, eval(args.cls), args.instance, args.parameters, eval(args.ignore_flag), name=args.name)
