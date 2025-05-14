# python built-ins
import copy
import logging
import random
import sys
import time
from collections import defaultdict

# external packages
import gurobipy as gp
from gurobipy import GRB, tuplelist
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from src.OR_scheduling import ProblemResult, Problem
from src.OR_scheduling_ILP import ILP_Problem
from src.classes import *
from src.graph import *
from src.lib import *


env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


class BnP_Problem_decentralized(Problem):
    def __init__(self, instance: str, parameters: str, name: Optional[str] = None):
        """
        parse input file and get output variables/parameters
        :param instance:
        :param parameters:
        """

        super().__init__(instance, parameters, name)
        assert self.decentralized
        self.is_callback = False
        self.logs = []

        # a_{s,k,b} \in {0, 1} a_{s,k,b} = 1 iff block is assigned to surgeon s in schedule k
        # initialize (dummy way) - for each surgeon assign only one schedule containing no blocks (a_{s,k,b} = 0)
        self.a: List[List[np.ndarray]] = [[] for _ in range(self.num_surgeons)]     # surgeon block assignment
        self.g: List[List[np.ndarray]] = [[] for _ in range(self.num_surgeons)]     # patient block assignment
        for s in range(self.num_surgeons):
            self.a[s].append(np.zeros(len(self.blocks), dtype=int))
            self.g[s].append(np.zeros(shape=(self.num_patients, len(self.blocks)), dtype=int))

        # delta_{s,k} and w_{s, k}
        self.delta: List[List[int]] = [[] for _ in range(self.num_surgeons)]
        self.w: List[List[int]] = [[] for _ in range(self.num_surgeons)]
        self.omega: List[List[int]] = [[] for _ in range(self.num_surgeons)]
        for s in range(self.num_surgeons):
            self.delta[s].append(0)
            self.w[s].append(sum([patient.leader_priority for patient in self.surgeons[s].patients]))
            self.omega[s].append(0)

        self.best_integral_solution: float = np.inf
        self.leaders_solution: float = np.inf
        self.best_blocks: dict = dict()
        self.best_patients_in_block: dict = dict()

        self.start_time = None
        self.stop_time = None
        self.overall_time: float = 0.
        self.is_optimal: bool = True
        self.mp_root_relaxation: float = 0.
        self.num_columns: int = 0
        self.num_CG_iterations: int = 0
        self.mp_time: float = 0.
        self.subproblem_time: float = 0.
        self.initial_heuristics_time = 0.
        self.ilp_time = 0.
        self.callback_time: float = 0.
        self.num_callbacks: int = 0
        self.num_LCs: int = 0
        self.num_nodes: int = 0
        self.global_LB: float = 0.

        # graph
        self.graph: Graph = Graph(name=f"{self.instance}_{self.parameters}")

        if self.lc_remembering:
            self.LCs: List[dict[str, tuple]] = [{} for _ in range(self.num_surgeons)]

        if self.is_initial_heuristics:
            self.initial_heuristics()

    def initial_heuristics(self):
        """
        initial heuristics based on paper
        :return:
        """
        start_time = time.time()

        # STEP 0: Initialization
        Phi: List[tuple] = []  # current block schedule
        P: List[List[Patient]] = [surgeon.patients for surgeon in self.surgeons]  # denotes P´, initialize P´[s] to P[s]
        while True:
            # STEP 1: Compose best block composition for every block duration
            scheduled_patients: List[dict] = [dict() for _ in range(self.num_surgeons)]  # scheduled patients for surgeon s and block durations B_d
            w: List[dict] = [dict() for _ in range(self.num_surgeons)]      # w_{s, B_d}
            delta: List[dict] = [dict() for _ in range(self.num_surgeons)]  # delta_{s, B_d}
            for s in range(self.num_surgeons):
                for block_duration in self.block_durations:
                    w[s][block_duration] = 0
                    delta[s][block_duration] = 0

            for s in range(self.num_surgeons):
                if not P[s]: continue
                for block_duration in self.block_durations:
                    # SOLVE KNAPSACK
                    model_knapsack, utilization_function, idle_time = self._get_knapsack_model(s, P, block_duration)
                    model_knapsack.optimize()

                    # store scheduled_patients, w and delta
                    scheduled_patients[s][block_duration] = []
                    for patient in P[s]:
                        if round(model_knapsack.getVarByName(f'x[{patient.id}]').x) == 1:
                            scheduled_patients[s][block_duration].append(patient)
                    delta[s][block_duration] = idle_time.getValue()
                    w[s][block_duration] = utilization_function.getValue()  # actually it's w´

            # STEP 2: Schedule one additional block per surgeon to block schedule
            model_leader = self.get_leader_model(Phi, P, delta, w)
            model_leader.optimize()

            if model_leader.Status == GRB.INFEASIBLE:
                self.theta = self.theta // 2
                if self.theta: continue
                else: break

            # STEP 3: Update data
            # Update current block schedule Phi with newly assigned blocks
            for s in range(self.num_surgeons):
                scheduled_patients_surgeon: List = []
                for block in self.blocks:
                    if round(model_leader.getVarByName(f'y[{s},{block.id}]').x) == 1 and (s, block.id) not in Phi:
                        Phi.append((s, block.id))
                        scheduled_patients_surgeon.extend(scheduled_patients[s][block.duration])
                P[s] = [patient for patient in P[s] if patient not in scheduled_patients_surgeon]
        # end while

        # STEP 4: Find bilevel feasible patient schedule based on selected blocks
        x_final: np.ndarray = np.zeros(shape=(self.num_patients, len(self.blocks)))
        for s in range(self.num_surgeons):
            # Solve follower problem given currently assigned blocks y_{s,b} resulting from the current block schedule

            # --- MODEL ---
            surgeon_blocks: List[Block] = [block for block in self.blocks if (s, block.id) in Phi]

            model_surgeon = self._get_surgeon_model(s, Phi, surgeon_blocks)
            model_surgeon.optimize()

            patients_assigned = []

            for patient in self.surgeons[s].patients:
                for block in surgeon_blocks:
                    x_final[patient.id, block.id] = round(model_surgeon.getVarByName(f'x[{patient.id},{block.id}]').x)
                    if x_final[patient.id, block.id] == 1:
                        patients_assigned.append(patient)

            patients_not_assigned = [patient for patient in self.surgeons[s].patients if not patient in patients_assigned]

            if self.lc_remembering:
                num_blocks_of_length: defaultdict = defaultdict(int, {l: 0 for l in self.block_durations})  # a.k.a. n_{s,l}^c
                for block in surgeon_blocks:
                    num_blocks_of_length[block.duration] += 1

                if self.LC_type == LazyConstraint.OLC:
                    lc = (num_blocks_of_length, round(model_surgeon.objVal))
                elif self.LC_type == LazyConstraint.ALC:
                    lc = (patients_assigned, patients_not_assigned, num_blocks_of_length)

                self.LCs[s][str(num_blocks_of_length)] = lc

        # STEP 5: Evaluation
        patient_schedule_value = self.alpha * (self.capacity - sum(patient.duration * x_final[patient.id, block.id]
                                                                   for patient in self.patients
                                                                   for block in self.blocks)
                                               ) + \
                                 self.beta * sum(patient.leader_priority * (1 - sum(x_final[patient.id, block.id]
                                                                                    for block in self.blocks))
                                                 for patient in self.patients)

        logger.debug(f"Initial heuristic's patient schedule value: {patient_schedule_value}")
        # set the evaluated patient schedule as best_integral_solution (so far) and create initial thetas from Phi
        self.best_integral_solution = round(patient_schedule_value)
        for s in range(self.num_surgeons):
            delta_surgeon = sum(x_final[patient.id, block.id] * patient.duration
                                for block in self.blocks
                                for patient in self.surgeons[s].patients)
            w_surgeon = sum(patient.leader_priority * (1 - sum(x_final[patient.id, block.id] for block in self.blocks))
                            for patient in self.surgeons[s].patients)

            self.a[s].append(np.array([int((s, block.id) in Phi) for block in self.blocks]))
            self.g[s].append(x_final)
            self.delta[s].append(delta_surgeon)
            self.w[s].append(w_surgeon)
            self.best_blocks[s] = [block for block in self.blocks if (s, block.id) in Phi]
            self.best_patients_in_block[s] = []
            for block in self.best_blocks[s]:
                assigned_patients = [patient for patient in self.surgeons[s].patients
                                     if x_final[patient.id, block.id] == 1]
                self.best_patients_in_block[s].append((block, assigned_patients))

        self.initial_heuristics_time += time.time() - start_time

    def fathom_node(self, model: gp.Model, a: List[List[np.ndarray]], g: List[List[np.ndarray]]):
        """
        node is fathomed if:
        1) objVal of linear relaxation is worse than the best integral solution so far
        2) the solution is integral
        :param a:
        :param g:
        :param model:
        :return:
        """
        if model.objVal + 1 - self.epsilon > self.best_integral_solution:
            logger.debug("Linear relaxation is worse than the best integral solution so far")
            return True

        if self._is_integral_solution(model=model):
            logger.debug("Solution is integral")
            if model.objVal + self.epsilon < self.best_integral_solution:
                self._on_BnP_best_integral_solution(model, a, g)
            return True

        return False

    def master_problem(self, a: List[List[np.ndarray]], delta: List[List[int]], w: List[List[int]], omega: Optional[List[List[int]]], vtype: Any = GRB.CONTINUOUS, cutoff: float = None):
        """
        solves the master problem, gives every pattern its theta value
        :param a:
        :param delta:
        :param w:
        :param vtype:
        :param cutoff:
        :return:
        """
        start_time = time.time()

        # --- MODEL ---
        model_mp: gp.Model = gp.Model(env=env)
        model_mp.Params.NumericFocus = 3

        # --- VARIABLES ---
        thetas = model_mp.addVars(
            [(s, k) for s in range(self.num_surgeons) for k in range(len(a[s]))],
            vtype=vtype,
            name="theta",
            lb=0
        )

        # --- CONSTRAINTS ---
        # capacity of ORs
        for d in range(self.num_days):
            for t in self.start_times:
                overlapping_blocks: List[Block] = self._get_overlapping_blocks(day=d, time=t)  # a.k.a. O_{d,t}
                model_mp.addConstr(
                    gp.quicksum(
                        a[s][k][block.id] * thetas[s, k]
                        for s in range(self.num_surgeons)
                        for k in range(len(a[s]))
                        for block in overlapping_blocks
                    ) <= self.num_ORs,
                    name=f'lambda__{d}_{t}'
                )

        # at least one schedule k must be selected for surgeon s
        for s in range(self.num_surgeons):
            model_mp.addConstr(gp.quicksum(thetas[s, k] for k in range(len(a[s]))) == 1, name=f'mu__{s}')

        # --- SET OBJECTIVE ---
        try:
            model_mp.setObjective(
                -1 * gp.quicksum(
                    omega[s][k] * thetas[s, k]
                    for s in range(self.num_surgeons)
                    for k in range(len(a[s]))
                ), GRB.MINIMIZE
            )
        except IndexError as e:
            print(e)
            raise e
        model_mp._leaders_obj = self.alpha * (self.capacity - gp.quicksum(
            delta[s][k] * thetas[s, k]
            for s in range(self.num_surgeons)
            for k in range(len(a[s])))
        ) + self.beta * gp.quicksum(
            w[s][k] * thetas[s, k]
            for s in range(self.num_surgeons)
            for k in range(len(a[s]))
        )

        # --- OPTIMIZE ---
        model_mp.update()
        model_mp.optimize()

        self.mp_time += time.time() - start_time
        return model_mp

    def subproblem(self, model_mp: gp.Model, a: List[List[np.ndarray]], g: List[List[np.ndarray]], delta: List[List[int]], w: List[List[int]], omega: Optional[List[List[int]]], branching_history: List):
        """
        subproblem a.k.a. pricing of the patterns
        :param model_mp:
        :param a:
        :param g:
        :param delta:
        :param w:
        :param branching_history:
        :return:
        """
        start_time = time.time()

        # --- INIT ---

        columns_generated: bool = False

        # copy the patterns and auxiliary variables, if not -> problems in recursion
        a, g, delta, w, omega = self._deepcopy(a, g, delta, w, omega)

        # get lambdas from master problem
        lambdas: dict = {
            (d, t): model_mp.getConstrByName(f'lambda__{d}_{t}').Pi
            for d in range(self.num_days)
            for t in self.start_times
        }
        # get mus from master problem
        mus: dict = {s: model_mp.getConstrByName(f'mu__{s}').Pi for s in range(self.num_surgeons)}

        for s in range(self.num_surgeons):
            model_surgeon = self._get_subproblem_model(s, lambdas, branching_history)
            optimize(model=model_surgeon, is_callback=self.is_callback)

            # update num_callbacks, num_LCs, time_callback, LCs
            self.num_callbacks += model_surgeon._num_callbacks
            self.num_LCs += model_surgeon._num_LCs
            self.callback_time += model_surgeon._time_callback
            if self.lc_remembering:
                self.LCs[s] = {**self.LCs[s], **model_surgeon._LCs}

            # a pattern is added iff model_surgeon.objVal + mus[s] > 0
            if model_surgeon.Status == GRB.OPTIMAL and model_surgeon.objVal + mus[s] > self.epsilon:
                self.num_columns += 1
                columns_generated = True

                x_ = np.zeros(shape=(self.num_patients, len(self.blocks)))
                for patient in self.surgeons[s].patients:
                    for block in self.blocks:
                        if round(model_surgeon._x[patient.id, block.id].x) == 1:
                            x_[patient.id, block.id] = 1
                a[s].append(np.array([round(value.x) for value in model_surgeon._o.values()]))
                g[s].append(x_)
                delta[s].append(model_surgeon._delta_expression.x)
                w[s].append(model_surgeon._w_expression.x)
                omega[s].append(model_surgeon._omega_expression.x)
                self.a[s].append(np.array([round(value.x) for value in model_surgeon._o.values()]))
                self.g[s].append(x_)
                self.delta[s].append(model_surgeon._delta_expression.x)
                self.w[s].append(model_surgeon._w_expression.x)
                self.omega[s].append(model_surgeon._omega_expression.x)

                if not self.multiple_patterns_per_iteration:
                    self.subproblem_time += time.time() - start_time
                    return columns_generated, a, g, delta, w, omega

        self.subproblem_time += time.time() - start_time
        return columns_generated, a, g, delta, w, omega

    def column_generation(self, a: List[List[np.ndarray]], g: List[List[np.ndarray]], delta: List[List[int]], w: List[List[int]], omega: Optional[List[List[int]]], branching_history: List):
        """
        iterate between master problem and subproblem until no new columns are generated
        :param a:
        :param g:
        :param delta:
        :param w:
        :param branching_history:
        :return:
        """
        columns_generated: bool = True
        num_columns_before: int = self.num_columns
        num_CG_iterations_before: int = self.num_CG_iterations

        # column generation - uses continuous variable theta
        while columns_generated:
            self.num_CG_iterations += 1

            model_mp = self.master_problem(a=a, delta=delta, w=w, omega=omega, vtype=GRB.CONTINUOUS)

            if self.subproblem_time + self.mp_time > self.timeout or time.time() >= self.stop_time:
                return self._on_time_over()

            columns_generated, a, g, delta, w, omega = self.subproblem(
                model_mp=model_mp,
                a=a, g=g,
                delta=delta, w=w, omega=omega,
                branching_history=branching_history
            )

        num_columns_added: int = self.num_columns - num_columns_before
        num_CG_iterations: int = self.num_CG_iterations - num_CG_iterations_before

        if not branching_history: self.mp_root_relaxation = model_mp.objVal

        return model_mp, a, g, delta, w, omega, num_columns_added, num_CG_iterations

    def branch(self, a: List[List[np.ndarray]], g: List[List[np.ndarray]], delta: List[List[int]], w: List[List[int]], omega: Optional[List[List[int]]], branching_history: List, depth: int, is_rightmost: bool, bound: float = -np.inf):
        """
        Do column generation. After you finish, check if you can fathom the node.
        Then find branching y_{s,b} and branch. When you branch over y_{s,b} = 1 (positive branching),
        initial dummy pattern gets deleted and you have to add a pattern according to a branching history.
        This is done because of infeasibility.
        :param depth:
        :param bound:
        :param is_rightmost:
        :param a:
        :param g:
        :param delta:
        :param w:
        :param branching_history:
        :return:
        """
        start = time.time()

        # --- INIT NODE ---

        current_node = self.num_nodes
        self.graph.add_node(id=current_node, depth=depth)
        self.num_nodes += 1

        # prune the node in case of timeout
        if time.time() - self.start_time > self.timeout:
            return self._on_time_over()

        # prune the node in case BIS <= its parent's relaxed objVal
        if self.best_integral_solution <= np.ceil(bound): return

        # copy the patterns and auxiliary variables, if not -> problems in recursion
        a, g, delta, w, omega = self._deepcopy(a, g, delta, w, omega)

        # --- COLUMN GENERATION & fathom ---

        # column generation
        column_generation_result = self.column_generation(
            a=a, g=g, delta=delta, w=w, omega=omega,
            branching_history=branching_history
        )
        if not column_generation_result:
            return
        model_mp, a, g, delta, w, omega, num_columns_added, num_CG_iterations = column_generation_result

        # if is_rightmost and model_mp.Status != GRB.INFEASIBLE:
        #     self.global_LB = min(np.ceil(round(model_mp.objVal, 2)), self.best_integral_solution)

        if self.fathom_node(model=model_mp, a=a, g=g):
            self.graph.update_node(id=current_node, updates={
               'objective_value': round(model_mp.objVal, 3),
               'incumbent_objective_value': round(self.best_integral_solution),
               'time': round(time.time() - start),
               'num_columns_added': num_columns_added,
               'num_CG_iterations': num_CG_iterations,
               'color': 'darkolivegreen1' if self._is_integral_solution(model=model_mp) else "red"
            })
            return

        # -- MP HEURISTICS & fathom ---
        model_heuristic = self.master_problem(a=self.a, delta=self.delta, w=self.w, omega=self.omega, vtype=GRB.INTEGER, cutoff=self.best_integral_solution - 1 + self.epsilon)
        if model_heuristic.Status == GRB.OPTIMAL and model_heuristic.objVal + self.epsilon < self.best_integral_solution:
            logger.debug(f'MP heuristic solution {model_heuristic.objVal}')
            self._on_BnP_best_integral_solution(model_heuristic, self.a, self.g)

        if self.fathom_node(model=model_mp, a=a, g=g):
            self.graph.update_node(
                id=current_node,
                updates={
                   'objective_value': round(model_mp.objVal, 3),
                   'incumbent_objective_value': round(self.best_integral_solution),
                   'time': round(time.time() - start),
                   'num_columns_added': num_columns_added,
                   'num_CG_iterations': num_CG_iterations,
                   'color': "lightskyblue1" if self._is_integral_solution(model=model_mp) else "red"
                })
            return

        # --- SOLVE AS ILP (exit) ---

        if depth > self.max_depth:
            logger.debug(f'Solving ILP at depth {depth}')
            self._run_as_ILP(branching_history, current_node, start)
            return

        # --- BRANCHING ---

        self.graph.update_node(id=current_node, updates={
           'objective_value': round(model_mp.objVal, 3),
           'incumbent_objective_value': round(self.best_integral_solution),
           'time': round(time.time() - start),
           'num_columns_added': num_columns_added,
           'num_CG_iterations': num_CG_iterations,
           'color': 'white'
        })

        branching_s, branching_b, branch_left, branch_right = self._get_most_fractional_branch(model_mp, a, g, delta, w, omega)

        # --- RIGHT
        # branch y_{s,b} = 1
        branching_y = {"s": branching_s, "b": branching_b, "value": 1}
        a[branching_s] = branch_right["a"]
        g[branching_s] = branch_right["g"]
        delta[branching_s] = branch_right["delta"]
        w[branching_s] = branch_right["w"]
        omega[branching_s] = branch_right["omega"]

        dummy_pattern = np.zeros(shape=len(self.blocks), dtype=int)
        for node in branching_history + [branching_y]:
            if node['s'] == branching_s:
                dummy_pattern[node['b']] = node['value']

        add_dummy_pattern: bool = True
        for pattern in a[branching_s]:
            if np.array_equal(pattern, dummy_pattern):
                add_dummy_pattern = False
                break
        if add_dummy_pattern:
            a, g, delta, w, omega = self._add_dummy_pattern(dummy_pattern, branching_s, a, g, delta, w, omega)

        self.graph.add_edge(start=current_node, end=self.num_nodes, s=branching_s, b=branching_b, value=1)
        self.branch(
            a=a, g=g, delta=delta, w=w, omega=omega,
            branching_history=branching_history + [branching_y],
            depth=depth + 1,
            bound=round(model_mp.objVal, 2),
            is_rightmost=is_rightmost
        )

        if time.time() - self.start_time > self.timeout:
            return self._on_time_over()

        # --- LEFT
        # branch y_{s,b} = 0
        branching_y["value"] = 1
        a[branching_s] = branch_left["a"]
        g[branching_s] = branch_left["g"]
        delta[branching_s] = branch_left["delta"]
        w[branching_s] = branch_left["w"]
        omega[branching_s] = branch_left["omega"]
        self.graph.add_edge(start=current_node, end=self.num_nodes, s=branching_s, b=branching_b, value=0)
        self.branch(
            a=a, g=g, delta=delta, w=w, omega=omega,
            branching_history=branching_history + [branching_y],
            depth=depth + 1,
            bound=round(model_mp.objVal, 2),
            is_rightmost=False
        )

        if time.time() - self.start_time > self.timeout:
            return self._on_time_over()

    def _get_subproblem_model(self, s: int, lambdas: dict, branching_history: List[dict]) -> gp.Model:
        # --- MODEL FOR SURGEON S ---
        model_surgeon: gp.Model = gp.Model(env=env)
        model_surgeon.Params.PreSolve = 0
        model_surgeon.Params.Heuristics = 0
        model_surgeon.Params.Cuts = 0
        model_surgeon.Params.NumericFocus = 3

        # --- VARIABLES ---
        o = model_surgeon.addVars(len(self.blocks), vtype=GRB.BINARY, name="o")
        x = model_surgeon.addVars(
            [(patient.id, block.id) for patient in self.surgeons[s].patients for block in self.blocks],
            vtype=GRB.BINARY,
            name='x'
        )

        if self.is_callback:
            q = model_surgeon.addVars(
                [(l, k) for l in self.K.keys() for k in range(self.num_days + 1)],
                vtype=GRB.BINARY, name='q'
            )

        delta_var: gp.Var() = model_surgeon.addVar(vtype=GRB.CONTINUOUS)
        w_var: gp.Var() = model_surgeon.addVar(vtype=GRB.CONTINUOUS)
        omega_var: gp.Var() = model_surgeon.addVar(vtype=GRB.CONTINUOUS)

        # --- CONSTRAINTS ---
        # maximum number of assigned blocks (eq 6.18)
        model_surgeon.addConstr(
            gp.quicksum(o[block.id] for block in self.blocks) >= self.m,
            name=f'constraint_max_number_of_assigned_blocks__{s}'
        )

        # according to branching_history, for surgeon s we should set o_{b} accordingly
        for block in self.blocks:
            for branching_y in branching_history:
                if branching_y['s'] == s and branching_y['b'] == block.id:
                    model_surgeon.addConstr(o[block.id] == branching_y['value'], name=f'constraint_branching_history__{branching_y["s"]}_{branching_y["b"]}')

        # a surgeon can be assigned at most one block a day (eq 6.19)
        for d in range(self.num_days):
            day_blocks: List[Block] = self._get_day_blocks(day=d)  # a.k.a. B_{d}
            model_surgeon.addConstr(gp.quicksum(o[block.id] for block in day_blocks) <= 1, name=f'constraint_2__{s}_{d}')

        # a patient can be scheduled only once
        for patient in self.surgeons[s].patients:
            model_surgeon.addConstr(
                gp.quicksum(x[patient.id, block.id] for block in self.blocks) <= 1,
                name=f'constraint_3__{s}_{patient.id}'
            )

        # capacity of the block b assigned to surgeon s cannot be exceeded
        for block in self.blocks:
            model_surgeon.addConstr(
                gp.quicksum(
                    patient.duration * x[patient.id, block.id] for patient in self.surgeons[s].patients
                ) <= block.duration * o[block.id],
                name=f'constraint_4__{s}_{block.id}'
            )

        # q constraints
        if self.is_callback:
            for l in self.K.keys():
                model_surgeon.addConstr(
                    gp.quicksum(k * q[l, k] for k in range(self.num_days + 1)) ==
                    gp.quicksum(o[block.id] for block in self.blocks_of_length[l]),
                    name=f'first_q_constraint__{l}'
                )
                model_surgeon.addConstr(
                    gp.quicksum(q[l, k] for k in range(self.num_days + 1)) == 1,
                    name=f'second_q_constraint__{l}'
                )

        # --- HELPER EXPRESSIONS

        delta_expression: gp.LinExpr = gp.quicksum(
            patient.duration * x[patient.id, block.id]
            for block in self.blocks
            for patient in self.surgeons[s].patients
        )
        w_expression: gp.LinExpr = gp.quicksum(
            patient.leader_priority * (1 - gp.quicksum(x[patient.id, block.id] for block in self.blocks))
            for patient in self.surgeons[s].patients
        )
        omega_expression: gp.LinExpr = gp.quicksum(
            patient.priority * x[patient.id, block.id]
            for block in self.blocks
            for patient in self.surgeons[s].patients
        )
        model_surgeon.addConstr(delta_var == delta_expression, name='delta_expression')
        model_surgeon.addConstr(w_var == w_expression, name='w_expression')
        model_surgeon.addConstr(omega_var == omega_expression, name='omega_expression')

        # --- REMEMBERED CONSTRAINTS

        if self.lc_remembering:
            if self.LC_type == LazyConstraint.OLC:
                M = len(self.surgeons[s].patients) * self.max_priority + 1
                for i, (num_blocks_of_length, right_side) in enumerate(self.LCs[s].values()):
                    left_side = gp.quicksum(
                        patient.priority * x[patient.id, block.id]
                        for patient in self.surgeons[s].patients
                        for block in self.blocks
                    ) + M * (
                                        self.L -
                                        gp.quicksum(q[l, num_blocks_of_length[l]] for l in num_blocks_of_length.keys())
                                )
                    model_surgeon.addConstr(left_side + self.epsilon >= round(right_side), name=f'remembered_LC_{s}_{i}')

            elif self.LC_type == LazyConstraint.ALC:
                M = len(self.surgeons[s].patients) + 1
                for i, (patients_assigned, patients_not_assigned, num_blocks_of_length) in enumerate(self.LCs[s].values()):
                    left_side = gp.quicksum(
                        x[patient.id, block.id]
                        for block in self.blocks
                        for patient in patients_assigned
                    ) - gp.quicksum(
                        x[patient.id, block.id]
                        for block in self.blocks
                        for patient in patients_not_assigned
                    ) + M * (
                                        self.L -
                                        gp.quicksum(q[l, num_blocks_of_length[l]] for l in num_blocks_of_length.keys())
                                )
                    right_side = len(patients_assigned)
                    constr = model_surgeon.addConstr(left_side + self.epsilon >= round(right_side), name=f'remembered_LC_{s}_{i}')
                    constr.Lazy = 1

        # --- SET OBJECTIVE ---

        model_surgeon.setObjective(
            gp.quicksum(
                lambdas[d, t] * o[block.id]
                for d in range(self.num_days)
                for t in self.start_times
                for block in self._get_overlapping_blocks(day=d, time=t)
            ) + omega_expression,
            GRB.MAXIMIZE
        )
        model_surgeon.Params.lazyConstraints = 1
        model_surgeon.Params.Presolve = 0
        model_surgeon.Params.Heuristics = 0
        model_surgeon.Params.Cuts = 0
        model_surgeon.Params.TimeLimit = max(10e-3, self.stop_time - time.time())

        if self.is_callback:
            model_surgeon._q = q
        model_surgeon._o = o
        model_surgeon._x = x
        model_surgeon._num_callbacks = 0
        model_surgeon._num_LCs = 0
        model_surgeon._num_columns = self.num_columns
        model_surgeon._time_callback = 0.
        model_surgeon._patients = self.patients
        model_surgeon._surgeons = self.surgeons
        model_surgeon._blocks = self.blocks
        model_surgeon._num_patients = self.num_patients
        model_surgeon._block_durations = self.block_durations
        model_surgeon._max_priority = self.max_priority
        model_surgeon._M = self.M
        model_surgeon._L = self.L
        model_surgeon._delta_expression = delta_var
        model_surgeon._w_expression = w_var
        model_surgeon._omega_expression = omega_var
        model_surgeon._alpha = self.alpha
        model_surgeon._beta = self.beta
        model_surgeon._s = s
        model_surgeon._LCs = {}
        model_surgeon._self_LCs = self.LCs[s] if self.lc_remembering else {}
        model_surgeon._epsilon = self.epsilon
        model_surgeon._LC_type = self.LC_type
        model_surgeon._timeout = max(1, self.stop_time - time.time())
        model_surgeon._start_time = self.start_time

        model_surgeon.update()
        return model_surgeon

    def _get_most_fractional_branch(self, model_mp: gp.Model, a: List[List[np.ndarray]], g: List[List[np.ndarray]], delta: List[List[int]], w: List[List[int]], omega: Optional[List[List[int]]]) -> [int, int, dict[str, List[Union[np.ndarray, int]]]]:
        # --- FIND MOST FRACTIONAL a AND b
        # y_{s,b} from which we should branch
        branching_s = 0
        branching_b = 0
        # lowest difference between y_{s,b} and branching threshold
        lowest_diff: Union[int, float] = np.inf
        for s in range(self.num_surgeons):
            a_s: np.ndarray = np.array(a[s]).T  # 2D matrix: rows = blocks, columns = patterns
            thetas_s: List = [model_mp.getVarByName(f'theta[{s},{k}]').x for k in range(a_s.shape[1])]
            weighted_a_s: np.ndarray = a_s * thetas_s
            y_s: np.ndarray = np.sum(weighted_a_s, axis=1)  # estimated y_{s,b}
            diff: np.ndarray = np.abs(y_s - self.branching_threshold)
            surgeon_lowest_diff, block = np.min(diff), np.argmin(diff)
            if surgeon_lowest_diff < lowest_diff:
                lowest_diff = surgeon_lowest_diff
                branching_s = s
                branching_b = block

        # --- COMPUTE a, g, delta, w FOR CHILDREN NODES
        # when block b is assigned to surgeon s, a.k.a. y_{s,b} = 1, then we have to discard all patterns
        # that have y_{s,b} = 0, in other words, discard all patterns k such that a_{s,k,b} = 0
        # and vice versa...
        new_a_for_left, new_a_for_right = [], []
        new_g_for_left, new_g_for_right = [], []
        new_delta_for_left, new_delta_for_right = [], []
        new_w_for_left, new_w_for_right = [], []
        new_omega_for_left, new_omega_for_right = [], []
        for pattern, assigned_patients, delta_, w_, omega_ in zip(a[branching_s], g[branching_s], delta[branching_s], w[branching_s], omega[branching_s]):
            if pattern[branching_b] == 0:
                new_a_for_left.append(pattern)
                new_g_for_left.append(assigned_patients)
                new_delta_for_left.append(delta_)
                new_w_for_left.append(w_)
                new_omega_for_left.append(omega_)
            else:
                new_a_for_right.append(pattern)
                new_g_for_right.append(assigned_patients)
                new_delta_for_right.append(delta_)
                new_w_for_right.append(w_)
                new_omega_for_right.append(omega_)

        return (
            branching_s,
            branching_b,
            {"a": new_a_for_left, "g": new_g_for_left, "delta": new_delta_for_left, "w": new_w_for_left, "omega": new_omega_for_left},
            {"a": new_a_for_right, "g": new_g_for_right, "delta": new_delta_for_right, "w": new_w_for_right, "omega": new_omega_for_right},
        )

    def _add_dummy_pattern(self, dummy_pattern: np.ndarray, branching_s: int, a: List[List[np.ndarray]], g: List[List[np.ndarray]], delta: List[List[int]], w: List[List[int]], omega: Optional[List[List[int]]]):
        # --- MODEL ---
        model_dummy: gp.Model = gp.Model(env=env)

        # --- VARIABLES ---
        x_ = tuplelist()
        for patient in self.surgeons[branching_s].patients:
            for block in self.blocks:
                x_.append((patient.id, block.id))
        x = model_dummy.addVars(x_, vtype=GRB.BINARY, name='x')

        # --- CONSTRAINTS ---
        # a patient can be scheduled only once
        for patient in self.surgeons[branching_s].patients:
            model_dummy.addConstr(
                gp.quicksum(x[patient.id, block.id] for block in self.blocks) <= 1,
                name=f'constraint_3__{patient.id}'
            )

        # capacity of the block b assigned to surgeon s cannot be exceeded
        for block in self.blocks:
            model_dummy.addConstr(
                gp.quicksum(
                    patient.duration * x[patient.id, block.id]
                    for patient in self.surgeons[branching_s].patients
                ) <= block.duration * dummy_pattern[block.id],
                name=f'constraint_4__{block.id}'
            )

        # --- SET OBJECTIVE ---
        delta_expression: gp.LinExpr = gp.quicksum(
            patient.duration * x[patient.id, block.id]
            for block in self.blocks
            for patient in self.surgeons[branching_s].patients
        )
        w_expression: gp.LinExpr = gp.quicksum(
            patient.leader_priority * (1 - gp.quicksum(x[patient.id, block.id] for block in self.blocks))
            for patient in self.surgeons[branching_s].patients
        )
        omega_expression: gp.LinExpr = gp.quicksum(
            patient.priority * x[patient.id, block.id]
            for block in self.blocks
            for patient in self.surgeons[branching_s].patients
        )

        model_dummy.setObjectiveN(
            -omega_expression,
            index=1,
            priority=0,
        )
        model_dummy.setObjectiveN(
            (self.alpha * delta_expression - self.beta * w_expression),
            index=0,
            priority=1,
        )
        model_dummy.ModelSense = GRB.MAXIMIZE

        # --- OPTIMIZE ---
        model_dummy.update()
        model_dummy.optimize()

        a[branching_s].append(dummy_pattern)
        x_ = np.zeros(shape=(self.num_patients, len(self.blocks)))
        for patient in self.surgeons[branching_s].patients:
            for block in self.blocks:
                if round(x[patient.id, block.id].x) == 1:
                    x_[patient.id, block.id] = 1
        g[branching_s].append(x_)
        delta[branching_s].append(delta_expression.getValue())
        w[branching_s].append(w_expression.getValue())
        omega[branching_s].append(omega_expression.getValue())

        return a, g, delta, w, omega

    def _get_knapsack_model(self, s: int, P: List[List[Patient]], block_duration) -> (gp.Model, gp.LinExpr, gp.LinExpr):
        # --- MODEL ---
        model_knapsack: gp.Model = gp.Model(env=env)

        # --- VARIABLES ---
        # x_{p} \in {0, 1} patient p is scheduled in current iteration for surgeon s and block duration B_d
        x_ = tuplelist()
        for patient in P[s]:
            x_.append(patient.id)
        x = model_knapsack.addVars(x_, vtype=GRB.BINARY, name='x')

        # --- CONSTRAINTS ---
        model_knapsack.addConstr(
            gp.quicksum(patient.duration * x[patient.id] for patient in P[s]) <= block_duration,
            name=f'constraint_capacity_{s}_{block_duration}'
        )

        # --- SET OBJECTIVE ---
        idle_time: gp.LinExpr = block_duration - gp.quicksum(patient.duration * x[patient.id] for patient in P[s])
        loss_function: gp.LinExpr = gp.quicksum(patient.leader_priority * (1 - x[patient.id]) for patient in P[s])
        utilization_function: gp.LinExpr = gp.quicksum(patient.leader_priority * x[patient.id] for patient in P[s])
        model_knapsack.setObjective(self.alpha * idle_time + self.beta * loss_function, GRB.MINIMIZE)
        model_knapsack.update()

        return model_knapsack, utilization_function, idle_time

    def get_leader_model(self, Phi: List[tuple], P: List[List[Patient]], delta: List[dict], w: List[dict]):
        # --- MODEL ---
        model_leader: gp.Model = gp.Model(env=env)

        # --- VARIABLES ---
        # y_{s,b} \in {0, 1} block b is assigned to surgeon s
        y = model_leader.addVars(self.num_surgeons, len(self.blocks), vtype=GRB.BINARY, name='y')

        # --- CONSTRAINTS ---
        # previously allocated blocks
        for s, b in Phi:
            model_leader.addConstr(y[s, b] == 1, name=f'constraint_Phi_{s}_{b}')

        # surgeon can be assigned to at most one block a day
        for s in range(self.num_surgeons):
            for d in range(self.num_days):
                day_blocks: List[Block] = self._get_day_blocks(day=d)  # a.k.a. B_{d}
                model_leader.addConstr(gp.quicksum(y[s, block.id] for block in day_blocks) <= 1,
                                       name=f'constraint_max_1_block_a_day_{s}_{d}')

        blocks_in_Phi = []
        for b in list(set([b for s, b in Phi])):
            blocks_in_Phi.append(self.blocks[b])
        blocks_not_in_Phi = [block for block in self.blocks if block not in blocks_in_Phi]
        assert len(self.blocks) == len(blocks_in_Phi) + len(blocks_not_in_Phi)

        # if no more patients for surgeon s, then don't assign any block
        for s in range(self.num_surgeons):
            if not P[s]:
                for block in blocks_not_in_Phi:
                    model_leader.addConstr(
                        y[s, block.id] == 0,
                       name=f'constraint_no_assignment_if_no_patients__{s}_{block.id}'
                    )

        # only a single block can be added per iteration for each surgeon
        for s in range(self.num_surgeons):
            model_leader.addConstr(
                gp.quicksum(y[s, block.id] for block in blocks_not_in_Phi) <= 1,
                name=f'constraint_single_block_per_iteration_{s}'
            )

        # controls the number of blocks that are scheduled in every iteration of the heuristic
        model_leader.addConstr(
            gp.quicksum(
                y[s, block.id]
                for s in range(self.num_surgeons)
                for block in blocks_not_in_Phi
            ) >= self.theta,
            name=f'constraint_max_scheduled_blocks_per_iter'
        )

        # capacity of ORs
        for d in range(self.num_days):
            for t in self.start_times:
                overlapping_blocks: List[Block] = self._get_overlapping_blocks(day=d, time=t)  # a.k.a. O_{d,t}
                model_leader.addConstr(
                    gp.quicksum(
                        y[s, block.id]
                        for s in range(self.num_surgeons)
                        for block in overlapping_blocks
                    ) <= self.num_ORs,
                    name=f'constraint_capacity_of_ORs__{d}_{t}'
                )

        # restricts the number of blocks that can be assigned to an individual surgeon
        for s in range(self.num_surgeons):
            model_leader.addConstr(gp.quicksum(y[s, block.id] for block in self.blocks) >= self.m, name=f'constraint_max_blocks_assigned_to_surgeon_{s}')

        # --- SET OBJECTIVE ---
        model_leader.setObjective(
            gp.quicksum(
                (self.alpha * delta[s][block.duration] - self.beta * w[s][block.duration] +
                 (block.day * 100 + block.start)) * y[s, block.id]
                for s in range(self.num_surgeons) for block in self.blocks
            ), GRB.MINIMIZE)
        model_leader.update()

        return model_leader

    def _get_surgeon_model(self, s: int, Phi: List[tuple], surgeon_blocks: List[Block]) -> gp.Model:
        # --- MODEL ---
        model_surgeon = gp.Model(env=env)

        # --- VARIABLES ---
        x = model_surgeon.addVars(
            [(patient.id, block.id) for patient in self.surgeons[s].patients for block in self.blocks],
            vtype=GRB.BINARY,
            name='x'
        )

        # --- CONSTRAINTS ---
        # a patient can be scheduled only once
        for patient in self.surgeons[s].patients:
            model_surgeon.addConstr(
                gp.quicksum(x[patient.id, block.id] for block in surgeon_blocks) <= 1,
                name=f'constraint_patient_scheduled_max_once_{patient.id}'
            )

        # capacity of the block b cannot be exceeded
        for block in surgeon_blocks:
            model_surgeon.addConstr(
                gp.quicksum(patient.duration * x[patient.id, block.id] for patient in self.surgeons[s].patients) <= block.duration,
                name=f'constraint_capacity_{s}_{block.id}'
            )

        # --- SET OBJECTIVE ---
        delta_expression: gp.LinExpr = gp.quicksum(
            patient.duration * x[patient.id, block.id]
            for block in surgeon_blocks
            for patient in self.surgeons[s].patients
        )
        w_expression: gp.LinExpr = gp.quicksum(
            patient.leader_priority * (1 - gp.quicksum(x[patient.id, block.id] for block in surgeon_blocks))
            for patient in self.surgeons[s].patients
        )
        fs_objective: gp.LinExpr = gp.quicksum(
            patient.priority * x[patient.id, block.id]
            for block in surgeon_blocks
            for patient in self.surgeons[s].patients
        )

        model_surgeon.setObjectiveN(
            fs_objective,
            index=0,
            priority=1,
        )
        model_surgeon.setObjectiveN(
            (self.alpha * delta_expression - self.beta * w_expression),
            1,
            priority=0,
        )
        model_surgeon.ModelSense = GRB.MAXIMIZE
        model_surgeon.update()

        return model_surgeon

    def _on_BnP_best_integral_solution(self, model: gp.Model, a: List[List[np.ndarray]], g: List[List[np.ndarray]]):
        self.best_integral_solution = round(model.objVal)
        self.leaders_solution = round(model._leaders_obj.getValue())
        for s in range(self.num_surgeons):
            for k in range(len(a[s])):
                if round(model.getVarByName(f'theta[{s},{k}]').x) == 1:
                    self.best_blocks[s] = [block for block in self.blocks if a[s][k][block.id] == 1]
                    self.best_patients_in_block[s] = []
                    for block in self.blocks:
                        if a[s][k][block.id] == 1:
                            assigned_block = block
                            assigned_patients = [
                                patient for patient in self.patients
                                if patient.id in list(np.where(g[s][k][:, block.id] == 1)[0])
                            ]
                            self.best_patients_in_block[s].append((assigned_block, assigned_patients))

    def _run_as_ILP(self, branching_history: List, current_node: int, start_time: float):
        ilp_start_time = time.time()

        ilp_problem: ILP_Problem = ILP_Problem(
            instance=self.instance,
            parameters=self.parameters,
            input_constraints={
                'y': branching_history,
                "LC": self.LCs if self.lc_remembering else []
            },
            cutoff=self.best_integral_solution - 1 + self.epsilon
        )
        ilp_problem.timeout = self.timeout - round(time.time() - self.start_time)
        ilp_problem.solve()
        if ilp_problem.model.Status == GRB.CUTOFF:
            self.graph.update_node(
                id=current_node,
                updates={
                   'incumbent_objective_value': round(self.best_integral_solution),
                   'time': round(time.time() - start_time),
                   'ILP_solved': True,
                   'color': 'sienna1'
                })
            return

        if ilp_problem.model.Status == GRB.TIME_LIMIT:
            self.is_optimal = False
            return

        if ilp_problem.return_objective_value() + self.epsilon < self.best_integral_solution:
            self.best_integral_solution = round(ilp_problem.return_objective_value())
            for s in range(self.num_surgeons):
                self.best_blocks[s] = [block for block in self.blocks if
                                       round(ilp_problem.model._y[s, block.id].x) == 1]
                self.best_patients_in_block[s] = []
                for block in self.best_blocks[s]:
                    assigned_patients = []
                    for patient in self.surgeons[s].patients:
                        if round(ilp_problem.model._x[patient.id, block.id].x) == 1:
                            assigned_patients.append(patient)
                    self.best_patients_in_block[s].append((block, assigned_patients))

        self.ilp_time = time.time() - ilp_start_time

        self.graph.update_node(id=current_node, updates={
            'objective_value': round(ilp_problem.return_objective_value()),
            'incumbent_objective_value': round(self.best_integral_solution),
            'time': round(time.time() - start_time),
            'ILP_solved': True,
            'color': 'yellow'
        })

    @staticmethod
    def _is_integral_solution(model: gp.Model):
        """
        returns True iff solution is integral (thetas)
        :param model:
        :return:
        """
        for var in model.getVars():
            if not round(var.x, 4).is_integer():
                return False

        return True

    @staticmethod
    def _deepcopy(a: List[List[np.ndarray]], g: List[List[np.ndarray]], delta: List[List[int]], w: List[List[int]], omega):
        a = copy.deepcopy(a)
        g = copy.deepcopy(g)
        delta = copy.deepcopy(delta)
        w = copy.deepcopy(w)
        omega = copy.deepcopy(omega)
        return a, g, delta, w, omega

    def _on_time_over(self) -> None:
        self.is_optimal = False
        return None

    def print_solution(self):
        """
        prints a solution regarding surgeons
        :return:
        """
        for s, blocks in self.best_blocks.items():
            for block in blocks:
                logger.info(f'Surgeon id={s} has a block id={block.id} on day {self.blocks[block.id].day}, '
                             f'starting at {self.blocks[block.id].start} and ending at {self.blocks[block.id].end}.')
        for s, tuples in self.best_patients_in_block.items():
            for block, patients in tuples:
                for patient in patients:
                    logger.info(
                        f'Patient id={patient.id}, duration={patient.duration}, priority={patient.priority} '
                        f'is scheduled to a block id={block.id} on day {block.day}, '
                        f'starting at {block.start} and ending at {block.end}')
        logger.info(f'Value of objective function: {self.best_integral_solution}')
        logger.info(f'MP relaxation: {self.mp_root_relaxation}')
        logger.info(f'Lower bound: {self.global_LB}')
        logger.info(f'#Nodes: {self.num_nodes}')
        logger.info(f'Overall time: {self.overall_time}')
        logger.info(f'Instance results: {self.return_instance_results()}')

    def timeline(self):
        """
        draws a timeline chart and saves it to a png file
        :return:
        """

        def s_bdt(block: Block, day: int, t: int):
            """
            returns true if block is in process in day d and time t
            :param block:
            :param day:
            :param t:
            :return:
            """
            return day == block.day and block.start <= t < block.end

        colors: dict = dict()
        patches: List[mpatches.Patch] = []
        for s in range(self.num_surgeons):
            colors[s] = (random.random(), random.random(), random.random())
            patches.append(mpatches.Patch(color=colors[s], label=f'Surgeon {s}'))

        fig, gnt = plt.subplots(figsize=(19.2, 10.8), dpi=100)
        gnt.grid(True, linestyle='dashed')

        day_margin: int = self.num_ORs * 2
        y_ticks_day: List[float] = [d * day_margin + self.num_ORs / 2 for d in range(self.num_days)]
        y_ticks_room: List[float] = [d * day_margin + r + 1 / 2 for d in range(self.num_days) for r in
                                     range(self.num_ORs)]
        labels_day: List[str] = [f'Day {d}' for d in range(self.num_days)]
        labels_room: List[str] = [f'Room {r}' for _ in range(self.num_days) for r in range(self.num_ORs)]

        sb_list: List[tuple] = [(s, block.id) for s, blocks in self.best_blocks.items() for block in blocks]

        # --- MODEL ---
        model_room_fitting: gp.Model = gp.Model(env=env)

        # --- VARIABLES ---
        x = model_room_fitting.addVars(((s, b, r) for r in range(self.num_ORs) for s, b in sb_list), vtype=GRB.BINARY, name='x')

        # --- CONSTRAINTS ---
        len_day = 32
        for d in range(self.num_days):
            day_blocks: List[Block] = self._get_day_blocks(day=d)  # a.k.a. B_{d}
            for r in range(self.num_ORs):
                model_room_fitting.addConstr(
                    gp.quicksum(x[s, b, r] * self.blocks[b].duration
                                for s, b in sb_list if self.blocks[b] in day_blocks
                    ) <= len_day,
                    name='dont_exceed'
                )

        for s, b in sb_list:
            model_room_fitting.addConstr(
                gp.quicksum(x[s, b, r] for r in range(self.num_ORs)) == 1,
                name='one_room_only'
            )

        # overlapping
        for d in range(self.num_days):
            for t in self.start_times:
                for r in range(self.num_ORs):
                    model_room_fitting.addConstr(
                        gp.quicksum(
                            x[s, b, r] * s_bdt(block=self.blocks[b], day=d, t=t) for s, b in sb_list
                        ) <= 1,
                        name=f'constraint_capacity_of_ORs__{d}_{t}'
                    )

        model_room_fitting.setObjective(0)
        model_room_fitting.optimize()

        sb_room_dict: dict = dict()
        for s, b in sb_list:
            for r in range(self.num_ORs):
                if round(x[s, b, r].x) == 1:
                    sb_room_dict[f'{s},{b}'] = r

        for s, blocks in self.best_blocks.items():
            for block in blocks:
                offset: int = self.blocks[block.id].start
                gnt.broken_barh(
                    [(offset, block.duration)],
                    (self.blocks[block.id].day * day_margin + sb_room_dict[f'{s},{block.id}'], 1),
                    facecolors=(colors[s]),
                    edgecolor='black'
                )
                gnt.text(offset + block.duration / 2,
                         self.blocks[block.id].day * day_margin + sb_room_dict[f'{s},{block.id}'] + 0.5,
                         f'S{s}, B{block.id}',
                         ha='center', va='center', size=10)

        gnt.set_xlabel('Time', fontsize=14)
        gnt.set_yticks(y_ticks_day, labels=labels_day)
        gnt.yaxis.set_ticks_position(position='left')
        gnt2 = gnt.twinx()
        gnt2.set_yticks(y_ticks_room, labels=labels_room)
        gnt2.yaxis.set_ticks_position(position='right')
        gnt2.set_ylim(gnt.get_ylim())
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.title('OR scheduling', fontsize=20)
        plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.savefig(f'outputs/graphical/schedules/{self.instance}.png', bbox_inches='tight')

    def return_objective_value(self):
        """
        returns the rounded objective function value
        :return:
        """
        return min(round(self.best_integral_solution, 3), 1000)

    def get_result(self) -> ProblemResult:
        result = ProblemResult()

        result.proved_optimal = self.is_optimal
        result.obj = self.leaders_solution
        if not self.decentralized:
            result.relax_obj = self.mp_root_relaxation
        else:
            result.relax_obj = self.best_integral_solution

        result.num_LCs = self.num_LCs
        result.num_callbacks = self.num_callbacks
        result.num_CG_iterations = self.num_CG_iterations
        result.num_columns = self.num_columns
        result.num_nodes = self.num_nodes

        result.callback_time = self.callback_time
        result.subproblem_time = self.subproblem_time
        result.initial_heuristics_time = self.initial_heuristics_time
        result.mp_time = self.mp_time
        result.ilp_time = self.ilp_time
        result.overall_time = self.overall_time

        for s, blocks in self.best_blocks.items():
            for block in blocks:
                result.surgeon_schedule[s].append(block.id)
        for _, tuples in self.best_patients_in_block.items():
            for block, patients in tuples:
                for patient in patients:
                    result.patient_schedule[patient.id] = block.id

        return result

    def run(self):
        self.start_time = time.time()
        self.stop_time = self.start_time + self.timeout
        self.branch(a=self.a, g=self.g, delta=self.delta, w=self.w, omega=self.omega, branching_history=[], depth=0, is_rightmost=True)
        self.overall_time = time.time() - self.start_time
        if not self.is_optimal:
            logger.info(f'Timeout was exceeded. BnP is not optimal.')

    def save(self, filepath: str, result: Optional[ProblemResult] = None):
        super().save(filepath, result)
        self.graph.save_and_render()


def optimize(model, is_callback=False):
    """
    wrapper function for optimizing a model, calls callback if set
    :param model:
    :param is_callback:
    :return:
    """
    if is_callback:
        model.optimize(callback)
    else:
        model.optimize()


def callback(model, where):
    if where != GRB.Callback.MIPSOL:
        return

    start_time = time.time()
    model._num_callbacks += 1
    o_c: gp.Var = model.cbGetSolution(model._o)
    x_c: gp.Var = model.cbGetSolution(model._x)
    s = model._s
    patients: List[Patient] = model._surgeons[s].patients
    blocks = [block for block in model._blocks if round(o_c[block.id]) == 1]

    num_blocks_of_length: defaultdict = defaultdict(int, {k: 0 for k in model._block_durations})  # a.k.a. n_{s,l}^c
    for block in blocks:
        num_blocks_of_length[block.duration] += 1

    if str(num_blocks_of_length) not in model._self_LCs.keys():

        # --- MODEL ---
        model_surgeon: gp.Model = gp.Model(env=env)

        # --- VARIABLES ---
        x_: tuplelist = tuplelist()
        for patient in patients:
            for block in blocks:
                x_.append((patient.id, block.id))
        x = model_surgeon.addVars(x_, vtype=GRB.BINARY, name='x')

        # --- CONSTRAINTS ---
        for patient in patients:
            model_surgeon.addConstr(
                gp.quicksum(x[patient.id, block.id] for block in blocks) <= 1,
                name='constraint_8'
            )

        for block in blocks:
            model_surgeon.addConstr(
                gp.quicksum(patient.duration * x[patient.id, block.id] for patient in patients) <= block.duration,
                name='constraint_9'
            )

        if model._LC_type == LazyConstraint.OLC:
            # --- SET OBJECTIVE ---
            model_surgeon.setObjective(
                gp.quicksum(patient.priority * x[patient.id, block.id]
                    for patient in patients
                    for block in blocks
                ), GRB.MAXIMIZE
            )

            # --- OPTIMIZE ---
            optimize(model_surgeon)
            fs: int = round(model_surgeon.objVal)
            fs_sp: int = 0
            for patient in patients:
                for block in blocks:
                    fs_sp += round(x_c[patient.id, block.id]) * patient.priority

            if fs > fs_sp + model._epsilon:
                M = len(patients) * model._max_priority + 1
                left_side: gp.LinExpr = gp.quicksum(
                    patient.priority * model._x[patient.id, block.id]
                    for patient in patients
                    for block in model._blocks
                ) + M * (model._L - gp.quicksum(
                    model._q[l, num_blocks_of_length[l]]
                    for l in num_blocks_of_length.keys()
                ))
                right_side = fs
                model.cbLazy(left_side + model._epsilon >= round(right_side))
                logger.debug(f'{left_side} >= {right_side}')
                model._num_LCs += 1
                model._LCs[str(num_blocks_of_length)] = (num_blocks_of_length, right_side)

        elif model._LC_type == LazyConstraint.ALC:
            # --- SET OBJECTIVE ---
            delta_expression: gp.LinExpr = gp.quicksum(
                patient.duration * x[patient.id, block.id]
                for block in blocks
                for patient in patients
            )
            w_expression: gp.LinExpr = gp.quicksum(
                patient.leader_priority * (1 - gp.quicksum(x[patient.id, block.id] for block in blocks))
                for patient in patients
            )
            fs_objective: gp.LinExpr = gp.quicksum(
                patient.priority * x[patient.id, block.id]
                for block in blocks
                for patient in patients
            )
            model_surgeon.setObjectiveN(
                fs_objective,
                index=0,
                priority=1,
            )
            model_surgeon.setObjectiveN(
                (model._alpha * delta_expression - model._beta * w_expression),
                1,
                priority=0,
            )
            model_surgeon.ModelSense = GRB.MAXIMIZE

            # --- OPTIMIZE ---
            optimize(model_surgeon)
            fs: int = round(fs_objective.getValue())
            fs_sp: int = 0
            for patient in patients:
                for block in blocks:
                    fs_sp += round(x_c[patient.id, block.id]) * patient.priority

            if fs > fs_sp + model._epsilon:
                patients_assigned: List[Patient] = list()  # a.k.a. PAs
                patients_not_assigned: List[Patient] = list()  # a.k.a. P_{s} \ PAs
                for patient in patients:
                    if sum(round(x[patient.id, block.id].x) for block in blocks) == 1:
                        patients_assigned.append(patient)
                    else:
                        patients_not_assigned.append(patient)
                M = len(patients) + 1
                left_side = gp.quicksum(
                    model._x[patient.id, block.id]
                    for block in model._blocks
                    for patient in patients_assigned
                ) - gp.quicksum(
                    model._x[patient.id, block.id]
                    for block in model._blocks
                    for patient in patients_not_assigned
                ) + M * (
                    model._L - gp.quicksum(model._q[l, num_blocks_of_length[l]] for l in num_blocks_of_length.keys())
                )
                right_side = len(patients_assigned)
                model.cbLazy(left_side + model._epsilon >= round(right_side))
                model._num_LCs += 1
                model._LCs[str(num_blocks_of_length)] = (patients_assigned, patients_not_assigned, num_blocks_of_length)

        logger.debug('--------------------')
        logger.debug(f'Surgeon {s} with assigned blocks {[v[0] for v in o_c.items() if round(v[1]) == 1]}:')
        logger.debug(f'Subproblem x_pb assignment '
                      f'{[v[0] for v in x_c.items() if round(v[1]) == 1 and model._patients[v[0][0]] in patients]}')
        logger.debug(f'Callback x_pb assignment {[var for var, value in x.items() if round(value.x) == 1]}')
        logger.debug(f'fs: {fs}, fs_optim: {model_surgeon.objVal}')
        logger.debug('--------------------')

    model._time_callback += time.time() - start_time


if __name__ == "__main__":
    problem = BnP_Problem("12_ordays_5_load_1.5_surgeons_12", "s5_a1_b1_t1200_InH0_MuP1_LCR1_LC2_decentralized")
    problem.run()
    print(problem.get_result())
