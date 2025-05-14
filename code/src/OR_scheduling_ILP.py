# python built-ins
import logging
import random
import time
from typing import Optional, Any
from collections import defaultdict

# external packages
import gurobipy as gp
import matplotlib.patches as mpatches
import numpy as np
from gurobipy import GRB, tuplelist
from matplotlib import pyplot as plt

from src.OR_scheduling import Problem, ProblemResult
from src.classes import *

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()

logging.basicConfig(format='%(message)s', level=logging.INFO)


class ILP_Problem(Problem):
    def __init__(self, instance: str, parameters: str, name: Optional[str] = None, input_constraints: Optional[dict] = None, cutoff: float = None):
        """
        parse input file and get output variables/parameters
        :param instance:
        :param parameters:
        """

        super().__init__(instance, parameters, name)

        # --- LOAD CONSTRAINTS

        self.input_x_constraints: List = []
        self.input_y_constraints: List = []
        self.input_LC_constraints: List = []
        if input_constraints:
            if 'y' in input_constraints:
                self.input_y_constraints: List = input_constraints['y']
            if 'LC' in input_constraints:
                self.input_LC_constraints: List = input_constraints['LC']

        # --- INIT MODEL
        self.model: gp.Model = gp.Model(env=env)
        self.model.Params.lazyConstraints = 1
        self.model.Params.TimeLimit = self.timeout
        self.model.Params.NumericFocus = 3
        self.model.Params.PreSolve = 0
        self.model.Params.Heuristics = 0
        self.model.Params.Cuts = 0
        if cutoff:
            self.model.Params.Cutoff = cutoff
        self.otype: Any = GRB.MAXIMIZE if self.decentralized else GRB.MINIMIZE

        if self.decentralized: self.is_callback = False

    def _add_variables(self):
        """
        adds variables x_{p,b}, y_{s,b}, q_{s,l,k}
        :return:
        """

        def inner_add_variables():
            # x_{p,b} \in {0, 1} patient p is allocated to block b
            self.model._x = self.model.addVars(self.num_patients, len(self.blocks), vtype=GRB.BINARY, name='x')

            # y_{s,b} \in {0, 1} block b is assigned to surgeon s
            self.model._y = self.model.addVars(self.num_surgeons, len(self.blocks), vtype=GRB.BINARY, name='y')

            # q_{s,l,k} \in {0, 1} q_{s,l,k} = 1 iff surgeon s has exactly k blocks of length at least l
            # adding variable q iterating over S, L, K_{l}
            q_: tuplelist = tuplelist()
            for s in range(self.num_surgeons):
                for l in self.K.keys():
                    for k in range(self.num_days + 1):
                        q_.append((s, l, k))
            self.model._q = self.model.addVars(q_, vtype=GRB.BINARY, name='q')

        def inner_add_quality_parameters():
            # OR load
            self.model._U: gp.LinExpr = gp.quicksum(patient.duration * self.model._x[patient.id, block.id]
                                                    for block in self.blocks
                                                    for patient in self.patients) / self.capacity

            # leader priority
            self.model._P_leader: gp.LinExpr = gp.quicksum(patient.leader_priority * self.model._x[patient.id, block.id]
                                                           for block in self.blocks
                                                           for patient in self.patients)

            # follower priority
            self.model._P_follower: gp.LinExpr = gp.quicksum(patient.priority * self.model._x[patient.id, block.id]
                                                             for block in self.blocks
                                                             for patient in self.patients)

        inner_add_variables()
        inner_add_quality_parameters()
        self.model.update()

    def _add_constraints(self):
        """
        adds constraints according to the paper
        :return:
        """
        # add x constraints
        for node in self.input_x_constraints:
            self.model.addConstr(self.model._x[node["p"], node["b"]] == node["value"],
                                 name=f'input_x_constraints__{node["p"]}_{node["b"]}_{node["value"]}')

        # add y constraints
        for node in self.input_y_constraints:
            self.model.addConstr(self.model._y[node["s"], node["b"]] == node["value"],
                                 name=f'input_y_constraints__{node["s"]}_{node["b"]}_{node["value"]}')

        # add LC constraints
        for s, LCs in enumerate(self.input_LC_constraints):
            if self.LC_type == LazyConstraint.OLC:
                M = len(self.surgeons[s].patients) * self.max_priority + 1
                for i, (num_blocks_of_length, right_side) in enumerate(LCs.values()):
                    left_side = gp.quicksum(
                        patient.priority * self.model._x[patient.id, block.id]
                        for patient in self.surgeons[s].patients
                        for block in self.blocks
                    ) + M * (self.L - gp.quicksum(self.model._q[s, l, num_blocks_of_length[l]] for l in num_blocks_of_length.keys()))
                    self.model.addConstr(left_side + self.epsilon >= round(right_side), name=f'remembered_LC_{s}_{i}')

            elif self.LC_type == LazyConstraint.ALC:
                M = len(self.surgeons[s].patients) + 1
                for i, (patients_assigned, patients_not_assigned, num_blocks_of_length) in enumerate(LCs.values()):
                    left_side = gp.quicksum(
                        self.model._x[patient.id, block.id]
                        for block in self.blocks
                        for patient in patients_assigned
                    ) - gp.quicksum(
                        self.model._x[patient.id, block.id]
                        for block in self.blocks
                        for patient in patients_not_assigned
                    ) + M * (
                        self.L -
                        gp.quicksum(self.model._q[s, l, num_blocks_of_length[l]] for l in num_blocks_of_length.keys())
                    )
                    right_side = len(patients_assigned)
                    self.model.addConstr(left_side + self.epsilon >= round(right_side), name=f'remembered_LC_{s}_{i}')

        # capacity of ORs (eq 5.2)
        for d in range(self.num_days):
            for t in self.start_times:
                overlapping_blocks: List[Block] = self._get_overlapping_blocks(day=d, time=t)  # a.k.a. O_{d,t}
                self.model.addConstr(
                    gp.quicksum(
                        self.model._y[s, block.id]
                        for block in overlapping_blocks
                        for s in range(self.num_surgeons)
                    ) <= self.num_ORs,
                    name=f'capacity_of_ORs__{d}_{t}'
                )

        # a surgeon can be assigned at most one block a day (eq 5.3)
        for s in range(self.num_surgeons):
            for d in range(self.num_days):
                day_blocks: List[Block] = self._get_day_blocks(day=d)  # a.k.a. B_{d}
                self.model.addConstr(
                    gp.quicksum(self.model._y[s, block.id] for block in day_blocks) <= 1,
                    name=f'at_most_1_block_per_day__{s}_{d}'
                )

        # the minimum number of the assigned blocks (eq 5.4)
        # if self.otype == GRB.MAXIMIZE:
        #     self.m = 1

        for s in range(self.num_surgeons):
            self.model.addConstr(
                gp.quicksum(self.model._y[s, block.id] for block in self.blocks) >= self.m,
                name=f'minimum_number_of_assigned_blocks__{s}'
            )

        # auxiliary constraints - amount of blocks of length l assigned to surgeon s (eq 5.5, 5.6)
        for s in range(self.num_surgeons):
            for l in self.K.keys():
                self.model.addConstr(
                    gp.quicksum(k * self.model._q[s, l, k] for k in range(self.num_days + 1))
                    ==
                    gp.quicksum(self.model._y[s, block.id] for block in self.blocks_of_length[l]),
                    name=f"first_q_constraint__{s}_{l}"
                )
                self.model.addConstr(
                    gp.quicksum(self.model._q[s, l, k] for k in range(self.num_days + 1)) == 1,
                    name=f"second_q_constraint__{s}_{l}"
                )

        # a patient can be scheduled only once (eq 5.7)
        for p in range(self.num_patients):
            self.model.addConstr(
                gp.quicksum(self.model._x[p, block.id] for block in self.blocks) <= 1,
                name=f'patient_can_be_scheduled_only_once__{p}'
            )

        # capacity of the block b assigned to surgeon s cannot be exceeded (eq 5.8)
        for s in range(self.num_surgeons):
            for block in self.blocks:
                self.model.addConstr(
                    gp.quicksum(
                        patient.duration * self.model._x[patient.id, block.id]
                        for patient in self.surgeons[s].patients
                    ) <= block.duration * self.model._y[s, block.id],
                    name=f'capacity_of_block_cannot_be_exceeded__{s}_{block.id}'
                )

        self.model.update()

    def _set_objective(self):
        """
        sets objective of the model
        :return:
        """

        idle_time: gp.LinExpr = self.capacity - gp.quicksum(
            patient.duration * self.model._x[patient.id, block.id]
            for block in self.blocks
            for patient in self.patients
        )
        loss_function: gp.LinExpr = gp.quicksum(
            patient.leader_priority * (1 - gp.quicksum(self.model._x[patient.id, block.id] for block in self.blocks))
            for patient in self.patients
        )
        min_objective = self.alpha * idle_time + self.beta * loss_function

        if self.otype == GRB.MINIMIZE:
            self.model.setObjective(min_objective, GRB.MINIMIZE)
        elif self.otype == GRB.MAXIMIZE:  # Decentralized Solution
            self.min_objective: gp.LinExpr = min_objective
            self.model.setObjective(
                gp.quicksum(
                    patient.priority * self.model._x[patient.id, block.id]
                    for patient in self.patients
                    for block in self.blocks
                ), GRB.MAXIMIZE
            )

        self.model.update()

    def solve(self):
        """
        calls Gurobi solve method and use callback function
        :return:
        """
        self._add_variables()
        self._add_constraints()
        self._set_objective()

        start = time.time()
        self.model._num_callbacks = 0
        self.model._num_LCs = 0
        self.model._cbtime = 0
        self.model._patients = self.patients
        self.model._surgeons = self.surgeons
        self.model._blocks = self.blocks
        self.model._num_patients = self.num_patients
        self.model._num_surgeons = self.num_surgeons
        self.model._block_durations = self.block_durations
        self.model._alpha = self.alpha
        self.model._beta = self.beta
        self.model._L = self.L
        self.model._M = self.M
        self.model._epsilon = self.epsilon
        self.model._LC_type = self.LC_type
        optimize(model=self.model, is_callback=self.is_callback)
        self.overall_time = time.time() - start

    def print_solution(self):
        """
        prints a solution regarding surgeons and patients
        :return:
        """
        self.model.write(f'outputs/models/{self.instance}.lp')
        for var, value in self.model._y.items():
            if round(value.x) == 1:
                logging.info(f'Surgeon id={var[0]} has a block id={var[1]} on day {self.blocks[var[1]].day}, '
                             f'starting at {self.blocks[var[1]].start} '
                             f'and ending at {self.blocks[var[1]].end}')
        for var, value in self.model._x.items():
            if round(value.x) == 1:
                logging.info(
                    f'Patient id={var[0]}, duration={self.patients[var[0]].duration}, priority={self.patients[var[0]].priority} is scheduled to a block id={var[1]} on day {self.blocks[var[1]].day}, '
                    f'starting at {self.blocks[var[1]].start} '
                    f'and ending at {self.blocks[var[1]].end}')
        logging.info(f'Value of objective function: {self.model.objVal}')
        logging.info(f'Overall time: {self.overall_time}')

    def timeline(self):
        """
        draws a timeline chart and saves it to a png file
        :return:
        """
        colors: dict = dict()
        patches: List[mpatches.Patch] = list()
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

        patients_in_block: List[List[Patient]] = [[] for _ in range(len(self.blocks))]
        for var, value in self.model._x.items():
            if round(value.x) == 1:
                patients_in_block[var[1]].append(self.patients[var[0]])

        for b, patients in enumerate(patients_in_block):
            if patients:
                offset: int = self.blocks[b].start
                sorted_patients: List[Patient] = sorted(patients, key=lambda x: x.priority, reverse=True)
                for i, patient in enumerate(sorted_patients):
                    gnt.broken_barh(
                        [(offset, patient.duration)],
                        (self.blocks[b].day * day_margin, 1),
                        facecolors=(colors[patient.surgeon_id]),
                        edgecolor='black'
                    )
                    gnt.text(offset + patient.duration / 2, self.blocks[b].day * day_margin + 0.5,
                             f'P{patient.id}',
                             ha='center', va='center', size=10)
                    offset += patient.duration

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
        plt.savefig(f'outputs/graphical/schedules/ILP_{self.instance}_{self.parameters}.png', bbox_inches='tight')

    def return_objective_value(self):
        return min(round(self.model.objVal, 3), 1000)

    def return_instance_results(self):
        return [
            self.overall_time,
            self.model.Status != GRB.TIME_LIMIT,
            self.model._cbtime,
            0.,
            round(self.model.objVal, 3),
            self.model._num_callbacks,
            self.model._num_LCs
        ]

    def return_solution_quality_parameters(self):
        return [self.model._U.getValue(), self.model._P_leader.getValue(), round(self.model.objVal, 3),
                self.model._P_follower.getValue(), self.overall_time]

    def return_F_lpr(self):
        relax_model = self.model.relax()
        optimize(relax_model)
        return relax_model.objVal

    def get_surgeon_block_schedule(self):
        surgeon_schedule: dict[Surgeon, list[Block]] = defaultdict(list)
        for var, value in self.model._y.items():
            if round(value.x) == 1:
                surgeon = var[0]
                block = var[1]
                surgeon_schedule[surgeon].append(self.blocks[block])
        return surgeon_schedule

    def get_surgeon_patient_schedule(self):
        schedule: dict[Surgeon, list[Block]] = defaultdict(list)
        patients_in_block = defaultdict(list)
        for var, value in self.model._x.items():
            if round(value.x) == 1:
                patient = var[0]
                block = var[1]
                patients_in_block[block].append(self.patients[patient])
        for var, value in self.model._y.items():
            if round(value.x) == 1:
                surgeon = var[0]
                block = var[1]
                schedule[surgeon].extend(patients_in_block[block])
        return schedule

    def get_result(self) -> ProblemResult:
        result = ProblemResult()

        result.proved_optimal = self.model.Status != GRB.TIME_LIMIT

        result.num_LCs = self.model._num_LCs
        result.num_callbacks = self.model._num_callbacks

        result.callback_time = self.model._cbtime
        result.overall_time = self.overall_time

        result.relax_obj = self.return_F_lpr()
        result.obj = round(self.min_objective.getValue()) if self.decentralized else self.model.objVal

        if result.obj != np.inf:
            result.obj = int(round(result.obj))

            for var, value in self.model._y.items():
                if round(value.x) == 1:
                    surgeon = var[0]
                    block = var[1]
                    result.surgeon_schedule[surgeon].append(block)
            for var, value in self.model._x.items():
                if round(value.x) == 1:
                    patient = var[0]
                    block = var[1]
                    result.patient_schedule[patient] = block

        return result

    def run(self):
        self.solve()


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

    start = time.time()
    model._num_callbacks += 1

    y_c: gp.Var = model.cbGetSolution(model._y)
    x_c: gp.Var = model.cbGetSolution(model._x)

    for s in range(model._num_surgeons):
        # y_{s,b} is set -> filter surgeon's patients and assigned blocks
        patients: List[Patient] = model._surgeons[s].patients
        blocks: List[Block] = list()
        for block in model._blocks:
            if round(y_c[s, block.id]) == 1:
                blocks.append(block)

        num_blocks_of_length: defaultdict = defaultdict(int, {k: 0 for k in model._block_durations})  # a.k.a. n_{s,l}^c
        for block in blocks:
            num_blocks_of_length[block.duration] += 1

        # --- INIT MODEL
        model_surgeon: gp.Model = gp.Model(env=env)

        # --- VARIABLES
        x = model_surgeon.addVars(
            [(patient.id, block.id) for patient in patients for block in blocks],
            vtype=GRB.BINARY, name='x'
        )
        # x_: tuplelist = tuplelist()
        # for patient in patients:
        #     for block in blocks:
        #         x_.append((patient.id, block.id))
        # x = model_surgeon.addVars(x_, vtype=GRB.BINARY, name='x')

        # --- CONSTRAINTS ---

        # a patient can be scheduled only once (eq 5.12)
        for patient in patients:
            model_surgeon.addConstr(
                gp.quicksum(x[patient.id, block.id] for block in blocks) <= 1,
                name=f'CB_patient_can_be_scheduled_only_once__{patient.id}'
            )

        # capacity of the block b cannot be exceeded (eq 5.12)
        for block in blocks:
            model_surgeon.addConstr(
                gp.quicksum(
                    patient.duration * x[patient.id, block.id]
                    for patient in patients
                ) <= block.duration,
                name=f'CB_capacity_of_block_cannot_be_exceeded__{block.id}'
            )

        # --- SET OBJECTIVE AND SOLVE

        if model._LC_type == LazyConstraint.OLC:
            # --- SET OBJECTIVE ---
            model_surgeon.setObjective(
                gp.quicksum(
                    patient.priority * x[patient.id, block.id]
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
                left_side: gp.LinExpr = gp.quicksum(patient.priority * model._x[patient.id, block.id]
                                                    for patient in patients
                                                    for block in model._blocks
                                                    ) + \
                                        (len(patients) * 4 + 1) * (
                                                model._L - gp.quicksum(model._q[s, l, num_blocks_of_length[l]]
                                                                       for l in num_blocks_of_length.keys()
                                                                       )
                                        )
                right_side = fs
                model.cbLazy(left_side + model._epsilon >= round(right_side))
                logging.debug(f'{left_side} >= {right_side}')
                model._num_LCs += 1

        elif model._LC_type == LazyConstraint.ALC:
            # --- SET OBJECTIVE ---
            delta_expression: gp.LinExpr = gp.quicksum(
                patient.duration * x[patient.id, block.id]
                for patient in patients
                for block in blocks
           )
            w_expression: gp.LinExpr = gp.quicksum(
                patient.leader_priority * (1 - gp.quicksum(x[patient.id, block.id] for block in blocks))
                for patient in patients
            )
            fs_objective: gp.LinExpr = gp.quicksum(
                patient.priority * x[patient.id, block.id]
                for patient in patients
                for block in blocks
            )
            model_surgeon.setObjective(
                fs_objective + 1 / model._M * (model._alpha * delta_expression - model._beta * w_expression),
                GRB.MAXIMIZE
            )

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
                    model._L - gp.quicksum(
                        model._q[s, l, num_blocks_of_length[l]]
                        for l in num_blocks_of_length.keys()
                    )
                )
                right_side = len(patients_assigned)
                model.cbLazy(left_side + model._epsilon >= round(right_side))
                logging.debug(f'{left_side} >= {right_side}')
                model._num_LCs += 1

        logging.debug(
            f'Surgeon {s} with assigned blocks {[v[0][1] for v in y_c.items() if round(v[1]) == 1 and v[0][0] == s]}:')
        logging.debug(
            f'ILP x_pb assignment {[v[0] for v in x_c.items() if round(v[1]) == 1 and model._patients[v[0][0]] in model._surgeons[s].patients]}')
        logging.debug(f'Callback x_pb assignment {[var for var, value in x.items() if round(value.x) == 1]}')
        logging.debug(f'fs: {fs}, fs_sp: {fs_sp}')
        logging.debug('--------------------')

    model._cbtime += time.time() - start
