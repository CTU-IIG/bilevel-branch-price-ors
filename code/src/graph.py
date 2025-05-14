import graphviz
from dataclasses import dataclass, field
from typing import List


@dataclass
class Node:
    id: int
    depth: int
    objective_value: float = None
    incumbent_objective_value: float = None
    time: float = None
    num_columns_added: int = None
    num_CG_iterations: int = None
    ILP_solved: bool = False
    color: str = None


@dataclass
class Edge:
    start: int
    end: int
    # branching variable
    s: int
    b: int
    value: int

    def __post_init__(self):
        self.label: str = f'y_{{{self.s},{self.b}}} = {self.value}'


@dataclass
class Graph:
    name: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, id: int, depth: int):
        self.nodes.append(Node(id=id, depth=depth))

    def add_edge(self, start: int, end: int, s: int, b: int, value: int):
        self.edges.append(Edge(start=start, end=end, s=s, b=b, value=value))

    def update_node(self, id: int, updates: dict):
        for parameter, value in updates.items():
            setattr(self.nodes[id], parameter, value)

    def save_and_render(self):
        # create graph
        graph = graphviz.Digraph(filename=self.name, format='pdf')

        # add nodes
        for node in self.nodes:
            graph.node(name=f'node_{node.id}', label='\n'.join([f'{key}={value}' for key, value in node.__dict__.items()
                                                                if (value and key != 'color')]),
                       style='filled', fillcolor=node.color)

        # add edges
        for edge in self.edges:
            graph.edge(f'node_{edge.start}', f'node_{edge.end}', label=edge.label)

        graph.save(directory="./outputs/graphical/trees")
        graph.render(filename=self.name, view=False, cleanup=True)
