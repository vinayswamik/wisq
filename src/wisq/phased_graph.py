import itertools
import random
import time
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit import QuantumCircuit
import numpy as np


## Random
def build_random_map(log_qubits, arch):
    faces = arch['alg_qubits']
    m = (list(zip(log_qubits, random.sample(faces, len(log_qubits)))))
    return m


def build_phased_connectivity_graph(circuit, include_t=True):
    dag = circuit_to_dag(circuit)
    gates = []
    phased_graphs = {i : np.zeros((dag.num_qubits()+1, dag.num_qubits()+1)) for i in range(dag.depth())}
    layer_counter = 0
    layers = dag.multigraph_layers()
    for layer in layers:
        layer_as_circuit = dag_to_circuit(layer['graph'])
        graph = phased_graphs[layer_counter]
        for j in range(len(layer_as_circuit)): 
            qubits = layer_as_circuit[j][1]
            op = layer_as_circuit[j][0]
            if len(qubits) == 2:
                c,t = qubits
                graph[c.index, t.index] = 1
                graph[t.index, c.index] = -1
            elif (op.name == "t" or op.name == "tdg") and include_t:
                q  = qubits[0]
                graph[q.index,dag.num_qubits()] = 1
        layer_counter += 1
    return phased_graphs

def build_phased_connectivity_graph_fast(circuit, include_t=True):
    dag = circuit_to_dag(circuit)
    gates = []
    phased_graphs = {i : {q : set() for q in range(dag.num_qubits()+1)} for i in range(dag.depth())}
    layers = dag.multigraph_layers()
    try:
        next(layers)  # Remove input nodes
    except StopIteration:
        return phased_graphs
    for i,layer in enumerate(layers):
        op_nodes = [node for node in layer if isinstance(node, DAGOpNode)]
        if not op_nodes:
            return phased_graphs
        graph = phased_graphs[i]
        for node in ((op_nodes)): 
            op = node.op
            qubits = node.qargs
            if len(qubits) == 2:
                c,t = qubits
                graph[c._index].add((c._index, t._index))
                graph[t._index].add((c._index, t._index))
            elif (op.name == "t" or op.name == "tdg") and include_t:
                q  = qubits[0]
                graph[q._index].add((q._index,dag.num_qubits()))
    return phased_graphs

def overlapping(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
    if xmax1 < xmin2 or xmax2 < xmin1:
        return False
    if ymax1 < ymin2 or ymax2 < ymin1:
        return False
    return True


def count_overlapping(mapping, phased_graphs, arch):
    overlaps = 0
    grid_len = arch['width']
    magic_states = arch['magic_states']
    for g in phased_graphs.values():
        edge_min_max = {}
        edges = np.argwhere(g>0) 
        for edge in edges:
            c,t = edge
            if t in mapping.keys():
                xmin, xmax = sorted([mapping[c][0], mapping[t][0]])
                ymin, ymax = sorted([mapping[c][1], mapping[t][1]])
            else:
                magic_states_2d = [tuple(reversed(divmod(m, grid_len))) for m in magic_states]
                closest = min(magic_states_2d, key=lambda p : abs(p[0] - mapping[c][0]) + abs(p[1] - mapping[c][1]) )
                xmin, xmax = sorted([mapping[c][0], closest[0]])
                ymin, ymax = sorted([mapping[c][1], closest[1]])
            edge_min_max[tuple(edge)] = (xmin, xmax, ymin, ymax)
        pairs = itertools.combinations(edges, r=2)
        for edge_pair in pairs:
            xmin1, xmax1, ymin1, ymax1 = edge_min_max[tuple(edge_pair[0])]
            xmin2, xmax2, ymin2, ymax2 =  edge_min_max[tuple(edge_pair[1])]
            if overlapping(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
                overlaps += 1
    return overlaps

def update_overlaps(phased_graphs, arch, old_mapping, new_mapping, qubit1, qubit2):
    overlap_delta = 0
    grid_len = arch['width']
    magic_states = arch['magic_states']
    for g in phased_graphs.values():
        modified_edges = []
        all_edges = np.argwhere(g>0)
        edge_min_max_old = {}
        edge_min_max_new = {}
        adj_q1 = np.argwhere(g[qubit1] != 0)
        adj_q2 = np.argwhere(g[qubit2] != 0)
        for partner in adj_q1:
            if g[qubit1][int(partner)] == -1:
                modified_edges.append([int(partner), qubit1])
            else:
                modified_edges.append([qubit1, int(partner)])
        for partner in adj_q2:
            if partner != qubit1:
                if g[qubit2][int(partner)] == -1:
                    modified_edges.append([int(partner), qubit2])
                else:
                    modified_edges.append([qubit2, int(partner)])
        for edge in all_edges:
            c,t = edge
            if t in old_mapping.keys():
                xmin_old, xmax_old = sorted([old_mapping[c][0], old_mapping[t][0]])
                ymin_old, ymax_old = sorted([old_mapping[c][1], old_mapping[t][1]])
                xmin_new, xmax_new = sorted([new_mapping[c][0], new_mapping[t][0]])
                ymin_new, ymax_new = sorted([new_mapping[c][1], new_mapping[t][1]])
            else:
                magic_states_2d = [tuple(reversed(divmod(m, grid_len))) for m in magic_states]
                closest_old = min(magic_states_2d, key=lambda p : abs(p[0] - old_mapping[c][0]) + abs(p[1] - old_mapping[c][1]) )
                xmin_old, xmax_old = sorted([old_mapping[c][0], closest_old[0]])
                ymin_old, ymax_old = sorted([old_mapping[c][1], closest_old[1]])
                closest_new = min(magic_states_2d, key=lambda p : abs(p[0] - new_mapping[c][0]) + abs(p[1] - new_mapping[c][1]) )
                xmin_new, xmax_new = sorted([new_mapping[c][0], closest_new[0]])
                ymin_new, ymax_new = sorted([new_mapping[c][1], closest_new[1]])
            edge_min_max_old[tuple(edge)] = (xmin_old, xmax_old, ymin_old, ymax_old)
            edge_min_max_new[tuple(edge)] = (xmin_new, xmax_new, ymin_new, ymax_new)
        pairs = itertools.product(all_edges,modified_edges)
        seen = set()
        for edge_pair in pairs:
            (c,t) = edge_pair[0] 
            (cp,tp) = edge_pair[1]
            seen.add(((c,t), (cp, tp)))
            if ((cp, tp), (c,t)) not in seen:
                xmin1, xmax1, ymin1, ymax1 = edge_min_max_old[(c,t)]
                xmin2, xmax2, ymin2, ymax2 =  edge_min_max_old[tuple(edge_pair[1])]
                if overlapping(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
                    overlap_delta -= 1
                xmin1, xmax1, ymin1, ymax1 = edge_min_max_new[(c,t)]
                xmin2, xmax2, ymin2, ymax2 =  edge_min_max_new[tuple(edge_pair[1])]
                if overlapping(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
                    overlap_delta += 1
    return overlap_delta

def count_overlapping_fast(mapping, phased_graphs, arch):
    overlaps = 0
    grid_len = arch['width']
    magic_states = arch['magic_states']
    for g in phased_graphs.values():
        edge_min_max = {}
        edges = {x  for q in g.keys() for x in g[q]}
        for edge in edges:
            c,t = edge
            if t in mapping.keys():
                xmin, xmax = sorted([mapping[c][0], mapping[t][0]])
                ymin, ymax = sorted([mapping[c][1], mapping[t][1]])
            else:
                magic_states_2d = [tuple(reversed(divmod(m, grid_len))) for m in magic_states]
                closest = min(magic_states_2d, key=lambda p : abs(p[0] - mapping[c][0]) + abs(p[1] - mapping[c][1]) )
                xmin, xmax = sorted([mapping[c][0], closest[0]])
                ymin, ymax = sorted([mapping[c][1], closest[1]])
            edge_min_max[tuple(edge)] = (xmin, xmax, ymin, ymax)
        pairs = itertools.combinations(edges, r=2)
        for edge_pair in pairs:
            xmin1, xmax1, ymin1, ymax1 = edge_min_max[tuple(edge_pair[0])]
            xmin2, xmax2, ymin2, ymax2 =  edge_min_max[tuple(edge_pair[1])]
            if overlapping(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
                overlaps += 1
    return overlaps

def update_overlaps_fast(phased_graphs, arch, old_mapping, new_mapping, qubit1, qubit2):
    overlap_delta = 0
    grid_len = arch['width']
    magic_states = arch['magic_states']
    for g in phased_graphs.values():
        modified_edges = g[qubit1].union(g[qubit2])
        edge_min_max_old = {}
        edge_min_max_new = {}
        all_edges = {x  for q in g.keys() for x in g[q]}
        for edge in all_edges:
            c,t = edge
            if t in old_mapping.keys():
                xmin_old, xmax_old = sorted([old_mapping[c][0], old_mapping[t][0]])
                ymin_old, ymax_old = sorted([old_mapping[c][1], old_mapping[t][1]])
                xmin_new, xmax_new = sorted([new_mapping[c][0], new_mapping[t][0]])
                ymin_new, ymax_new = sorted([new_mapping[c][1], new_mapping[t][1]])
            else:
                magic_states_2d = [tuple(reversed(divmod(m, grid_len))) for m in magic_states]
                closest_old = min(magic_states_2d, key=lambda p : abs(p[0] - old_mapping[c][0]) + abs(p[1] - old_mapping[c][1]) )
                xmin_old, xmax_old = sorted([old_mapping[c][0], closest_old[0]])
                ymin_old, ymax_old = sorted([old_mapping[c][1], closest_old[1]])
                closest_new = min(magic_states_2d, key=lambda p : abs(p[0] - new_mapping[c][0]) + abs(p[1] - new_mapping[c][1]) )
                xmin_new, xmax_new = sorted([new_mapping[c][0], closest_new[0]])
                ymin_new, ymax_new = sorted([new_mapping[c][1], closest_new[1]])
            edge_min_max_old[tuple(edge)] = (xmin_old, xmax_old, ymin_old, ymax_old)
            edge_min_max_new[tuple(edge)] = (xmin_new, xmax_new, ymin_new, ymax_new)
        pairs = itertools.product(all_edges,modified_edges)
        seen = set()
        for edge_pair in pairs:
            (c,t) = edge_pair[0] 
            (cp,tp) = edge_pair[1]
            seen.add(((c,t), (cp, tp)))
            if ((cp, tp), (c,t)) not in seen:
                xmin1, xmax1, ymin1, ymax1 = edge_min_max_old[(c,t)]
                xmin2, xmax2, ymin2, ymax2 =  edge_min_max_old[tuple(edge_pair[1])]
                if overlapping(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
                    overlap_delta -= 1
                xmin1, xmax1, ymin1, ymax1 = edge_min_max_new[(c,t)]
                xmin2, xmax2, ymin2, ymax2 =  edge_min_max_new[tuple(edge_pair[1])]
                if overlapping(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
                    overlap_delta += 1
    return overlap_delta

def sim_anneal(mapping, phased_graphs_fast, arch, retain_history, temperature=100, cooling_rate=0.1, termination_temp=0.1, timeout=3600):
    current_mapping = mapping.copy()
    best_mapping = mapping.copy()
    #best_overlaps = count_overlapping(mapping, phased_graphs,arch)
    best_overlaps = count_overlapping_fast(mapping, phased_graphs_fast,arch)
    current_overlaps = best_overlaps
    #assert(best_overlaps == best_overlaps_fast)
    visited = []
    start = time.time()
    current = start
    steps = 0
    while temperature > termination_temp and best_overlaps > 0 and current-start < timeout:
        steps += 1
        new_mapping = current_mapping.copy()
        qubit1, qubit2 = np.random.choice(np.fromiter(mapping.keys(), dtype=int), size=2, replace=False)
        new_mapping[qubit1], new_mapping[qubit2] = new_mapping[qubit2], new_mapping[qubit1]
        #delta = update_overlaps(phased_graphs, arch, best_mapping, new_mapping, qubit1, qubit2)
        delta_curr = update_overlaps_fast(phased_graphs_fast, arch, current_mapping, new_mapping, qubit1, qubit2)
        delta_best = update_overlaps_fast(phased_graphs_fast, arch, best_mapping, new_mapping, qubit1, qubit2)
        #assert delta == delta_fast, f"{delta}, {delta_fast}"
        new_overlaps = current_overlaps + delta_curr
        if retain_history:
            visited.append((new_mapping, new_overlaps))
        if delta_curr < 0 or np.random.rand() < np.exp(-delta_curr / temperature):
            current_mapping = new_mapping
            current_overlaps = new_overlaps
        if delta_best < 0 or np.random.rand():
            best_mapping = new_mapping
            best_overlaps = new_overlaps
        temperature *= 1 - cooling_rate
        current = time.time()
    #print(f"mapping sa steps {steps}")
    if retain_history:
        return visited
    else:
        return best_mapping, best_overlaps


def build_phased_map(log_qubits, circ, arch, initial_temp, cooling_rate, term_temp,  timeout, include_t=True, retain_history=False):
    grid_len = arch['width']
    faces = arch['alg_qubits']
    map_tuples = build_random_map(log_qubits, arch)
    map_flat = {t[0] :  t[1] for t in map_tuples}
    map_2d = {k : tuple(reversed(divmod(v, grid_len))) for k, v in map_flat.items()}
    initial_mapping = map_2d
    #initial_mapping = {i : tuple(reversed(divmod(faces[i], grid_len))) for i in range(log_num)}

    p_g_fast = build_phased_connectivity_graph_fast(circ, include_t=include_t)
    if retain_history:
        mappings = sim_anneal(initial_mapping, p_g_fast, arch, timeout=timeout, temperature=1, cooling_rate=0.001, retain_history=True)
        return [([(key, val[1]*grid_len + val[0]) for key, val in mapping.items()], overlaps) for mapping, overlaps  in mappings]
    else:
        final_mapping, cost = sim_anneal(initial_mapping, p_g_fast, arch,timeout=timeout, temperature=initial_temp, cooling_rate=cooling_rate, termination_temp=term_temp, retain_history=False)
        tuples = [(key, val[1]*grid_len + val[0]) for key, val in final_mapping.items()]
        return tuples, cost
 
if __name__ == "__main__":
    log_num = 100
    circ = QuantumCircuit.from_qasm_file("qft-like/simple_qft_q100.qasm")
    build_phased_map(log_num, circ, 10, 10, [], retain_history=False)