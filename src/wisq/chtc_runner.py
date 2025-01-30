import argparse
import ast
import itertools
import os
import re
import time
import  optimal
from phased_graph import build_phased_map
import relaxed
import math
from qiskit import QuantumCircuit
from os import mkdir
import signal
import architecture
from autobraid_interface import check_mapping, run_autobraid, t_to_cnot
from qiskit.converters import circuit_to_dag, dag_to_circuit
import relaxed_arb_layout
import optimal_arb_layout
from relaxed_arb_layout import vertical_neighbors, horizontal_neighbors
from pyautobraid import autobraid_map, autobraid_route
from sarouting import sim_anneal_route


class TimeoutError(Exception):
    pass

def _sig_alarm(sig, tb):
    raise TimeoutError("timeout")

def extract_gates(circ : QuantumCircuit):
    gates = []
    with_reg = []
    for j in range(len(circ)):
        qubits = circ[j][1]
        if len(qubits) == 2:
            with_reg.append([(circ.find_bit(q)[1][0][0].name, q.index) for q in qubits])
        elif len(qubits) > 2:
            print('Warning: ignoring gate with more than 2 qubits')
        rs = [ind for gate in with_reg for (reg, ind) in gate if reg == 'r']
        if len(rs) > 0:
            max_orig = max([ind for gate in with_reg for (reg, ind) in gate if reg == 'q'])
            offset = lambda x : x[1] if x[0] == 'q' else max_orig + 1 + x[1]
            gates = [[offset(ctrl), offset(tar)] for [ctrl, tar] in with_reg]
        else:
            gates = [[ctrl[1], tar[1]] for [ctrl, tar] in with_reg]
    return gates

def multiregister(circ):
    last_reg_name = None
    for j in range(len(circ)): 
        qubits = circ[j][1]
        new_regs = [circ.find_bit(q)[1][0][0].name for q in qubits]
        if last_reg_name != None and any([r != last_reg_name for r in new_regs]):
            return True
        last_reg_name = new_regs[0]
    return False

def extract_gates_from_file(fname):
    gates = []
    with open(fname) as f:
        for line in f:
            match = re.match(r'cx\s+q\[(\d+)\],\s*q\[(\d+)\];', line)
            if match:
                gates.append((int(match.group(1)), int(match.group(2))))
            match = re.match(r'(?:t|tdg)\s+q\[(\d+)\];', line)
            if match:
                gates.append((int(match.group(1)),))
    return gates

def extract_qubits(fname):
    # Returns highest-value register used
    highest_register_value = 0
    with open(fname) as f:
        for line in f:
            if 'qreg' in line or 'creg' in line:  # these are not actual instructions
                continue  # skip adding qbits to the set
            match = re.findall(r'\[(\d+)\]', line)
            for num in match:
                num = int(num)
                if num > highest_register_value:
                    highest_register_value = num
    return highest_register_value + 1  # <--register values start at 0

def extract_qubits_from_gates(gate_list):
    qubits = set()
    for gate in gate_list:
        for qubit in gate:
            qubits.add(qubit)
    return qubits

def center_column(width, height):
    return [(width*i)+(width//2) for i in range(height)]

def right_column(width, height):
    return [(width*i)+(width-1) for i in range(height)]

def all_sides(width, height):
    left_column = [(width*i) for i in range(height)]
    right_column =  [(width*i)+(width-1) for i in range(height)]
    top_row = [i for i in range(width)]
    bottom_row = [(width-1) + (height-1) + i for i in range(width)]
    return list(set(left_column + right_column + top_row + bottom_row))

def solve_phased_placement(fname, include_t, arch, timeout, t_aware_mapping, initial_temp=10, cooling_rate=0.1, term_temp=0.1, routing ='greedy'):
    num_qubits_input = extract_qubits(fname)
    if include_t == 'as_is':
            gates = extract_gates_from_file(fname)
            circ = QuantumCircuit.from_qasm_file(fname)
            total_qubits = num_qubits_input
    elif include_t == 'to_cnot':
            num_t = 4*(math.ceil(math.sqrt(num_qubits_input))+1)
            ghost_ts = t_to_cnot(fname,num_t, cut_other_1q=True)
            circ = QuantumCircuit.from_qasm_str(ghost_ts.qasm())
            print("qubits:", circ.num_qubits)
            gates = extract_gates(QuantumCircuit.from_qasm_str(ghost_ts.qasm()))
            total_qubits = circ.num_qubits
    elif include_t == 'ignore':
            gates = [g for g in extract_gates_from_file(fname) if len(g) == 2]
            total_qubits = num_qubits_input
            circ = QuantumCircuit.from_qasm_file(fname)
    start = time.time()
    phased_map, cost = build_phased_map(total_qubits, circ, arch, include_t=t_aware_mapping, timeout=timeout, initial_temp=initial_temp, cooling_rate=cooling_rate, term_temp=term_temp)
    end = time.time()
    print(f"mapping time {end-start}")
    if routing == 'autobraid':
        return phased_map, autobraid_route(gates, arch, mapping = phased_map)
    elif routing == 'reorder':
        start = time.time()
        route = sim_anneal_route(gates, arch, mapping=phased_map)
        end = time.time()
        print(f"routing time {end-start}")
        return phased_map, route
    else:
     return phased_map, relaxed_arb_layout.solve(gates, arch, initial_map=phased_map)
    
def verify(gates, timesteps, arch, mapping):
    gate_id_table = { i : gate for i,gate, in enumerate(gates)}
    gate_to_step = {}
    mapping = {q : p for (q,p) in mapping}
    assert len(set(mapping.values())) == len(mapping.values()), 'mapping not injective'
    assert all(v in arch['alg_qubits'] for v in mapping.values()), 'mapping target invalid'
    #all gates executed
    for gate in gate_id_table:
        for i,step in enumerate(timesteps):
            executed_ids = [x[0] for x in step]
            if gate in executed_ids:
                gate_to_step[gate] = i
    assert all([key  in gate_to_step.keys() for key in  gate_id_table.keys()]), 'gate not scheduled'
    for i in range(len(gates)):
        for prev in range(i):
            qubits_prev = gate_id_table[prev]
            qubits_this = gate_id_table[i]
            if len([q for q in qubits_prev if q in qubits_this]) > 0:
                assert(gate_to_step[prev] < gate_to_step[i]), f'gate scheduled out of order: {prev, i, gate_to_step}'
    # paths disjoint
    for step in timesteps:
        paths = [x[2] for x in step]
        for p1, p2 in itertools.combinations(paths, 2):
            assert(len(set(p1) & set(p2)) == 0), 'paths intersect'
    for g in gate_id_table:
        gate_step = timesteps[gate_to_step[g]]
        id, qubits, path = [(id,qubits, path) for id, qubits, path in gate_step if id == g][0]
        assert all(q not in path for q in mapping.values()), 'circuit qubit used for routing'
        assert all(q not in path for q in arch['magic_states']), 'magic state qubit used for routing'
        assert(path[0] in vertical_neighbors(mapping[qubits[0]], arch['width'], arch['height'], [])), 'path start wrong'
        if len(gates[id])==2:
            assert(path[-1] in horizontal_neighbors(mapping[qubits[1]], arch['width'], arch['height'], [])), 'cx path endpoint wrong'
        else:
             assert(path[-1] in [n for m in arch['magic_states'] for n in horizontal_neighbors(m, arch['width'], arch['height'], []) ]), 't path endpoint wrong'
        for i in range(len(path)-1):
            assert(abs(path[i]-path[i+1]) in {1,arch['width']}), 'path invalid'
    return


def run_methods(width, height, msf_faces, arch_name, example, methods_list, place_shots, exploration_depth, include_t, t_aware_mapping, rid, timeout, sim_anneal_params, sim_anneal_params_r, take_first_ms, order_fraction):
    trivial_route = math.inf
    circ = QuantumCircuit.from_qasm_file(example)
    if multiregister(circ):
        flat_circuit = QuantumCircuit(circ.num_qubits, circ.num_clbits)
        flat_circuit.compose(circ, inplace=True)
        circ=flat_circuit
        circ.qasm(filename="flat.qasm")
        example = "flat.qasm"
    depth = circ.depth(filter_function=lambda x: x[0].name in ['cx', 't', 'tdg'])
    num_qubits_input = extract_qubits(example)
    if include_t == 'as_is':
        gates = extract_gates_from_file(example)
        total_qubits = len(extract_qubits_from_gates(gates))
    elif include_t == 'to_cnot':
        num_t = 4*(math.ceil(math.sqrt(num_qubits_input))+1)
        ghost_ts = t_to_cnot(example, 4*(math.ceil(math.sqrt(num_qubits_input))+1), cut_other_1q=True)
        rt_ghost_ts = QuantumCircuit.from_qasm_str(ghost_ts.qasm())
        gates = extract_gates(rt_ghost_ts)
        total_qubits = rt_ghost_ts.num_qubits
    elif include_t == 'ignore':
        gates = [g for g in extract_gates_from_file(example) if len(g) == 2]
        total_qubits = len(extract_qubits_from_gates(gates))
        depth = circ.depth(filter_function=lambda x: x[0].name in ['cx'])
        print(len(gates))
    graph, removed_edges = (None, [])
    signal.signal(signal.SIGALRM, _sig_alarm)
    signal.signal(signal.SIGINT, _sig_alarm)
    signal.signal(signal.SIGTERM, _sig_alarm)
    signal.alarm(timeout)
    if arch_name == 'square_sparse_layout':
        arch = architecture.square_sparse_layout(total_qubits, magic_states="all_sides")
    elif arch_name == 'compact_layout':
        arch = architecture.compact_layout(total_qubits, magic_states='all_sides')
    try:
        mkdir(f'results_{os.path.basename(example)}/')
    except:
        pass #dir already exists
    # Optimal #
    ###########################
    if 'optimal' in methods_list:
        width = arch['width']
        height = arch['height']
        msf_faces = arch['magic_states']
        alg_qubits = arch['alg_qubits']
        start = time.time()
        (opt, model) = optimal_arb_layout.solve(gates, msf_faces, alg_qubits, width,height, start_from=depth)
        end = time.time()
        full_sol_initial_map = [m[2:4] for m in model if m[1] == 'f']
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'arch_name': arch_name, 'method': 'opt', 'steps' : opt, 'time' : end-start}))
            f.write("\n")
        # for step in range(opt):
        #     viz.draw_step(model, gates, step, msf_faces, width, height, fname=f"step {step}")
    # Opt-Route #
    ######################################
    if 'heuristic_placement_opt_route' in methods_list:
        start = time.time()
        man_map = relaxed.max_sat_build_map(gates, msf_faces, width, height, exploration_depth=exploration_depth)
        (opt, model) = optimal.solve(gates, msf_faces, width, height, fixed_map=man_map)
        end = time.time()
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'depth': depth, 'width' : width, 'height' : height, 'msf_faces' : msf_faces, 'method': f'depth {exploration_depth} heuristic placement opt route', 'steps' : opt, 'time' : end-start}))
            f.write("\n")

    if 'ensemble_placement_opt_route' in methods_list:
        start = time.time()
        for i in range(place_shots):
            map = relaxed.build_random_map(total_qubits, width*height, msf_faces)
            (opt, model) = optimal.solve(gates, msf_faces, width, height, fixed_map=dict(map))
            if i == 0 or opt < best_so_far[0]:
                best_so_far = (opt, model)
        end = time.time()
        ensemble_map =  best_so_far
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'depth': depth, 'width' : width, 'height' : height, 'msf_faces' : msf_faces, 'method': f'{place_shots}-fold ensemble placement opt route', 'steps' : ensemble_map[0], 'time' : end-start}))
            f.write("\n")
    if 'identity_placement_opt_route' in methods_list:
        targets = [i for i in range(width*height) if i not in msf_faces]
        mapping = {q : targets[q] for  q in range(total_qubits)}
        start = time.time()
        (opt, model) = optimal.solve(gates, msf_faces, width, height, fixed_map=mapping, start_from=depth)
        end = time.time()
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'depth': depth, 'width' : width, 'height' : height, 'msf_faces' : msf_faces, 'method': 'identity placement opt route', 'steps' : opt,  'time' : end-start}))
            f.write("\n")
    if 'tket_placement_opt_route' in methods_list:
        start = time.time()
        mapping = dict(relaxed.tket_place_qubit(gates, msf_faces, width, height))
        (opt, model) = optimal.solve(gates, msf_faces, width, height, fixed_map=mapping, start_from=depth)
        end = time.time()
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'depth': depth, 'width' : width, 'height' : height, 'msf_faces' : msf_faces, 'method': 'tket-style placement opt route', 'steps' : opt,  'time' : end-start}))
            f.write("\n")
    # Greedy-Route #
    ######################################
    if 'optimal placement' in methods_list:
        start = time.time()
        opt_map = relaxed.solve(gates, msf_faces, width, height, initial_map=full_sol_initial_map, exploration_depth=exploration_depth)
        end = time.time()
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'depth': depth, 'width' : width, 'height' : height, 'msf_faces' : msf_faces,  'method': 'optimal placement', 'steps' : len(opt_map), 'time' : end-start}))
            f.write("\n")
    if 'heuristic_placement' in methods_list:
        start = time.time()
        man_map = relaxed.solve(gates, msf_faces, width, height, exploration_depth=exploration_depth)
        end = time.time()
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'depth': depth, 'width' : width, 'height' : height, 'msf_faces' : msf_faces, 'method': f'depth {exploration_depth} heuristic placement', 'steps' : len(man_map), 'time' : end-start}))
            f.write("\n")
        
    if 'ensemble_placement' in methods_list:
        start = time.time()
        for i in range(place_shots):
            map = relaxed_arb_layout.build_random_map(extract_qubits_from_gates(gates),arch)
            sol = relaxed_arb_layout.solve(gates, arch, initial_map=map)
            if i == 0 or len(sol) < len(best_so_far):
                best_so_far = sol
        end = time.time()
        ensemble_map = best_so_far
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': f'{place_shots}-fold ensemble placement', 'steps' : len(ensemble_map), 'time' : end-start}))
            f.write("\n")

    if 'identity_placement' in methods_list:
        start = time.time()
        id_map = relaxed_arb_layout.solve(gates, arch, initial_map="id")
        end = time.time()
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'identity placement', 'steps' : len(id_map),  'time' : end-start}))
            f.write("\n")
        #viz.draw_step_relaxed(id_map, gates, 0, MSF_FACES, WIDTH, HEIGHT)
    if 'tket_placement' in methods_list:
        start = time.time()
        tket_map, tket_steps = relaxed_arb_layout.solve_tket_placement(gates,arch)
        end = time.time()
        verify(gates, tket_steps, arch, tket_map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input
                         , 'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'tket-style placement', 'steps' : len(tket_steps), 'time' : end-start}))
            f.write("\n")
    if 'phased_placement' in methods_list:
        start = time.time()
        capped_depth = depth
        scaled_sim_anneal_params = [sim_anneal_params[0], sim_anneal_params[1]/capped_depth, sim_anneal_params[2]/capped_depth]
        mapping_to = timeout/2
        phased_map, phased_steps =  solve_phased_placement(example, include_t, arch, mapping_to,t_aware_mapping, *scaled_sim_anneal_params)
        end = time.time()
        verify(gates, phased_steps, arch, phased_map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input
                         , 'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'phased placement', 'placement_sa_params' : sim_anneal_params, 'steps' : len(phased_steps),  'time' : end-start, 'include_t' : include_t, 't_aware_mapping': t_aware_mapping}))
            f.write("\n")
    if 'autobraid-greedy' in methods_list:
        start = time.time()
        map = autobraid_map(gates, arch).items()
        steps = relaxed_arb_layout.solve(gates, arch, initial_map= map)
        end = time.time()
        verify(gates, steps, arch, map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input
                        , 'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'autobraid-greedy', 'steps' : len(steps),  'time' : end-start}))
            f.write("\n")
    # autobraid-route #
    ######################################
    if 'autobraid' in methods_list:
        data = run_autobraid(example, num_qubits_input)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 
                         'qubits' : num_qubits_input, 
                         'depth': depth, 
                         'h_critpath' : data['h']['crit_path_cycles'],
                         'no_h_critpath' : data['no_h']['crit_path_cycles'],
                         'msf_qubit_count' :  data['h']['tqubits'], 
                         'method': 'autobraid', 
                         'h_steps' : data['h']['steps'],
                         'h_cycles' : data['h']['ab_cycles'],
                         'no_h_steps' : data['no_h']['steps'], 
                         'no_h_cycles' : data['no_h']['ab_cycles'],
                         'h_time' :  data['h']['time'],
                         'no_h_time' :  data['no_h']['time'],
                         })),

            
            f.write("\n")
    
    if 'opt-autobraid' in methods_list:
        (opt, model) = optimal.solve(gates, msf_faces,width,height, start_from=depth)
        map = [m[1:3] for m in model if m[0] == 'f' and m[3] == 0]  + [[i+num_qubits_input+1, msf_faces[i]] for i in range(len(msf_faces))]
        data = run_autobraid(example, num_qubits_input, initial_map = map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 
                         'qubits' : num_qubits_input, 
                         'depth': depth, 
                         'h_critpath' : data['h']['crit_path_cycles'],
                         'no_h_critpath' : data['no_h']['crit_path_cycles'],
                         'msf_qubit_count' :  data['h']['tqubits'], 
                         'method': 'opt-autobraid', 
                         'h_steps' : data['h']['steps'],
                         'h_cycles' : data['h']['ab_cycles'],
                         'no_h_steps' : data['no_h']['steps'], 
                         'no_h_cycles' : data['no_h']['ab_cycles'],
                         'h_time' :  data['h']['time'],
                         'no_h_time' :  data['no_h']['time'],
                         })),

            
            f.write("\n")
    if 'tket-autobraid' in methods_list:
        start = time.time()
        map = relaxed_arb_layout.tket_place_qubit(gates, arch) 
        steps = autobraid_route(gates, arch, mapping = map)
        end = time.time()
        verify(gates, steps, arch, map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input
                         , 'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'tket-autobraid', 'steps' : len(steps),  'time' : end-start}))
            f.write("\n")

    if 'phased-autobraid' in methods_list:
        capped_depth = depth
        scaled_sim_anneal_params = [sim_anneal_params[0], sim_anneal_params[1]/capped_depth, sim_anneal_params[2]/capped_depth]
        mapping_to = timeout/2
        start = time.time()
        phased_map, steps = solve_phased_placement(example, include_t, arch, mapping_to, t_aware_mapping, *scaled_sim_anneal_params, routing='autobraid')
        end = time.time()
        verify(gates, steps, arch, phased_map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input
                         ,'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 
                         'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'include_t' : include_t,
                         't_aware_mapping' : t_aware_mapping,
                         'method': 'phased-autobraid','placement_sa_params' : sim_anneal_params, 'steps' : len(steps),  'time' : end-start}))
            f.write("\n")

    if 'ensemble-autobraid' in methods_list:
        for i in range(place_shots):
            map = relaxed.build_random_map(extract_qubits_from_gates(gates), width*height, msf_faces)
            sol = relaxed.solve(gates, msf_faces, width, height, initial_map=map)
            if i == 0 or len(sol) < len(best_so_far[0]):
                best_so_far = sol,map
        ens_map = list(best_so_far[1]) + [[i+num_qubits_input, msf_faces[i]] for i in range(len(msf_faces))]
        data = run_autobraid(example, num_qubits_input, initial_map = ens_map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 
                         'qubits' : num_qubits_input, 
                         'depth': depth, 
                         'h_critpath' : data['h']['crit_path_cycles'],
                         'no_h_critpath' : data['no_h']['crit_path_cycles'],
                         'msf_qubit_count' :  data['h']['tqubits'], 
                         'method': 'tket-autobraid', 
                         'h_steps' : data['h']['steps'],
                         'h_cycles' : data['h']['ab_cycles'],
                         'no_h_steps' : data['no_h']['steps'], 
                         'no_h_cycles' : data['no_h']['ab_cycles'],
                         'h_time' :  data['h']['time'],
                         'no_h_time' :  data['no_h']['time'],
                         })),

            
            f.write("\n")
    if 'pyautobraid' in methods_list:
        start = time.time()
        map = autobraid_map(gates, arch).items()
        steps = autobraid_route(gates, arch, mapping = map)
        end = time.time()
        verify(gates, steps, arch, map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'include_t' : include_t
                        , 'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'autobraid-autobraid', 'steps' : len(steps),  'time' : end-start}))
            f.write("\n")
#    # reorder-route #
    ######################################
    if 'identity-reorder' in methods_list:
        start = time.time()
        targets = sorted(arch['alg_qubits'])
        map = [(i, targets[i]) for i in range(total_qubits)]
        steps = sim_anneal_route(gates, arch, mapping = map)
        end = time.time()
        verify(gates, steps, arch, map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'include_t' : include_t,
                        'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'identity-reorder', 'steps' : len(steps),  'time' : end-start}))
            f.write("\n")
    if 'tket-reorder' in methods_list:
        start = time.time()
        targets = sorted(arch['alg_qubits'])
        map = relaxed_arb_layout.tket_place_qubit(gates, arch)
        steps = sim_anneal_route(gates, arch, mapping = map)
        end = time.time()
        verify(gates, steps, arch, map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'include_t' : include_t,
                        'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'struct-reorder', 'steps' : len(steps),  'time' : end-start}))
            f.write("\n")

    if 'phased-reorder' in methods_list:
        capped_depth = depth
        scaled_sim_anneal_params = [sim_anneal_params[0], sim_anneal_params[1]/capped_depth, 10*sim_anneal_params[2]/capped_depth]
        if sim_anneal_params[1] == 10:
            scaled_sim_anneal_params[1] = 1
        mapping_to = timeout/2
        # trivial_map, cost = build_phased_map(extract_qubits_from_gates(gates), circ, arch, include_t = t_aware_mapping, timeout=mapping_to, initial_temp=1, cooling_rate=1,term_temp=10)
        # trivial_route = sim_anneal_route(gates, arch, trivial_map, 1,1,10,)
        try:
            start = time.time()
            phased_map, cost = build_phased_map(extract_qubits_from_gates(gates), circ, arch, include_t = t_aware_mapping, timeout=mapping_to, *scaled_sim_anneal_params)
            map_end = time.time()
            steps, orders_tried = sim_anneal_route(gates, arch, phased_map, reward_name = 'criticality', order_fraction=order_fraction, take_first_ms=take_first_ms, *sim_anneal_params_r)
            end = time.time()
            verify(gates, steps, arch, phased_map)
            with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
                f.write(str({ 'circ' : example, 'qubits' : num_qubits_input, 'include_t' : include_t, 'take_first_ms' : take_first_ms
                            , 'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 
                              'arch_name' : arch_name, 'method': 'phased-reorder', 'placement_sa_params' : sim_anneal_params, 
                              'r_orders_tried' : orders_tried,
                              'routing_sa_params' : sim_anneal_params_r, 'steps' : len(steps), 't_aware_mapping' : t_aware_mapping,  'map_time' : map_end-start, 'route_time' : end-map_end, 'time' : end-start, 'order_fraction' : order_fraction}))
                f.write("\n")
        except TimeoutError:
            with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
                f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'include_t' : include_t,
                            'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 
                            'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'phased-reorder', 
                            'placement_sa_params' : sim_anneal_params, 'routing_sa_params' : sim_anneal_params_r, 'steps' : len(trivial_route), 't_aware_mapping' : t_aware_mapping, 'time' : "TIMEOUT", 'order_fraction' : order_fraction}))
                f.write("\n")
    if 'autobraid-reorder' in methods_list:
        start = time.time()
        map = autobraid_map(gates, arch).items()
        steps = sim_anneal_route(gates, arch, map,  *sim_anneal_params_r)
        end = time.time()
        verify(gates, steps, arch, map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input, 'include_t' : include_t,
                        'depth': depth, 'width' : arch['width'], 'height' : arch['height'], 'msf_faces' : arch['magic_states'], 'arch_name' : arch_name, 'method': 'autobraid-reorder', 'steps' : len(steps),  'time' : end-start}))
            f.write("\n")
# debug
    if 'compare-routes' in methods_list:
        capped_depth = depth
        scaled_sim_anneal_params = [sim_anneal_params[0], sim_anneal_params[1]/capped_depth, 10*sim_anneal_params[2]/capped_depth]
        mapping_to = timeout/2
        start = time.time()
        phased_map, cost = build_phased_map(num_qubits_input, circ, arch, include_t = t_aware_mapping, timeout=timeout, *scaled_sim_anneal_params)
        auto_steps = autobraid_route(gates, arch, mapping = phased_map)
        verify(gates, auto_steps, arch, phased_map)
        end = time.time()
        print("auto done")
        reorder_naive = sim_anneal_route(gates, arch, mapping = phased_map, reward_name='gates_routed')
        print("gates_routed done")
        verify(gates, reorder_naive, arch, phased_map)
        reorder_crit = sim_anneal_route(gates, arch, mapping = phased_map, reward_name='criticality')
        print("crit done")
        reorder_dep = sim_anneal_route(gates, arch, mapping = phased_map, reward_name='dependent')
        print("dep done")
        verify(gates, reorder_dep, arch, phased_map)
        with open(f'results_{os.path.basename(example)}/{methods_list[0]}.{rid}.txt', 'w+') as f:
            f.write(str({'circ' : example, 'qubits' : num_qubits_input,
                         'depth': depth, 'width' : arch['width'], 'height' : arch['height'],  'arch_name' : arch_name, 'placement_sa_params' : sim_anneal_params, 
                         'gates-routed-steps' : len(reorder_naive), 
                         'crit-steps' : len(reorder_crit),
                         'dep-steps' : len(reorder_dep),
                         't_aware_mapping' : t_aware_mapping,
                         'auto-steps' : len(auto_steps), 'time' : end-start}))
            f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('circ', help='path to input circuit')
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--arch', choices=['square_sparse_layout', 'compact_layout'])
    parser.add_argument('--method', choices= [
                                              'optimal', 
                                              'optimal_placement', 
                                              'heuristic_placement', 
                                              'ensemble_placement', 
                                              'phased_placement', 
                                              'identity_placement',
                                              'identity_placement_opt_route', 
                                              'tket_placement', 
                                              'tket_placement_opt_route',
                                              'heuristic_placement_opt_route', 
                                              'ensemble_placement_opt_route', 
                                              'autobraid', 
                                              'opt-autobraid' , 
                                              'tket-autobraid', 
                                              'ensemble-autobraid', 
                                              'phased-autobraid',
                                              'phased-reorder',
                                              'pyautobraid',
                                              'identity-reorder',
                                              'tket-reorder',
                                              'autobraid-reorder',
                                              'autobraid-greedy',
                                              'compare-routes'
                                            ])
    parser.add_argument('--ensemble_size',type=int, default=20)
    parser.add_argument('--sim_anneal_params',type=ast.literal_eval, default=[100,0.1,0.1])
    parser.add_argument('--sim_anneal_params_r',type=ast.literal_eval, default=[10,0.1,0.1])
    parser.add_argument('--exp_depth',type=int,)
    parser.add_argument('--order_fraction',type=float,default=1.0)
    parser.add_argument('--include_t', choices=['as_is', 'ignore', 'to_cnot'], default='as_is')
    parser.add_argument('--t_aware_mapping', action="store_true")
    parser.add_argument('--take_first_ms', action="store_true")
    parser.add_argument("-to", "--timeout", type=int, default=3600, help="maximum run time for an algorithm in seconds")
    parser.add_argument("-rid", "--run_id", help="ID to uniquely identify this run (if necessary)")
    args = parser.parse_args()
    os.makedirs(f'results_{os.path.basename(args.circ)}/', exist_ok=True)

    run_methods(
        width=args.width, 
        height=args.height, 
        msf_faces=all_sides(args.width, args.height), 
        arch_name=args.arch, 
        example=args.circ, methods_list=[args.method], 
        place_shots=args.ensemble_size, 
        sim_anneal_params = args.sim_anneal_params,
        sim_anneal_params_r = args.sim_anneal_params_r,
        exploration_depth=args.exp_depth if args.exp_depth else None, 
        rid=args.run_id, 
        timeout=args.timeout, 
        include_t=args.include_t, 
        t_aware_mapping=args.t_aware_mapping,
        take_first_ms=args.take_first_ms, 
        order_fraction=args.order_fraction
        )
    # except Exception as e:
        # signal.alarm(0)
        # print(e)
        # with open(f'results_{os.path.basename(args.circ)}/{args.method}.{args.run_id}.txt', 'w+') as f:
        #     f.write(str({'circ' : args.circ, 'method': args.method, 'steps' : math.inf,  'time' : 'timeout', 'exception' : e}))
        #     f.write("\n")