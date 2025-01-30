from pysat.card import *
import numpy as np
from pysat.solvers import Solver


def gate_has_time_step(gate_num, log_num, step_num, sem_vars, aux_vars, s):
    for i in range(gate_num):
        lits = [to_int(sem_vars, (False, "e", i, j)) for j in range(step_num)]
        for clause in CardEnc.equals(lits=lits, bound=1, vpool=aux_vars).clauses:
            s.add_clause(clause)

def maps_are_injective(grid_len, grid_height, gate_num, log_num, step_num, msf_faces, alg_qubits, sem_vars, aux_vars, s):
    face_num = grid_len*grid_height
    print(alg_qubits)
    for k in range(step_num):
        for i in range(log_num):
            lits = [to_int(sem_vars, (False, "f",i, p, k)) for p in alg_qubits]
            for clause in CardEnc.equals(lits, 1, vpool=aux_vars).clauses:
                s.add_clause(clause)
            for j in range(face_num):
                if j not in alg_qubits:
                    s.add_clause([to_int(sem_vars, (True, "f", i, j, k))])
        for j in alg_qubits:
            lits = [to_int(sem_vars, (False, "f",q, j, k)) for q in range(log_num)]
            for clause in CardEnc.atmost(lits, 1, vpool=aux_vars).clauses:
                s.add_clause(clause)
                
def map_is_given(map_dict, sem_vars, s):
    print(map_dict)
    for q, p in map_dict.items():
        s.add_clause([to_int(sem_vars, (False, "f", q, p, 0))])

def magic_states_preserved(grid_len, grid_height, gate_num, log_num, step_num, msf_faces, sem_vars, aux_vars, s):
    clauses =[[to_int(sem_vars,(True, "l", f,u, g, k))] for  f in msf_faces for g in range(gate_num) for k in range(step_num) for u in neighbors(f, grid_len, grid_height, omitted_edges=[])] 
    for clause in clauses:
        s.add_clause(clause)

def data_preserved(grid_len, grid_height, gate_num, log_num, step_num, msf_faces, sem_vars, aux_vars, s):
    clauses =[[to_int(sem_vars, (True, "l", f,u, g, k)), to_int(sem_vars, (True, "l", v,f, g, k)), to_int(sem_vars, (True, "f", q,f, k))] for  f in range(grid_len*grid_height) for g in range(gate_num) for k in range(step_num) for q in range(log_num) for u in neighbors(f, grid_len, grid_height, omitted_edges=[]) for v in neighbors(f, grid_len, grid_height, omitted_edges=[])] 
    for clause in clauses:
        s.add_clause(clause)

# this just imposes a fixed mapping for now
def swap_effect_constraint(grid_len, grid_height, gate_num, log_num, step_num, sem_vars, s):
    face_num = grid_len * grid_height   
    for k in range(step_num-1):
        for i in range(log_num):
            for j in range(face_num):
                for j2 in range(face_num):
                    if j != j2:
                        clause = [(True, 'f', i, j, k),(True, 'f', i, j2, k+1)]
                        flattened_clause = [to_int(sem_vars, lit) for lit in clause]
                        s.add_clause(flattened_clause)

# def swap_effect_constraint(grid_len, grid_height, gate_num, log_num, step_num, node_num):
#     clauses = []
#     face_num = grid_len * grid_height   
#     for k in range(step_num-1):
#         for i in range(log_num):
#             for j in range(face_num):
#                     clause = [(True, 'f', i, j, k),(False, 'f', i, j, k+1)]
#                     flattened_clause = [flattenedIndex(lit,  gate_num, log_num, step_num, node_num) for lit in clause]
#                     clauses.append(flattened_clause)
#     return clauses


def dependencies_respected(edge_list, gate_num, log_num, step_num, sem_vars, s):
    clauses = []
    for k in range(step_num):
        for edge in edge_list:
            clause = [(True, "e", edge[1], k)] + [(False, 'e', edge[0], k1) for k1 in range(k)]
            flattened_clause = [to_int(sem_vars, lit) for lit in clause]
            s.add_clause(flattened_clause)

def braids_nonintersecting(grid_len, grid_height, gate_num, log_num, step_num, sem_vars, aux_vars, s):
    face_num = grid_len * grid_height  
    for n in range(face_num):
        for k in range(step_num):
            lits = [to_int(sem_vars, (False, "b",n, i, k)) for i in range(gate_num)]
            for clause in CardEnc.atmost(lits, 1, vpool=aux_vars).clauses:
                s.add_clause(clause)

def edges_match_colors(grid_len, grid_height, omitted_edges, gate_num, log_num, step_num, sem_vars, s):
    face_num = grid_len*grid_height
    for j in range(face_num):
        for k in range(step_num):
            for g in range(gate_num):
              for n in neighbors(j, grid_len, grid_height, omitted_edges):
                clause = [(False, 'b', n, g, k), (True, 'l', n, j, g, k)]
                flattened_clause = [to_int(sem_vars, lit) for lit in clause]
                s.add_clause(flattened_clause)
                clause = [(False, 'b', j, g, k), (True, 'l', n, j, g, k)]
                flattened_clause = [to_int(sem_vars, lit) for lit in clause]
                s.add_clause(flattened_clause)
                clause = [(True, 'l', j, n, g, k), (True, 'l', n, j, g, k)]
                flattened_clause = [to_int(sem_vars, lit) for lit in clause]
                s.add_clause(flattened_clause)



def path_control_target(grid_len, grid_height, omitted_edges, gate_list, gate_num, log_num, step_num, msf_faces, sem_vars, aux_vars, s):
    clauses = []
    face_num = grid_len * grid_height  
    for g in range(len(gate_list)):
        if g >= len(gate_list):
            # index = g-len(gate_list)
            # c = index // face_num
            # t = index % face_num
            # for k in range(step_num):
            #     lits_1 =  [to_int(sem_vars, (False, 'l', n, t, g, k)) for n in neighbors(t, grid_len+1, grid_height+1, omitted_edges)]
            #     exact_1 = CardEnc.equals(lits_1, 1, vpool=aux_vars)
            #     for clause in exact_1.clauses:
            #         clause.append(to_int(sem_vars, (True, 'e', g, k)))
            #         s.add_clause(clause)
            #     no_out =  [to_int(sem_vars, (True, 'b', t,  g, k))] 
            #     s.add_clause(no_out)
            #     for j in range(node_num):
            #         lits_in = [to_int(sem_vars, (False, "l", n, j, g, k)) for n in neighbors(j, grid_len+1, grid_height+1, omitted_edges)] 
            #         exact_in = CardEnc.equals(lits_in,1, vpool=aux_vars)
            #         lits_out = [to_int(sem_vars, (False, "l", j, n, g, k)) for n in neighbors(j, grid_len+1, grid_height+1, omitted_edges)] 
            #         exact_out = CardEnc.equals(lits_out,1, vpool=aux_vars)
            #         for clause in exact_in.clauses:
            #             if j != c:
            #                 clause.append(to_int(sem_vars, (True, 'b', j, g, k)))
            #                 s.add_clause(clause)
            #         for clause in exact_out.clauses:
            #                 clause.append(to_int(sem_vars, (True, 'b', j, g, k)))
            #                 s.add_clause(clause)
            raise NotImplementedError 
        elif len(gate_list[g]) == 2:
            c, t = gate_list[g]
            for k in range(step_num):
                for j in range(face_num):
                    lits_1 =  [to_int(sem_vars, (False, 'l', n, j, g, k)) for n in horizontal_neighbors(j, grid_len, grid_height, omitted_edges)]
                    exact_1 = CardEnc.equals(lits_1, 1, vpool=aux_vars)
                    for clause in exact_1.clauses:
                        clause.append(to_int(sem_vars, (True, 'e', g, k)))
                        clause.append(to_int(sem_vars, (True, "f", t, j, k)))
                        s.add_clause(clause)
                    lits_1_ctrl =  [to_int(sem_vars, (False, 'l', j, n, g, k)) for n in vertical_neighbors(j, grid_len, grid_height, omitted_edges)]
                    exact_1_ctrl = CardEnc.equals(lits_1_ctrl, 1, vpool=aux_vars)
                    for clause in exact_1_ctrl.clauses:
                        clause.append(to_int(sem_vars, (True, 'e', g, k)))
                        clause.append(to_int(sem_vars, (True, "f", c, j, k)))
                        s.add_clause(clause)
                    for n in neighbors(j, grid_len, grid_height, omitted_edges):
                        no_out =  [to_int(sem_vars, (True, 'l', j, n,  g, k)), to_int(sem_vars, (True, "f", t, j, k))] 
                        s.add_clause(no_out)
                    lits_in = [to_int(sem_vars, (False, "l", n, j, g, k)) for n in neighbors(j, grid_len, grid_height, omitted_edges)] 
                    exact_in = CardEnc.equals(lits_in,1, vpool=aux_vars)
                    lits_out = [to_int(sem_vars, (False, "l", j, n, g, k)) for n in neighbors(j, grid_len, grid_height, omitted_edges)] 
                    exact_out = CardEnc.equals(lits_out,1, vpool=aux_vars)
                    for clause in exact_in.clauses:
                        clause.append(to_int(sem_vars, (True, 'b', j, g, k)))
                        clause.append(to_int(sem_vars, (False, 'f', c, j, k)))
                        #clause.append(to_int(sem_vars, (True, 'e', g, k),  gate_num, log_num, step_num, node_num))
                        s.add_clause(clause)
                    for clause in exact_out.clauses:
                        clause.append(to_int(sem_vars, (True, 'b', j, g, k)))
                        clause.append(to_int(sem_vars, (False, 'f', t, j, k)))
                        s.add_clause(clause)

        else:
            t = gate_list[g][0]
            for k in range(step_num):
                for j in range(face_num):
                    lits_1 =  [to_int(sem_vars, (False, 'l', j, n, g, k)) for n in vertical_neighbors(j, grid_len, grid_height, omitted_edges)]
                    exact_1 = CardEnc.equals(lits_1, 1, vpool=aux_vars)
                    for clause in exact_1.clauses:
                        clause.append(to_int(sem_vars, (True, 'e', g, k)))
                        clause.append(to_int(sem_vars, (True, "f", t, j, k)))
                        s.add_clause(clause)
                    lits_1_msf =  [to_int(sem_vars, (False, 'l', n, j, g, k)) for m in msf_faces for n in horizontal_neighbors(m, grid_len, grid_height, omitted_edges)]
                    exact_1_msf = CardEnc.equals(lits_1_msf, 1, vpool=aux_vars)
                    for clause in exact_1_msf.clauses:
                        clause.append(to_int(sem_vars, (True, 'e', g, k)))
                        s.add_clause(clause)
                    lits_in = [to_int(sem_vars, (False, "l", n, j, g, k)) for n in neighbors(j, grid_len, grid_height, omitted_edges)] 
                    exact_in = CardEnc.equals(lits_in,1, vpool=aux_vars)
                    lits_out = [to_int(sem_vars, (False, "l", j, n, g, k)) for n in neighbors(j, grid_len, grid_height, omitted_edges)] 
                    exact_out = CardEnc.equals(lits_out,1, vpool=aux_vars)
                    for clause in exact_in.clauses:
                            clause.append(to_int(sem_vars, (True, 'b', j, g, k)))
                            clause.append(to_int(sem_vars, (False, 'f', t, j, k)))
                            #clause.append(to_int(sem_vars, (True, 'e', g, k),  gate_num, log_num, step_num, node_num))
                            s.add_clause(clause)
                    for clause in exact_out.clauses:
                        if j not in msf_faces:
                            clause.append(to_int(sem_vars, (True, 'b', j, g, k)))
                            s.add_clause(clause)


def bandwidth_constraint( gate_list, gate_num, log_num, step_num, bandwidth, sem_vars, aux_vars,s):
    if bandwidth:
        t_indices = [i for i in range(len(gate_list)) if len(gate_list[i]) == 1]
        for k in range(step_num):
            lits = [to_int(sem_vars, (False, "e", t, k)) for t in t_indices ]
            for clause in CardEnc.atmost(lits, bandwidth, vpool=aux_vars).clauses:
                s.add_clause(clause)

def vertical_neighbors(n, grid_len, grid_height, omitted_edges):
    neighbors = []
    down = n + grid_len
    up = n - grid_len
    if n // grid_len != 0 and (n,up) not in omitted_edges and (up, n) not in omitted_edges:
        neighbors.append(up)
    if n // grid_len != grid_height-1 and (n,down) not in omitted_edges and (down,n) not in omitted_edges:
        neighbors.append(down)
    return neighbors

def horizontal_neighbors(n, grid_len, grid_height, omitted_edges):
    neighbors = []
    left = n - 1
    right = n + 1
    if n % grid_len != 0 and (n,left) not in omitted_edges and (left,n) not in omitted_edges:
        neighbors.append(left)
    if n % grid_len != grid_len-1 and (n,right) not in omitted_edges and (right,n) not in omitted_edges:
        neighbors.append(right)
    return neighbors


def neighbors(n, grid_len, grid_height, omitted_edges):
    return horizontal_neighbors(n, grid_len, grid_height, omitted_edges) + vertical_neighbors(n, grid_len, grid_height, omitted_edges)

def edge_list_from_gate_list(gate_list):
    edge_list = []
    qubit_depth = {}
    for i in range(len(gate_list)): 
        for qubit in gate_list[i]:
            if qubit in qubit_depth:
                edge_list.append((qubit_depth[qubit], i))
            qubit_depth[qubit] = i
    return edge_list

def extract_qubits(gate_list):
    qubits = set()
    for gate in gate_list:
        for qubit in gate:
            qubits.add(qubit)
    return qubits

def solve_k(grid_len, grid_height, msf_faces, alg_qubits, omitted_edges, gate_list, step_num, fixed_map, bandwidth):
    gate_num = len(gate_list)
    log_num = len(extract_qubits(gate_list))
    face_num = grid_len*grid_height
    # mappable_faces = [p for p in range(face_num) if p not in msf_faces]
    #map_face_num = len(mappable_faces)
    num_E = (gate_num)*step_num
    num_B = face_num*(gate_num)*step_num
    num_F  = log_num*face_num*step_num
    num_L = face_num*face_num*(gate_num)*step_num
    top = num_E + num_B + num_F + num_L
    edge_list = edge_list_from_gate_list(gate_list)
    aux_vars = IDPool(start_from=top+1)
    vpool = IDPool()
    with Solver(name='cd') as s:
        gate_has_time_step(gate_num, log_num, step_num, vpool, aux_vars, s)
        if fixed_map: 
            map_is_given(fixed_map, vpool, s) 
        maps_are_injective(grid_len, grid_height, gate_num, log_num, step_num,  msf_faces, alg_qubits, vpool, aux_vars,  s)
        data_preserved(grid_len, grid_height, gate_num, log_num, step_num,  msf_faces, vpool, aux_vars,  s)
        magic_states_preserved(grid_len, grid_height, gate_num, log_num, step_num,  msf_faces, vpool, aux_vars,  s)
        dependencies_respected(edge_list, gate_num, log_num, step_num, vpool,  s) 
        braids_nonintersecting(grid_len, grid_height, gate_num, log_num, step_num, vpool, aux_vars,  s) 
        path_control_target(grid_len, grid_height, omitted_edges, gate_list, gate_num, log_num, step_num, msf_faces, vpool, aux_vars, s)
        bandwidth_constraint(gate_list, gate_num, log_num, step_num, bandwidth, vpool, aux_vars, s)
        swap_effect_constraint(grid_len, grid_height, gate_num, log_num, step_num, vpool, s)
        edges_match_colors(grid_len, grid_height, omitted_edges, gate_num, log_num, step_num, vpool, s)
        (s.nof_clauses())
        s.solve()
        if s.get_status():
            model=([vpool.obj(lit) for lit in s.get_model() if  vpool.obj((lit))])
            verify(model, grid_len, grid_height, msf_faces, gate_list, step_num)
            return model
        return s.get_status()

def solve(gates, msf_faces, alg_qubits, grid_len, grid_height, omitted_edges=[], fixed_map=None, bandwidth=None, start_from=1):
    solved = False
    step_num = start_from-1
    while not solved:
        step_num += 1
        print(step_num)
        if step_num > len(gates):
            print('no sol') 
            return(-1,"no solution")
        solved = solve_k(grid_len, grid_height, msf_faces, alg_qubits, omitted_edges, gates, step_num, fixed_map, bandwidth)
    return (step_num, solved)

def solve_parallel(gates, msf_faces, grid_len, grid_height, fname, bandwidth=None, start_from=None, return_code=None, run=None):
    solved = False
    step_num = start_from-1
    while run.is_set():
        while step_num <= len(gates):
            step_num += 1
            solved = solve_k(grid_len, grid_height, msf_faces, gates, step_num, fname, bandwidth)
            if solved:
                example = ''.join(fname.split("_")[:-1])
                if return_code.get(example):
                    old = return_code[example]
                    return_code[example] = min(old, step_num)
                else:
                    return_code[example] = step_num
                run.clear()
    return step_num

def to_int(vpool, lit):
    if lit[0]:
        return -vpool.id(lit[1:])
    else:
        return vpool.id(lit[1:])

    

def writeClause(f, clause):
        f.write(" ")
        for lit in clause:
            f.write(str(lit))
            f.write(" ")
        f.write("0\n")

def verify(model, grid_len, grid_height, msf_faces, gate_list, k):
    gate_to_step = {}
    face_num = (grid_len)*(grid_height)
    mappable_faces = [p for p in range(face_num) if p not in msf_faces]
    qubits = extract_qubits(gate_list)
    for q in qubits:
        image = {v for v in model if v[0] == 'f' and v[1]==q and v[3]==0}
        assert(len(image) == 1)
    for f in mappable_faces:
        preimage =  {v for v in model if v[0] == 'f' and v[2]==f and v[3]==0}
        assert(len(preimage) <= 1)
    for g in range(len(gate_list)):
        executed = False
        for i in range(k):
            if ('e', g , i) in model:
                gate_to_step[g] = i
                
                executed = True
        assert(executed)
        if len(gate_list[g]) == 2:
            path =  {v for v in model if v[0] == 'l' and v[3] == g and v[4] == gate_to_step[g]}
            m_ctrl =  {v for v in model if v[0] == 'f' and v[1] == gate_list[g][0] and v[3] == gate_to_step[g]}
            assert(len(m_ctrl) == 1)
            m_tar =  {v for v in model if v[0] == 'f' and v[1] == gate_list[g][1] and v[3] == gate_to_step[g]}
            assert(len(m_tar) == 1)
            edges_only = {(v[1],v[2]) for v in path}
            ctrl = m_ctrl.pop()[2]
            tar = m_tar.pop()[2]
            p = reachable_from(ctrl, edges_only)
            assert(tar in {vertex for edge in p for vertex in edge} )
            for u,v in p:
                assert(abs(u-v) == 1 or abs(u-v) == grid_len)
    deps = edge_list_from_gate_list(gate_list)
    steps = {}
    for key, value in gate_to_step.items():
        steps.setdefault(value, []).append(gate_list[key])
    print(steps)
    for (i,j) in deps:
        assert(gate_to_step[i] < gate_to_step[j])
    for step in range(k):
        for j in range((grid_len)*(grid_height+1)):
            vertex_set = {v for v in model if v[0] == 'b' and v[1] == j and v[3] == step}
            assert(len(vertex_set) <= 1)

def reachable_from(start, edge_list):
    edge_dict = {u : v for u,v in edge_list}
    path = []
    current = start
    while current in edge_dict.keys():
        path.append((current, edge_dict[current]))
        current =  edge_dict[current]
    return path