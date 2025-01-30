
import json
import re


def extract_gates_from_file(fname):
    gates = []
    ops = []
    with open(fname) as f:
        for line in f:
            match = re.match(r'(cx)\s+q\[(\d+)\],\s*q\[(\d+)\];', line)
            if match:
                gates.append((int(match.group(2)), int(match.group(3))))
                ops.append(match.group(1))
            match = re.match(r'(t|tdg)\s+q\[(\d+)\];', line)
            if match:
                ops.append(match.group(1))
                gates.append((int(match.group(2)),))
    return gates,ops


def extract_qubits_from_gates(gate_list):
    qubits = set()
    for gate in gate_list:
        for qubit in gate:
            qubits.add(qubit)
    return qubits

def dump(map, steps, id_to_op, output_path):
    output = {}
    output['map'] = {k :v for k,v in map}
    output['steps'] = [label_step(id_to_op, step) for step in steps]
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

def label_step(id_to_op, step):
    return [labeled_gate_path(id_to_op, *gate_path) for gate_path in step]

def labeled_gate_path(id_to_op, id, args, path):
    dict = {}
    dict['id'] = id
    dict['op'] = id_to_op[id]
    dict['qubits'] = args
    dict['path'] = path
    return dict