import math


def insert_row_above(arch):
    new = arch.copy()
    new['height'] = arch['height']+1
    new['alg_qubits'] = [q+new['width'] for q in arch['alg_qubits']]
    new['magic_states'] = [q+new['width'] for q in arch['magic_states']]
    return new

def insert_row_below(arch):
    new = arch.copy()
    new['height'] = arch['height']+1
    return new

def insert_column_left(arch):
    new = arch.copy()
    new['width'] = arch['width']+1
    new['alg_qubits'] = []
    new['magic_states'] = []
    for q in arch['alg_qubits']:
        # need to count first row
        row = q // arch['width']+1
        new['alg_qubits'].append(q+row)
    for m in arch['alg_qubits']:
        # need to count first row
        row = m // arch['width']+1
        new['magic_states'].append(q+row)
    return new

def insert_column_right(arch):
    new = arch.copy()
    new['width'] = arch['width']+1
    new['alg_qubits'] = []
    new['magic_states'] = []
    for q in arch['alg_qubits']:
        # now don't coount the one in my row
        row = q // arch['width']
        new['alg_qubits'].append(q+row)
    for m in arch['alg_qubits']:
        # don't coujnt my row
        row = m // arch['width']
        new['magic_states'].append(q+row)
    return new

def center_column(width, height):
    return [(width*i)+(width//2) for i in range(height)]

def right_column(width, height):
    return [(width*i)+(width-1) for i in range(height)]

def all_sides(width, height):

    left_column = [(width*i) for i in range(height)]
    right_column =  [(width*i)+(width-1) for i in range(height)]
    top_row = [i for i in range(width)]
    bottom_row = [(width)*(height-1) + i for i in range(width)]
    all_slots =  list(dict.fromkeys(top_row + right_column + list(reversed(bottom_row))+left_column))
    msf = []
    for i in range(1,len(all_slots),2):
        msf.append(all_slots[i])
    return msf

def square_sparse_layout(alg_qubit_count, magic_states):
    grid_len = 2*math.ceil(math.sqrt(alg_qubit_count))+1
    grid_height = grid_len
    for_circ = []
    for i in range(grid_height*grid_len):
       x,y = reversed(divmod(i, grid_len))
       if x % 2 == y % 2 == 1:
            for_circ.append(i)
    arch = {"height" : grid_height, "width" : grid_len, "alg_qubits" : for_circ, "magic_states" : [] }
    if magic_states == 'all_sides':
        arch = insert_row_below(insert_row_above(insert_column_right(insert_column_left(arch))))
        msf_faces = all_sides(arch['width'], arch['height'])
        arch['magic_states'] = msf_faces
    elif magic_states == "center_column":
        msf_faces = center_column(grid_len, grid_height)
    elif magic_states == 'right_column':
        msf_faces = right_column(grid_len, grid_height)
    else: msf_faces = magic_states
    return arch

def compact_layout(alg_qubit_count, magic_states):
    grid_height = 3
    grid_len = (2*(math.ceil(alg_qubit_count/2))-1)
    for_circ = []
    for i in range(0,grid_len,2):
        for_circ.append(i)
        for_circ.append((grid_len)*2 + i )
    arch = {"height" : grid_height, "width" : grid_len, "alg_qubits" : for_circ, "magic_states" : [] }
    if magic_states == 'all_sides':
        arch = insert_row_below(insert_row_above(insert_column_right(insert_column_left(arch))))
        msf_faces = all_sides(arch['width'], arch['height'])
        arch['magic_states'] = msf_faces
    return arch

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