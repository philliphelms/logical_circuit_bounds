from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn

# Operator measurement
def measure_op(ket, bra, opdict, draw=False):
    # Copy the supplied mps
    bra = bra.copy()
    ket = ket.copy()

    # Prepare ops and state for contraction
    for key in opdict:
        opdict[key] = qtn.Tensor(opdict[key],
                                 inds = [f'k{key}', f'b{key}'])
        bra = bra.reindex(dict(zip([f'k{key}'], [f'b{key}'])))

    # Create the expectation value
    expectation = bra & ket
    for key in opdict:
        expectation = expectation & opdict[key]

    # Evaluate the expectation value
    if draw:
        expectation.draw()
    return (expectation ^ all)

def nsite_operator(stater, statel, sites, ops, draw=False):
    opdict = dict(zip(sites, ops))
    return measure_op(stater, statel, opdict, draw=draw)

def get_ops(d):
    ops = dict()

    # Identity
    ops['I'] = np.eye(d)

    # Occupation operators
    for i in range(d):
        op = np.zeros((d, d))
        op[i,i] = 1
        ops['n'+str(i)] = op

    # Transfer operators
    for i in range(d):
        for j in range(d):
            if i != j:
                op = np.zeros((d, d))
                op[i, j] = 1
                ops['t'+str(i)+str(j)] = op

    # Return result
    return ops

# Function to evaluate densities
def evaluate_gate_densities(ket):
    # Figure out how many elements there are in each gate
    ngate_states = ket.ind_size('k2')

    # Create the |-> vectore
    bra = [np.array([[1, 1]]),
           np.array([[[1, 1]]]),
           np.array([[[1]*ngate_states]]),
           np.array([[[1, 1]]]),
           np.array([[[1, 1]]]),
           np.array([[1]*ngate_states]),
           np.array([[[1, 1]]]),
           np.array([[[1, 1]]]),
           np.array([[1]*ngate_states])]
    bra = qtn.MatrixProductState(bra)

    # Normalize Everything
    norm = bra & ket
    norm = norm ^ all
    ket /= norm

    # Measure the operators
    ops = get_ops(ngate_states)
    gate1_occ = np.zeros(ngate_states)
    gate2_occ = np.zeros(ngate_states)
    gate3_occ = np.zeros(ngate_states)
    for gate_state_ind in np.arange(ngate_states):
        opi = ops[f'n{gate_state_ind}']
        gate1_occ[gate_state_ind] = nsite_operator(bra, ket, [2], [opi]).real
        gate2_occ[gate_state_ind] = nsite_operator(bra, ket, [5], [opi]).real
        gate3_occ[gate_state_ind] = nsite_operator(bra, ket, [8], [opi]).real
    return gate1_occ, gate2_occ, gate3_occ
