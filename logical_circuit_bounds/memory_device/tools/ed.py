import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, expm
from scipy.sparse import csc_matrix
import quimb.tensor as qtn
from logical_circuit_bounds.not_gate.tools.op_tools import *

def get_generator(params, gate1_vals, gate2_vals,
                  absorbing_inds1 = [], absorbing_inds2 = [],
                  square=True, sparse=True):
    # -----------------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------------
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Cg = params['Cg']

    Ngate1 = len(gate1_vals)
    Ngate2 = len(gate2_vals)

    # -----------------------------------------------------------
    # GENERATOR
    # -----------------------------------------------------------
    print('Populating Generator')
    left_inds = []
    right_inds = []
    matrix_entries = []

    # Put each term in the generator

    # N1 -> S1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for P1i in range(2):
                P1j = P1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (0, P1i, G1i, N2i, P2i, G2i) )
                            right_inds.append( (1, P1j, G1j, N2j, P2j, G2j) )
                            matrix_entries.append( gamma / (np.exp(beta*(-1.5*Vd-G2/Cg))+1) )

    # S1 -> N1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for P1i in range(2):
                P1j = P1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (1, P1i, G1i, N2i, P2i, G2i) )
                            right_inds.append( (0, P1j, G1j, N2j, P2j, G2j) )
                            matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G2/Cg))+1)) )

    # N1 -> P1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N2i in range(2):
                N2j = N2i
                for P2i in range(2):
                    P2j = P2i
                    # Put in matrix element
                    if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                        left_inds.append(  (0, 1, G1i, N2i, P2i, G2i) )
                        right_inds.append( (1, 0, G1j, N2j, P2j, G2j) )
                        matrix_entries.append( gamma * (1. / (np.exp(beta*abs(-1.5*Vd-G2/Cg-G2/Cg))-1.+1e-6)) )

    # P1 -> N1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N2i in range(2):
                N2j = N2i
                for P2i in range(2):
                    P2j = P2i
                    # Put in matrix element
                    if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                        left_inds.append(  (1, 0, G1i, N2i, P2i, G2i) )
                        right_inds.append( (0, 1, G1j, N2j, P2j, G2j) )
                        matrix_entries.append( gamma * (1. + 1. / (np.exp(beta*abs(-1.5*Vd-G2/Cg-G2/Cg))-1.+1e-6)) )

    # P1 -> D1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, 0, G1i, N2i, P2i, G2i) )
                            right_inds.append( (N1j, 1, G1j, N2j, P2j, G2j) )
                            matrix_entries.append( gamma / (np.exp(beta*(G2/Cg-Vd))+1) )
    # D1 -> P1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, 1, G1i, N2i, P2i, G2i) )
                            right_inds.append( (N1j, 0, G1j, N2j, P2j, G2j) )
                            matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(G2/Cg-Vd))+1)) )

    # P1 -> G1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1j, G1 in enumerate(gate1_vals[:-1]):
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, 0, G1j+1, N2i, P2i, G2i) )
                            right_inds.append( (N1j, 1, G1j  , N2j, P2j, G2j) )
                            matrix_entries.append( gamma * ( 1. / (np.exp(beta*(G2/Cg+G1/Cg))+1)) )

    # G1 -> P1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1j, G1 in enumerate(gate1_vals[:-1]):
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j+1 not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, 1, G1j  , N2i, P2i, G2i) )
                            right_inds.append( (N1j, 0, G1j+1, N2j, P2j, G2j) )
                            matrix_entries.append( gamma * ( 1. - 1. / (np.exp(beta*(G2/Cg+gate1_vals[G1j+1]/Cg))+1)) )

    # N1 -> G1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1j, G1 in enumerate(gate1_vals[:-1]):
            for P1i in range(2):
                P1j = P1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (0, P1i, G1j+1, N2i, P2i, G2i) )
                            right_inds.append( (1, P1j, G1j  , N2j, P2j, G2j) )
                            matrix_entries.append( gamma * ( 1. / (np.exp(beta*(-1.5*Vd-G2/Cg+G1/Cg))+1)) )

    # G1 -> N1
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1j, G1 in enumerate(gate1_vals[:-1]):
            for P1i in range(2):
                P1j = P1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j+1 not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (1, P1i, G1j  , N2i, P2i, G2i) )
                            right_inds.append( (0, P1j, G1j+1, N2j, P2j, G2j) )
                            matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G2/Cg+gate1_vals[G1j+1]/Cg))+1)) )

    # N2 -> S2
    for G1i, G1 in enumerate(gate1_vals):
        G1j = G1i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for P1i in range(2):
                P1j = P1i
                for N1i in range(2):
                    N1j = N1i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, P1i, G1i, 0, P2i, G2i) )
                            right_inds.append( (N1j, P1j, G1j, 1, P2j, G2j) )
                            matrix_entries.append( gamma / (np.exp(beta*(-1.5*Vd-G1/Cg))+1) )

    # S2 -> N2
    for G1i, G1 in enumerate(gate1_vals):
        G1j = G1i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for P1i in range(2):
                P1j = P1i
                for N1i in range(2):
                    N1j = N1i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, P1i, G1i, 1, P2i, G2i) )
                            right_inds.append( (N1j, P1j, G1j, 0, P2j, G2j) )
                            matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G1/Cg))+1)) )

    # N2 -> P2
    for G1i, G1 in enumerate(gate1_vals):
        G1j = G1i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for N1i in range(2):
                N1j = N1i
                for P1i in range(2):
                    P1j = P1i
                    # Put in matrix element
                    if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                        left_inds.append(  (N1i, P1i, G1i, 0, 1, G2i) )
                        right_inds.append( (N1j, P1j, G1j, 1, 0, G2j) )
                        matrix_entries.append( gamma * (1. / (np.exp(beta*abs(-1.5*Vd-G1/Cg-G1/Cg))-1.+1e-6)) )

    # P2 -> N2
    for G1i, G1 in enumerate(gate1_vals):
        G1j = G1i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for N1i in range(2):
                N1j = N1i
                for P1i in range(2):
                    P1j = P1i
                    # Put in matrix element
                    if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                        left_inds.append(  (N1i, P1i, G1i, 1, 0, G2i) )
                        right_inds.append( (N1j, P1j, G1j, 0, 1, G2j) )
                        matrix_entries.append( gamma * (1. + 1. / (np.exp(beta*abs(-1.5*Vd-G1/Cg-G1/Cg))-1.+1e-6)) )

    # P2 -> D2
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P1i in range(2):
                        P1j = P1i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, P1i, G1i, N2i, 0, G2i) )
                            right_inds.append( (N1j, P1j, G1j, N2j, 1, G2j) )
                            matrix_entries.append( gamma / (np.exp(beta*(G1/Cg-Vd))+1) )

    # D2 -> P2
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P1i in range(2):
                        P1j = P1i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, P1i, G1i, N2i, 1, G2i) )
                            right_inds.append( (N1j, P1j, G1j, N2j, 0, G2j) )
                            matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(G1/Cg-Vd))+1)) )

    # P2 -> G2
    for G2j, G2 in enumerate(gate2_vals[:-1]):
        for G1j, G1 in enumerate(gate1_vals):
            G1i = G1j
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P1i in range(2):
                        P1j = P1i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, P1i, G1i, N2i, 0, G2j+1) )
                            right_inds.append( (N1j, P1j, G1j, N2j, 1, G2j  ) )
                            matrix_entries.append( gamma * ( 1. / (np.exp(beta*(G1/Cg+G2/Cg))+1)) )

    # G2 -> P2
    for G2j, G2 in enumerate(gate2_vals[:-1]):
        for G1j, G1 in enumerate(gate1_vals):
            G1i = G1j
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P1i in range(2):
                        P1j = P1i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j+1 not in absorbing_inds2):
                            left_inds.append(  (N1i, P1i, G1i, N2i, 1, G2j  ) )
                            right_inds.append( (N1j, P1j, G1j, N2j, 0, G2j+1) )
                            matrix_entries.append( gamma * ( 1. - 1. / (np.exp(beta*(G1/Cg+gate2_vals[G2j+1]/Cg))+1)) )

    # N2 -> G2
    for G2j, G2 in enumerate(gate2_vals[:-1]):
        for G1j, G1 in enumerate(gate1_vals):
            G1i = G1j
            for P1i in range(2):
                P1j = P1i
                for N1i in range(2):
                    N1j = N1i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j not in absorbing_inds2):
                            left_inds.append(  (N1i, P1i, G1i, 0, P2i, G2j+1) )
                            right_inds.append( (N1j, P1j, G1j, 1, P2j, G2j  ) )
                            matrix_entries.append( gamma * ( 1. / (np.exp(beta*(-1.5*Vd-G1/Cg+G2/Cg))+1)) )

    # G2 -> N2
    for G2j, G2 in enumerate(gate2_vals[:-1]):
        for G1j, G1 in enumerate(gate1_vals):
            G1i = G1j
            for P1i in range(2):
                P1j = P1i
                for N1i in range(2):
                    N1j = N1i
                    for P2i in range(2):
                        P2j = P2i
                        # Put in matrix element
                        if (G1j not in absorbing_inds1) and (G2j+1 not in absorbing_inds2):
                            left_inds.append(  (N1i, P1i, G1i, 1, P2i, G2j  ) )
                            right_inds.append( (N1j, P1j, G1j, 0, P2j, G2j+1) )
                            matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G1/Cg+gate2_vals[G2j+1]/Cg))+1)) )

    # Populate the diagonal
    diag_entries = dict()
    for i in range(len(right_inds)):
        if right_inds[i] in diag_entries:
            diag_entries[right_inds[i]] -= matrix_entries[i]
        else:
            diag_entries[right_inds[i]] = -matrix_entries[i]

    for diag_ind in diag_entries:
        left_inds.append(diag_ind)
        right_inds.append(diag_ind)
        matrix_entries.append(diag_entries[diag_ind])

    # Convert inds to linear index
    gen_shape = (2, 2, Ngate1, 2, 2, Ngate2)
    for i in range(len(right_inds)):
        right_inds[i] = np.ravel_multi_index(right_inds[i], gen_shape)
        left_inds[i] = np.ravel_multi_index(left_inds[i], gen_shape)

    # Put into a matrix
    W = csc_matrix((matrix_entries, (left_inds, right_inds)),
                   shape=(Ngate1*Ngate2*2*2*2*2, Ngate1*Ngate2*2*2*2*2))

    # Print the memory size of the array
    print(f'Generator Number of elements: {len(matrix_entries)}')
    print(f'Generator Memory Entries: {W.data.nbytes/1000000} Mb')
    print(f'Generator Memory Entries: {W.indptr.nbytes/1000000} Mb')
    print(f'Generator Memory Entries: {W.indices.nbytes/1000000} Mb')
    print(f'Generator Memory: {(W.indices.nbytes+W.data.nbytes+W.indptr.nbytes)/1000000} Mb')

    # Convert to a dense matrix if wanted
    if not sparse:
        W = W.toarray()

    # Reshape the initial tensor into a matrix for diagonalization
    if not square:
        W = np.reshape(W, (2, 2, Ngate1, 2, 2, Ngate2,
                           2, 2, Ngate1, 2, 2, Ngate2))

    return W

def measure_current(vr, vl, params, gate1_vals, gate2_vals,
                    absorbing_inds1 = [], absorbing_inds2 = []):
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Cg = params['Cg']

    Ngate1 = len(gate1_vals)
    Ngate2 = len(gate2_vals)

    # Normalize the provided states
    vl /= np.dot(vr, vl.T)

    # Put the states into the correct shape
    vl = vl.reshape((2, 2, Ngate1, 2, 2, Ngate2))
    vr = vr.reshape((2, 2, Ngate1, 2, 2, Ngate2))

    # Measure the entropy production
    current = dict()

    # N1 <-> S1
    current[('N1', 'S1')] = 0.
    current[('S1', 'N1')] = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for P1i in range(2):
                P1j = P1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma / (np.exp(beta*(-1.5*Vd-G2/Cg))+1)
                        Wji = gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G2/Cg))+1))
                        probi = vr[0, P1i, G1i, N2i, P2i, G2i]
                        probj = vr[1, P1j, G1j, N2j, P2j, G2j]
                        # Calc Currents
                        current[('N1', 'S1')] += Wji*probi
                        current[('S1', 'N1')] += Wij*probj

    # N1 <-> P1
    current[('N1', 'P1')] = 0.
    current[('P1', 'N1')] = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N2i in range(2):
                N2j = N2i
                for P2i in range(2):
                    P2j = P2i
                    # Useful measurements
                    Wij = gamma * (1. / (np.exp(beta*abs(-1.5*Vd-G2/Cg-G2/Cg))-1.+1e-6))
                    Wji = gamma * (1. + 1. / (np.exp(beta*abs(-1.5*Vd-G2/Cg-G2/Cg))-1.+1e-6))
                    probi = vr[0, 1, G1i, N2i, P2i, G2i]
                    probj = vr[1, 0, G1j, N2j, P2j, G2j]
                    # Calc Currents
                    current[('N1', 'P1')] += Wji*probi
                    current[('P1', 'N1')] += Wij*probj

    # P1 <-> D1
    current[('P1', 'D1')] = 0.
    current[('D1', 'P1')] = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma / (np.exp(beta*(G2/Cg-Vd))+1)
                        Wji = gamma * (1. - 1. / (np.exp(beta*(G2/Cg-Vd))+1))
                        probi = vr[N1i, 0, G1i, N2i, P2i, G2i]
                        probj = vr[N1j, 1, G1j, N2j, P2j, G2j]
                        # Calc Currents
                        current[('P1', 'D1')] += Wji*probi
                        current[('D1', 'P1')] += Wij*probj

    # P1 <-> G1
    current[('P1', 'G1')] = 0.
    current[('G1', 'P1')] = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1j, G1 in enumerate(gate1_vals[:-1]):
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma * ( 1. / (np.exp(beta*(G2/Cg+G1/Cg))+1))
                        Wji = gamma * ( 1. - 1. / (np.exp(beta*(G2/Cg+gate1_vals[G1j+1]/Cg))+1))
                        probi = vr[N1i, 0, G1j+1, N2i, P2i, G2i]
                        probj = vr[N1j, 1, G1j  , N2j, P2j, G2j]
                        # Calc Currents
                        current[('P1', 'G1')] += Wji*probi
                        current[('G1', 'P1')] += Wij*probj

    # N1 <-> G1
    current[('N1', 'G1')] = 0.
    current[('G1', 'N1')] = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1j, G1 in enumerate(gate1_vals[:-1]):
            for P1i in range(2):
                P1j = P1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma * ( 1. / (np.exp(beta*(-1.5*Vd-G2/Cg+G1/Cg))+1))
                        Wji = gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G2/Cg+gate1_vals[G1j+1]/Cg))+1))
                        probi = vr[0, P1i, G1j+1, N2i, P2i, G2i]
                        probj = vr[1, P1j, G1j  , N2j, P2j, G2j]
                        # Calc Currents
                        current[('N1', 'G1')] += Wji*probi
                        current[('G1', 'N1')] += Wij*probj

    ##

    # N2 <-> S2
    current[('N2', 'S2')] = 0.
    current[('S2', 'N2')] = 0.
    for G1i, G1 in enumerate(gate1_vals):
        G1j = G1i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for P1i in range(2):
                P1j = P1i
                for N1i in range(2):
                    N1j = N1i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma / (np.exp(beta*(-1.5*Vd-G1/Cg))+1)
                        Wji = gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G1/Cg))+1))
                        probi = vr[N1i, P1i, G1i, 0, P2i, G2i]
                        probj = vr[N1j, P1j, G1j, 1, P2j, G2j]
                        # Calc Currents
                        current[('N2', 'S2')] += Wji*probi
                        current[('S2', 'N2')] += Wij*probj

    # N2 <-> P2
    current[('N2', 'P2')] = 0.
    current[('P2', 'N2')] = 0.
    for G1i, G1 in enumerate(gate1_vals):
        G1j = G1i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for N1i in range(2):
                N1j = N1i
                for P1i in range(2):
                    P1j = P1i
                    # Useful measurements
                    Wij = gamma * (1. / (np.exp(beta*abs(-1.5*Vd-G1/Cg-G1/Cg))-1.+1e-6))
                    Wji = gamma * (1. + 1. / (np.exp(beta*abs(-1.5*Vd-G1/Cg-G1/Cg))-1.+1e-6))
                    probi = vr[N1i, P1i, G1i, 0, 1, G2i]
                    probj = vr[N1j, P1j, G1j, 1, 0, G2j]
                    # Calc Currents
                    current[('N2', 'P2')] += Wji*probi
                    current[('P2', 'N2')] += Wij*probj

    # P2 <-> D2
    current[('P2', 'D2')] = 0.
    current[('D2', 'P2')] = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P1i in range(2):
                        P1j = P1i
                        # Useful measurements
                        Wij = gamma / (np.exp(beta*(G1/Cg-Vd))+1)
                        Wji = gamma * (1. - 1. / (np.exp(beta*(G1/Cg-Vd))+1))
                        probi = vr[N1i, P1i, G1i, N2i, 0, G2i]
                        probj = vr[N1j, P1j, G1j, N2j, 1, G2j]
                        # Calc Currents
                        current[('P2', 'D2')] += Wji*probi
                        current[('D2', 'P2')] += Wij*probj

    # P2 <-> G2
    current[('P2', 'G2')] = 0.
    current[('G2', 'P2')] = 0.
    for G2j, G2 in enumerate(gate2_vals[:-1]):
        for G1j, G1 in enumerate(gate1_vals):
            G1i = G1j
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P1i in range(2):
                        P1j = P1i
                        # Useful measurements
                        Wij = gamma * ( 1. / (np.exp(beta*(G1/Cg+G2/Cg))+1))
                        Wji = gamma * ( 1. - 1. / (np.exp(beta*(G1/Cg+gate2_vals[G2j+1]/Cg))+1))
                        probi = vr[N1i, P1i, G1i, N2i, 0, G2j+1]
                        probj = vr[N1j, P1j, G1j, N2j, 1, G2j  ]
                        # Calc Currents
                        current[('P2', 'G2')] += Wji*probi
                        current[('G2', 'P2')] += Wij*probj

    # N2 <-> G2
    current[('N2', 'G2')] = 0.
    current[('G2', 'N2')] = 0.
    for G2j, G2 in enumerate(gate2_vals[:-1]):
        for G1j, G1 in enumerate(gate1_vals):
            G1i = G1j
            for P1i in range(2):
                P1j = P1i
                for N1i in range(2):
                    N1j = N1i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma * ( 1. / (np.exp(beta*(-1.5*Vd-G1/Cg+G2/Cg))+1))
                        Wji = gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G1/Cg+gate2_vals[G2j+1]/Cg))+1))
                        probi = vr[N1i, P1i, G1i, 0, P2i, G2j+1]
                        probj = vr[N1j, P1j, G1j, 1, P2j, G2j  ]
                        # Calc Currents
                        current[('N2', 'G2')] += Wji*probi
                        current[('G2', 'N2')] += Wij*probj

    return current

def measure_entropy_production(vr, vl, params, gate1_vals, gate2_vals,
                               absorbing_inds1 = [], absorbing_inds2 = []):
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Cg = params['Cg']

    Ngate1 = len(gate1_vals)
    Ngate2 = len(gate2_vals)

    # Normalize the provided states
    vl /= np.dot(vr, vl.T)

    # Put the states into the correct shape
    vl = vl.reshape((2, 2, Ngate1, 2, 2, Ngate2))
    vr = vr.reshape((2, 2, Ngate1, 2, 2, Ngate2))

    # Measure the entropy production
    dS_flow = []

    # N1 <-> S1
    dSi_flow = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for P1i in range(2):
                P1j = P1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma / (np.exp(beta*(-1.5*Vd-G2/Cg))+1)
                        Wji = gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G2/Cg))+1))
                        probi = vr[0, P1i, G1i, N2i, P2i, G2i]
                        probj = vr[1, P1j, G1j, N2j, P2j, G2j]
                        # Calc Entropy Flow
                        if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                            dSi_flow += (Wij*probj) * (-beta*(-1.5*Vd-G2/Cg))
                        if (not G1i in absorbing_inds1) and (not G2i in absorbing_inds2):
                            dSi_flow += (- Wji*probi) * (-beta*(-1.5*Vd-G2/Cg))
    dS_flow.append(dSi_flow)

    # N1 <-> P1
    dSi_flow = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N2i in range(2):
                N2j = N2i
                for P2i in range(2):
                    P2j = P2i
                    # Useful measurements
                    Wij = gamma * (1. / (np.exp(beta*abs(-1.5*Vd-G2/Cg-G2/Cg))-1.+1e-6))
                    Wji = gamma * (1. + 1. / (np.exp(beta*abs(-1.5*Vd-G2/Cg-G2/Cg))-1.+1e-6))
                    probi = vr[0, 1, G1i, N2i, P2i, G2i]
                    probj = vr[1, 0, G1j, N2j, P2j, G2j]
                    # Calc Entropy Flow
                    if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                        dSi_flow += (Wij*probj) * (-beta*abs(-1.5*Vd-G2/Cg-G2/Cg))
                    if (not G1i in absorbing_inds1) and (not G2i in absorbing_inds2):
                        dSi_flow += (- Wji*probi) * (-beta*abs(-1.5*Vd-G2/Cg-G2/Cg))
    dS_flow.append(dSi_flow)

    # P1 <-> D1
    dSi_flow = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma / (np.exp(beta*(G2/Cg-Vd))+1)
                        Wji = gamma * (1. - 1. / (np.exp(beta*(G2/Cg-Vd))+1))
                        probi = vr[N1i, 0, G1i, N2i, P2i, G2i]
                        probj = vr[N1j, 1, G1j, N2j, P2j, G2j]
                        # Calc Entropy Flow
                        if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                            dSi_flow += (Wij*probj) * (-beta*(G2/Cg-Vd))
                        if (not G1i in absorbing_inds1) and (not G2i in absorbing_inds2):
                            dSi_flow += (- Wji*probi) * (-beta*(G2/Cg-Vd))
    dS_flow.append(dSi_flow)

    # P1 <-> G1
    dSi_flow = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1j, G1 in enumerate(gate1_vals[:-1]):
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma * ( 1. / (np.exp(beta*(G2/Cg+G1/Cg))+1))
                        Wji = gamma * ( 1. - 1. / (np.exp(beta*(G2/Cg+gate1_vals[G1j+1]/Cg))+1))
                        probi = vr[N1i, 0, G1j+1, N2i, P2i, G2i]
                        probj = vr[N1j, 1, G1j  , N2j, P2j, G2j]
                        # Calc Entropy Flow
                        if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                            dSi_flow += (Wij*probj) * (-beta*(G2/Cg+G1/Cg))
                        if (not G1j+1 in absorbing_inds1) and (not G2i in absorbing_inds2):
                            dSi_flow += (- Wji*probi) * (-beta*(G2/Cg+gate1_vals[G1j+1]/Cg))
    dS_flow.append(dSi_flow)

    # N1 <-> G1
    dSi_flow = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1j, G1 in enumerate(gate1_vals[:-1]):
            for P1i in range(2):
                P1j = P1i
                for N2i in range(2):
                    N2j = N2i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma * ( 1. / (np.exp(beta*(-1.5*Vd-G2/Cg+G1/Cg))+1))
                        Wji = gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G2/Cg+gate1_vals[G1j+1]/Cg))+1))
                        probi = vr[0, P1i, G1j+1, N2i, P2i, G2i]
                        probj = vr[1, P1j, G1j  , N2j, P2j, G2j]
                        # Calc Entropy Flow
                        if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                            dSi_flow += (Wij*probj) * (-beta*(-1.5*Vd-G2/Cg+G1/Cg))
                        if (not G1j+1 in absorbing_inds1) and (not G2i in absorbing_inds2):
                            dSi_flow += (- Wji*probi) * (-beta*(-1.5*Vd-G2/Cg+gate1_vals[G1j+1]/Cg))
    dS_flow.append(dSi_flow)

    ##

    # N2 <-> S2
    dSi_flow = 0.
    for G1i, G1 in enumerate(gate1_vals):
        G1j = G1i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for P1i in range(2):
                P1j = P1i
                for N1i in range(2):
                    N1j = N1i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma / (np.exp(beta*(-1.5*Vd-G1/Cg))+1)
                        Wji = gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G1/Cg))+1))
                        probi = vr[N1i, P1i, G1i, 0, P2i, G2i]
                        probj = vr[N1j, P1j, G1j, 1, P2j, G2j]
                        # Calc Entropy Flow
                        if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                            dSi_flow += (Wij*probj) * (-beta*(-1.5*Vd-G1/Cg))
                        if (not G1i in absorbing_inds1) and (not G2i in absorbing_inds2):
                            dSi_flow += (- Wji*probi) * (-beta*(-1.5*Vd-G1/Cg))
    dS_flow.append(dSi_flow)

    # N2 <-> P2
    dSi_flow = 0.
    for G1i, G1 in enumerate(gate1_vals):
        G1j = G1i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for N1i in range(2):
                N1j = N1i
                for P1i in range(2):
                    P1j = P1i
                    # Useful measurements
                    Wij = gamma * (1. / (np.exp(beta*abs(-1.5*Vd-G1/Cg-G1/Cg))-1.+1e-6))
                    Wji = gamma * (1. + 1. / (np.exp(beta*abs(-1.5*Vd-G1/Cg-G1/Cg))-1.+1e-6))
                    probi = vr[N1i, P1i, G1i, 0, 1, G2i]
                    probj = vr[N1j, P1j, G1j, 1, 0, G2j]
                    # Calc Entropy Flow
                    if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                        dSi_flow += (Wij*probj) * (-beta*abs(-1.5*Vd-G1/Cg-G1/Cg))
                    if (not G1i in absorbing_inds1) and (not G2i in absorbing_inds2):
                        dSi_flow += (- Wji*probi) * (-beta*abs(-1.5*Vd-G1/Cg-G1/Cg))
    dS_flow.append(dSi_flow)

    # P2 <-> D2
    dSi_flow = 0.
    for G2i, G2 in enumerate(gate2_vals):
        G2j = G2i
        for G1i, G1 in enumerate(gate1_vals):
            G1j = G1i
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P1i in range(2):
                        P1j = P1i
                        # Useful measurements
                        Wij = gamma / (np.exp(beta*(G1/Cg-Vd))+1)
                        Wji = gamma * (1. - 1. / (np.exp(beta*(G1/Cg-Vd))+1))
                        probi = vr[N1i, P1i, G1i, N2i, 0, G2i]
                        probj = vr[N1j, P1j, G1j, N2j, 1, G2j]
                        # Calc Entropy Flow
                        if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                            dSi_flow += (Wij*probj) * (-beta*(G1/Cg-Vd))
                        if (not G1i in absorbing_inds1) and (not G2i in absorbing_inds2):
                            dSi_flow += (- Wji*probi) * (-beta*(G1/Cg-Vd))
    dS_flow.append(dSi_flow)

    # P2 <-> G2
    dSi_flow = 0.
    for G2j, G2 in enumerate(gate2_vals[:-1]):
        for G1j, G1 in enumerate(gate1_vals):
            G1i = G1j
            for N1i in range(2):
                N1j = N1i
                for N2i in range(2):
                    N2j = N2i
                    for P1i in range(2):
                        P1j = P1i
                        # Useful measurements
                        Wij = gamma * ( 1. / (np.exp(beta*(G1/Cg+G2/Cg))+1))
                        Wji = gamma * ( 1. - 1. / (np.exp(beta*(G1/Cg+gate2_vals[G2j+1]/Cg))+1))
                        probi = vr[N1i, P1i, G1i, N2i, 0, G2j+1]
                        probj = vr[N1j, P1j, G1j, N2j, 1, G2j  ]
                        # Calc Entropy Flow
                        if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                            dSi_flow += (Wij*probj) * (-beta*(G1/Cg+G2/Cg))
                        if (not G1i in absorbing_inds1) and (not G2j+1 in absorbing_inds2):
                            dSi_flow += (- Wji*probi) * (-beta*(G1/Cg+gate2_vals[G2j+1]/Cg))
    dS_flow.append(dSi_flow)

    # N2 <-> G2
    dSi_flow = 0.
    for G2j, G2 in enumerate(gate2_vals[:-1]):
        for G1j, G1 in enumerate(gate1_vals):
            G1i = G1j
            for P1i in range(2):
                P1j = P1i
                for N1i in range(2):
                    N1j = N1i
                    for P2i in range(2):
                        P2j = P2i
                        # Useful measurements
                        Wij = gamma * ( 1. / (np.exp(beta*(-1.5*Vd-G1/Cg+G2/Cg))+1))
                        Wji = gamma * (1. - 1. / (np.exp(beta*(-1.5*Vd-G1/Cg+gate2_vals[G2j+1]/Cg))+1))
                        probi = vr[N1i, P1i, G1i, 0, P2i, G2j+1]
                        probj = vr[N1j, P1j, G1j, 1, P2j, G2j  ]
                        # Calc Entropy Flow
                        if (not G1j in absorbing_inds1) and (not G2j in absorbing_inds2):
                            dSi_flow += (Wij*probj) * (-beta*(-1.5*Vd-G1/Cg+G2/Cg))
                        if (not G1i in absorbing_inds1) and (not G2j+1 in absorbing_inds2):
                            dSi_flow += (- Wji*probi) * (-beta*(-1.5*Vd-G1/Cg+gate2_vals[G2j+1]/Cg))
    dS_flow.append(dSi_flow)

    return sum(dS_flow)

def run_ed(params, gate1_vals, gate2_vals,
           vl0=None, vr0=None, nstate=1):
    """
    Run an exact diagonalization calculation for a set of
    coupled not gates
    """

    # Get the generator
    W = get_generator(params, gate1_vals, gate2_vals)

    # Solve the right eigenproblem
    if vr0 is not None:
        if len(vr0.shape) == 2:
            vr0 = vr0[:, 0]
    er, vr = eigs(W, k=nstate, which='LR', maxiter=10000, tol=1e-8, v0=vr0)

    for i in range(len(er)):
        vr[:, i] /= np.sum(vr[:, i])

    # Solve the left eigenproblem
    if vl0 is not None:
        if len(vl0.shape) == 2:
            vl0 = vl0[:, 0]
    el, vl = eigs(W, k=nstate, which='LR', maxiter=10000, tol=1e-8, v0=vl0)

    for i in range(len(el)):
        vl[:, i] /= np.sum(vl[:, i])

    return er, el, vr, vl

def measure_target_probability_density(vr, vl, gate1_vals, gate2_vals, source_inds1, source_inds2):

    Ngate1 = len(gate1_vals)
    Ngate2 = len(gate2_vals)

    # Copy the states
    vl = vl.copy()
    vr = vr.copy()

    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate1, 2, 2, Ngate2))
    vr = vr.reshape((2, 2, Ngate1, 2, 2, Ngate2))

    # Normalize
    norm = np.einsum('ijklmn,ijklmn->', vl, vr)
    vr /= norm
    norm = np.einsum('ijklmn,ijklmn->', vl, vr)

    # Calculate occupation of either
    val = 0.
    for G1i in range(Ngate1):
        for G2i in range(Ngate2):
            if (G1i in source_inds1) or (G2i in source_inds2):
                val += np.sum(vr[:, :, G1i, :, :, G2i])
    return val

def measure_gate_occs(vr, vl, gate1_vals, gate2_vals, plot=False):

    Ngate1 = len(gate1_vals)
    Ngate2 = len(gate2_vals)

    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate1, 2, 2, Ngate2))
    vr = vr.reshape((2, 2, Ngate1, 2, 2, Ngate2))

    # Put into quimb tensors
    #vr = qtn.MatrixProductState.from_dense(vr, [2, 2, Ngate1, 2, 2, Ngate2])
    #vl = qtn.MatrixProductState.from_dense(vl, [2, 2, Ngate1, 2, 2, Ngate2])
    vr = qtn.tensor_1d.Dense1D(vr, phys_dim=[2, 2, Ngate1, 2, 2, Ngate2])
    vl = qtn.tensor_1d.Dense1D(vl, phys_dim=[2, 2, Ngate1, 2, 2, Ngate2])

    # Normalize
    norm = vr & vl
    norm = norm ^ all
    vr /= norm

    # Measure occupation of first gate
    opsG1 = get_ops(Ngate1)
    rhoG1 = np.zeros(Ngate1)
    for occi in range(Ngate1):
        rhoG1[occi] = nsite_operator(vr, vl, [2], [opsG1[f'n{occi}']])

    # Measure occupation of second gate
    opsG2 = get_ops(Ngate2)
    rhoG2 = np.zeros(Ngate2)
    for occi in range(Ngate2):
        rhoG2[occi] = nsite_operator(vr, vl, [5], [opsG2[f'n{occi}']])

    # Plot Densities
    if plot:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.semilogy(gate1_vals, rhoG1, 'b+')
        ax.semilogy(gate1_vals, rhoG2, 'gx')
        plt.show()

    return rhoG1, rhoG2

def get_time_evolution_operator(params, gate1_vals, gate2_vals, dt, W=None,
                                absorbing_inds1=[],
                                absorbing_inds2=[]):
    # Get the generator
    if W is None:
        W = get_generator(params, gate1_vals, gate2_vals,
                          absorbing_inds1=absorbing_inds1,
                          absorbing_inds2=absorbing_inds2)

    # Take the exponential of the generator
    print('Exponentiating Generator')
    U = expm(dt*W)

    #print(f'Evolution Operator Memory Entries: {U.data.nbytes/1000000} Mb')
    #print(f'Evolution Operator Memory Entries: {U.indptr.nbytes/1000000} Mb')
    #print(f'Evolution Operator Memory Entries: {U.indices.nbytes/1000000} Mb')
    #print(f'Evolution Operator Memory: {(U.indices.nbytes+U.data.nbytes+U.indptr.nbytes)/1000000} Mb')

    return U

def time_evolution_step(params, v0, gate1_vals, gate2_vals, dt, U=None, W=None, method='exact',
                        absorbing_inds1=[],
                        absorbing_inds2=[]):
    # Get the generator (needed for all evolution methods)
    if W is None:
        W = get_generator(params, gate1_vals, gate2_vals,
                          absorbing_inds1=absorbing_inds1,
                          absorbing_inds2=absorbing_inds2)

    if method == 'exact':
        # Get the exact evolution operator
        if U is None:
            U = expm(dt*W)

        # Apply the time evolution operator to the state
        v = U.dot(v0)

    elif method == 'euler':
        # Do the time evolution
        v = dt * W.dot(v0) + v0

    elif method == 'rk4':
        # Do the time evolution
        k1 = W.dot(v0)
        k2 = W.dot(v0+dt/2*k1)
        k3 = W.dot(v0+dt/2*k2)
        k4 = W.dot(v0+dt*k3)
        v = v0 + dt/6 * (k1 + 2.*k2 + 2.*k3 + k4)
    else:
        raise ValueError(f'Invalid time evolution method {method} provided')

    # Take a smaller step if necessary
    if (len(np.where( v < 0 )[0]) > 0) or (abs(np.sum(v)-1.) > 1e-3):
        # Do two smaller steps if it was too large of a step
        #if (len(np.where( v < 0 )[0]) > 0):
        #    print(f'Splitting to {0.5*dt} because {len(np.where( v < 0 )[0])} negative elements (max size: {max(abs(v[np.where( v < 0 )[0]]))})')
        #elif (abs(np.sum(v)-1.) > 1e-3):
        #    print(f'Splitting to {0.5*dt} because sum(v) - 1. = {np.sum(v)-1.}')
        if max(abs(v[np.where( v < 0 )[0]])) < 1e-32:
            pass
        else:
            v0 = time_evolution_step(params, v0, gate1_vals, gate2_vals, 0.5*dt, U=U, W=W, method=method)
            v = time_evolution_step(params, v0, gate1_vals, gate2_vals, 0.5*dt, U=U, W=W, method=method)
            print(f'Finished with step {dt}')

    # Renormalize
    v[np.where(v<0)[0]] = 0.
    v /= np.sum(v)

    # Return the time evolved state
    return v

from logical_circuit_bounds.not_gate.tools.ed import time_evolution_step as not_gate_time_evolution_step

def get_steady_initial_state(params, gate1_vals, gate2_vals, dt=1e15, nsteps=1, biased=True):
    """
    Use the steady state distribution of a single not gate
    to set up an initial distribution for the inverter
    """
    Cg = params['Cg']
    Vd = params['Vd']

    # Get the distribution for Vin = 0
    # ---------------------------------------------
    # Set an initial state
    init_state = np.zeros((2, 2, len(gate1_vals)))
    ind = np.argmin(np.abs(gate1_vals + Cg*Vd))
    init_state[0, 0, ind] = 1.
    init_state = init_state.reshape(-1)
    state0 = init_state.copy()

    # Do a long time evolution
    params['Vin'] = 0.
    for step in range(nsteps):
        state0 = not_gate_time_evolution_step(params, state0, gate1_vals, dt)
    state0 = state0.reshape(2, 2, len(gate1_vals))

    # Get the distribution for Vin = 1
    # ---------------------------------------------
    # Set an initial state
    init_state = np.zeros((2, 2, len(gate2_vals)))
    ind = np.argmin(np.abs(gate2_vals - 0))
    init_state[0, 0, ind] = 1.
    init_state = init_state.reshape(-1)
    state1 = init_state.copy()

    # Do a long time evolution
    params['Vin'] = Vd
    for step in range(nsteps):
        state1 = not_gate_time_evolution_step(params, state1, gate1_vals, dt)
    state1 = state1.reshape(2, 2, len(gate1_vals))

    # Combine to get simple initial state
    # ---------------------------------------------
    Ngate1, Ngate2 = len(gate1_vals), len(gate2_vals)
    assert(Ngate1==Ngate2)
    state = np.zeros((2, 2, Ngate1, 2, 2, Ngate2))
    if biased:
        for N1i in range(2):
            for P1i in range(2):
                for N2i in range(2):
                    for P2i in range(2):
                        for Gi in range(Ngate1):
                            state[N1i, P1i, Gi, N2i, P2i, Ngate2-Gi-1] = state1[N1i, P1i, Gi]
    else:
        for N1i in range(2):
            for P1i in range(2):
                for N2i in range(2):
                    for P2i in range(2):
                        for Gi in range(Ngate1):
                            state[N1i, P1i, Gi, N2i, P2i, Gi] = max(state0[N1i, P1i, Gi],
                                                                    state1[N2i, P2i, Gi])

    # Normalize
    state /= np.sum(state)

    return state

if __name__ == "__main__":
    from sys import argv
    params = {'beta': 1.,
              'gamma': 1.,
              'Cg': 1.0,
              'Vd': 7.}
    gate1_vals = np.arange(-13, 5)
    gate2_vals = np.arange(-13, 5)

    er, vr, el, vl = run_ed(params, gate1_vals, gate2_vals)
