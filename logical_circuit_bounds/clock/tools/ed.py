import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, expm
from scipy.sparse import csc_matrix
import quimb.tensor as qtn
from logic_dmrg.clock.tools.op_tools import *

def get_generator(params, gate1_vals, gate2_vals, gate3_vals,
                  absorbing_inds1 = [], absorbing_inds2 = [], absorbing_inds3 = [],
                  signal1 = 'signal3', signal2='signal1', signal3='signal2',
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
    Ngate3 = len(gate3_vals)

    # -----------------------------------------------------------
    # GENERATOR
    # -----------------------------------------------------------
    print('Populating Generator')
    left_inds = []
    right_inds = []
    matrix_entries = []

    # Put each term in the generator
    # GATE 1 (Depends on Gate 3)
    # -------------------------------
    # N1 <-> S1 (Depends on Gate 3)
    print('\tN1<->S1')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal1 == 'signal3':
                    epsilon_P1 = 0.0 + G3/Cg
                    epsilon_N1 = -1.5*Vd - G3/Cg
                elif signal1 == 'signal2':
                    epsilon_P1 = 0.0 + G2/Cg
                    epsilon_N1 = -1.5*Vd - G2/Cg
                elif signal1 == 'signal1':
                    epsilon_P1 = 0.0 + G1/Cg
                    epsilon_N1 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P1 = 0.0 - signal1
                    epsilon_N1 = -1.5*Vd + signal1
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for P1i in range(2):
                            P1j = P1i
                            for N3i in range(2):
                                N3j = N3i
                                for N2i in range(2):
                                    N2j = N2i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # N1 -> S1
                                        left_inds.append( (0, P1i, G1i, N2i, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((1, P1j, G1j, N2j, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma / (np.exp(beta*epsilon_N1)+1.))
                                        # S1 -> N1
                                        left_inds.append( (1, P1i, G1i, N2i, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((0, P1j, G1j, N2j, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*epsilon_N1)+1.)) )


    # N1 <-> P1 (Depends on Gate 3)
    print('\tN1<->P1')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal1 == 'signal3':
                    epsilon_P1 = 0.0 + G3/Cg
                    epsilon_N1 = -1.5*Vd - G3/Cg
                elif signal1 == 'signal2':
                    epsilon_P1 = 0.0 + G2/Cg
                    epsilon_N1 = -1.5*Vd - G2/Cg
                elif signal1 == 'signal1':
                    epsilon_P1 = 0.0 + G1/Cg
                    epsilon_N1 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P1 = 0.0 - signal1
                    epsilon_N1 = -1.5*Vd + signal1
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for N3i in range(2):
                            N3j = N3i
                            for N2i in range(2):
                                N2j = N2i
                                # Put in matrix element
                                if (G1j not in absorbing_inds1) and\
                                   (G2j not in absorbing_inds2) and\
                                   (G3j not in absorbing_inds3):
                                    # N1 -> P1
                                    left_inds.append( (0, 1, G1i, N2i, P2i, G2i, N3i, P3i, G3i))
                                    right_inds.append((1, 0, G1j, N2j, P2j, G2j, N3j, P3j, G3j))
                                    matrix_entries.append(  gamma * ( 1. / (np.exp(beta*(abs(epsilon_P1-epsilon_N1)))-1.+1e-6)) )
                                    # P1 -> N1
                                    left_inds.append( (1, 0, G1i, N2i, P2i, G2i, N3i, P3i, G3i))
                                    right_inds.append((0, 1, G1j, N2j, P2j, G2j, N3j, P3j, G3j))
                                    matrix_entries.append( gamma * (1. + 1. / (np.exp(beta*(abs(epsilon_P1-epsilon_N1)))-1.+1e-6)) )

    # P1 <-> D1 (Depends on Gate 3)
    print('\tP1<->D1')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal1 == 'signal3':
                    epsilon_P1 = 0.0 + G3/Cg
                    epsilon_N1 = -1.5*Vd - G3/Cg
                elif signal1 == 'signal2':
                    epsilon_P1 = 0.0 + G2/Cg
                    epsilon_N1 = -1.5*Vd - G2/Cg
                elif signal1 == 'signal1':
                    epsilon_P1 = 0.0 + G1/Cg
                    epsilon_N1 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P1 = 0.0 - signal1
                    epsilon_N1 = -1.5*Vd + signal1
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for N3i in range(2):
                            N3j = N3i
                            for N2i in range(2):
                                N2j = N2i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # P1 -> D1
                                        left_inds.append( (N1i, 0, G1i, N2i, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((N1j, 1, G1j, N2j, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma * ( 1. / (np.exp(beta*(epsilon_P1-Vd))+1.)) )
                                        # D1 -> P1
                                        left_inds.append( (N1i, 1, G1i, N2i, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((N1j, 0, G1j, N2j, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(epsilon_P1-Vd))+1.)) )

    # P1 <-> G1 (Depends on Gate 3)
    print('\tP1<->G1')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1j, G1 in enumerate(gate1_vals[:-1]):
                if signal1 == 'signal3':
                    epsilon_P1 = 0.0 + G3/Cg
                    epsilon_N1 = -1.5*Vd - G3/Cg
                elif signal1 == 'signal2':
                    epsilon_P1 = 0.0 + G2/Cg
                    epsilon_N1 = -1.5*Vd - G2/Cg
                elif signal1 == 'signal1':
                    epsilon_P1 = 0.0 + G1/Cg
                    epsilon_N1 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P1 = 0.0 - signal1
                    epsilon_N1 = -1.5*Vd + signal1
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for N3i in range(2):
                            N3j = N3i
                            for N2i in range(2):
                                N2j = N2i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # P1 -> G1
                                        left_inds.append( (N1i, 0, G1j+1, N2i, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((N1j, 1, G1j  , N2j, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma*1./(np.exp(beta*(epsilon_P1+G1/Cg))+1.) )
                                    if (G1j+1 not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # G1 -> P1
                                        left_inds.append( (N1i, 1, G1j  , N2i, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((N1j, 0, G1j+1, N2j, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma*(1-1./(np.exp(beta*(epsilon_P1+G1/Cg))+1.)) )

    # N1 -> G1 (Depends on Gate 3)
    print('\tN1<->G1')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1j, G1 in enumerate(gate1_vals[:-1]):
                if signal1 == 'signal3':
                    epsilon_P1 = 0.0 + G3/Cg
                    epsilon_N1 = -1.5*Vd - G3/Cg
                elif signal1 == 'signal2':
                    epsilon_P1 = 0.0 + G2/Cg
                    epsilon_N1 = -1.5*Vd - G2/Cg
                elif signal1 == 'signal1':
                    epsilon_P1 = 0.0 + G1/Cg
                    epsilon_N1 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P1 = 0.0 - signal1
                    epsilon_N1 = -1.5*Vd + signal1
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for P1i in range(2):
                            P1j = P1i
                            for N3i in range(2):
                                N3j = N3i
                                for N2i in range(2):
                                    N2j = N2i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # N1 -> G1
                                        left_inds.append( (0, P1i, G1j+1, N2i, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((1, P1j, G1j  , N2j, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma*1./(np.exp(beta*(epsilon_N1+G1/Cg))+1.) )
                                    if (G1j+1 not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # G1 -> N1
                                        left_inds.append( (1, P1i, G1j  , N2i, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((0, P1j, G1j+1, N2j, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma*(1-1./(np.exp(beta*(epsilon_N1+G1/Cg))+1.)) )

    # GATE 2 (Depends on Gate 1)
    # -------------------------------
    # N2 <-> S2 (Depends on Gate 1)
    print('\tN2<->S2')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal2 == 'signal3':
                    epsilon_P2 = 0.0 + G3/Cg
                    epsilon_N2 = -1.5*Vd - G3/Cg
                elif signal2 == 'signal2':
                    epsilon_P2 = 0.0 + G2/Cg
                    epsilon_N2 = -1.5*Vd - G2/Cg
                elif signal2 == 'signal1':
                    epsilon_P2 = 0.0 + G1/Cg
                    epsilon_N2 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P2 = 0.0 - signal2
                    epsilon_N2 = -1.5*Vd + signal2
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for P1i in range(2):
                            P1j = P1i
                            for N3i in range(2):
                                N3j = N3i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # N2 -> S2
                                        left_inds.append( (N1i, P1i, G1i, 0, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, 1, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma / (np.exp(beta*epsilon_N2)+1.))
                                        # S2 -> N2
                                        left_inds.append( (N1i, P1i, G1i, 1, P2i, G2i, N3i, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, 0, P2j, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*epsilon_N2)+1.)) )


    # N2 <-> P2 (Depends on Gate 1)
    print('\tN2<->P2')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal2 == 'signal3':
                    epsilon_P2 = 0.0 + G3/Cg
                    epsilon_N2 = -1.5*Vd - G3/Cg
                elif signal2 == 'signal2':
                    epsilon_P2 = 0.0 + G2/Cg
                    epsilon_N2 = -1.5*Vd - G2/Cg
                elif signal2 == 'signal1':
                    epsilon_P2 = 0.0 + G1/Cg
                    epsilon_N2 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P2 = 0.0 - signal2
                    epsilon_N2 = -1.5*Vd + signal2
                for P3i in range(2):
                    P3j = P3i
                    for P1i in range(2):
                        P1j = P1i
                        for N3i in range(2):
                            N3j = N3i
                            for N1i in range(2):
                                N1j = N1i
                                # Put in matrix element
                                if (G1j not in absorbing_inds1) and\
                                   (G2j not in absorbing_inds2) and\
                                   (G3j not in absorbing_inds3):
                                    # N2 -> P2
                                    left_inds.append( (N1i, P1i, G1i, 0, 1, G2i, N3i, P3i, G3i))
                                    right_inds.append((N1j, P1j, G1j, 1, 0, G2j, N3j, P3j, G3j))
                                    matrix_entries.append(  gamma * ( 1. / (np.exp(beta*(abs(epsilon_P2-epsilon_N2)))-1.+1e-6)) )
                                    # P2 -> N2
                                    left_inds.append( (N1i, P1i, G1i, 1, 0, G2i, N3i, P3i, G3i))
                                    right_inds.append((N1j, P1j, G1j, 0, 1, G2j, N3j, P3j, G3j))
                                    matrix_entries.append( gamma * (1. + 1. / (np.exp(beta*(abs(epsilon_P2-epsilon_N2)))-1.+1e-6)) )

    # P2 <-> D2 (Depends on Gate 1)
    print('\tP2<->D2')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal2 == 'signal3':
                    epsilon_P2 = 0.0 + G3/Cg
                    epsilon_N2 = -1.5*Vd - G3/Cg
                elif signal2 == 'signal2':
                    epsilon_P2 = 0.0 + G2/Cg
                    epsilon_N2 = -1.5*Vd - G2/Cg
                elif signal2 == 'signal1':
                    epsilon_P2 = 0.0 + G1/Cg
                    epsilon_N2 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P2 = 0.0 - signal2
                    epsilon_N2 = -1.5*Vd + signal2
                for P3i in range(2):
                    P3j = P3i
                    for P1i in range(2):
                        P1j = P1i
                        for N3i in range(2):
                            N3j = N3i
                            for N2i in range(2):
                                N2j = N2i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # P2 -> D2
                                        left_inds.append( (N1i, P1i, G1i, N2i, 0, G2i, N3i, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, N2j, 1, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma * ( 1. / (np.exp(beta*(epsilon_P2-Vd))+1.)) )
                                        # D2 -> P2
                                        left_inds.append( (N1i, P1i, G1i, N2i, 1, G2i, N3i, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, N2j, 0, G2j, N3j, P3j, G3j))
                                        matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(epsilon_P2-Vd))+1.)) )

    # P2 <-> G2 (Depends on Gate 1)
    print('\tP2<->G2')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2j, G2 in enumerate(gate2_vals[:-1]):
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal2 == 'signal3':
                    epsilon_P2 = 0.0 + G3/Cg
                    epsilon_N2 = -1.5*Vd - G3/Cg
                elif signal2 == 'signal2':
                    epsilon_P2 = 0.0 + G2/Cg
                    epsilon_N2 = -1.5*Vd - G2/Cg
                elif signal2 == 'signal1':
                    epsilon_P2 = 0.0 + G1/Cg
                    epsilon_N2 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P2 = 0.0 - signal2
                    epsilon_N2 = -1.5*Vd + signal2
                for P3i in range(2):
                    P3j = P3i
                    for P1i in range(2):
                        P1j = P1i
                        for N3i in range(2):
                            N3j = N3i
                            for N2i in range(2):
                                N2j = N2i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # P2 -> G2
                                        left_inds.append( (N1i, P1i, G1i, N2i, 0, G2j+1, N3i, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, N2j, 1, G2j  , N3j, P3j, G3j))
                                        matrix_entries.append( gamma*1./(np.exp(beta*(epsilon_P2+G2/Cg))+1.) )
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j+1 not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # G2 -> P2
                                        left_inds.append( (N1i, P1i, G1i, N2i, 1, G2j  , N3i, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, N2j, 0, G2j+1, N3j, P3j, G3j))
                                        matrix_entries.append( gamma*(1-1./(np.exp(beta*(epsilon_P2+G2/Cg))+1.)) )

    # N2 -> G2 (Depends on Gate 1)
    print('\tN2<->G2')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2j, G2 in enumerate(gate2_vals[:-1]):
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal2 == 'signal3':
                    epsilon_P2 = 0.0 + G3/Cg
                    epsilon_N2 = -1.5*Vd - G3/Cg
                elif signal2 == 'signal2':
                    epsilon_P2 = 0.0 + G2/Cg
                    epsilon_N2 = -1.5*Vd - G2/Cg
                elif signal2 == 'signal1':
                    epsilon_P2 = 0.0 + G1/Cg
                    epsilon_N2 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P2 = 0.0 - signal2
                    epsilon_N2 = -1.5*Vd + signal2
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for P1i in range(2):
                            P1j = P1i
                            for N3i in range(2):
                                N3j = N3i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # N2 -> G2
                                        left_inds.append( (N1i, P1i, G1i, 0, P2i, G2j+1, N3i, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, 1, P2j, G2j  , N3j, P3j, G3j))
                                        matrix_entries.append( gamma*1./(np.exp(beta*(epsilon_N2+G2/Cg))+1.) )
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j+1 not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # G2 -> N2
                                        left_inds.append( (N1i, P1i, G1i, 1, P2i, G2j  , N3i, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, 0, P2j, G2j+1, N3j, P3j, G3j))
                                        matrix_entries.append( gamma*(1-1./(np.exp(beta*(epsilon_N2+G2/Cg))+1.)) )

    # GATE 3 (Depends on Gate 2)
    # -------------------------------
    # N3 <-> S3 (Depends on Gate 2)
    print('\tN3<->S3')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal3 == 'signal3':
                    epsilon_P3 = 0.0 + G3/Cg
                    epsilon_N3 = -1.5*Vd - G3/Cg
                elif signal3 == 'signal2':
                    epsilon_P3 = 0.0 + G2/Cg
                    epsilon_N3 = -1.5*Vd - G2/Cg
                elif signal3 == 'signal1':
                    epsilon_P3 = 0.0 + G1/Cg
                    epsilon_N3 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P3 = 0.0 - signal3
                    epsilon_N3 = -1.5*Vd + signal3
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for P1i in range(2):
                            P1j = P1i
                            for N2i in range(2):
                                N2j = N2i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # N3 -> S3
                                        left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, 0, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, 1, P3j, G3j))
                                        matrix_entries.append( gamma / (np.exp(beta*epsilon_N3)+1.))
                                        # S3 -> N3
                                        left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, 1, P3i, G3i))
                                        right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, 0, P3j, G3j))
                                        matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*epsilon_N3)+1.)) )


    # N3 <-> P3 (Depends on Gate 2)
    print('\tN3<->P3')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal3 == 'signal3':
                    epsilon_P3 = 0.0 + G3/Cg
                    epsilon_N3 = -1.5*Vd - G3/Cg
                elif signal3 == 'signal2':
                    epsilon_P3 = 0.0 + G2/Cg
                    epsilon_N3 = -1.5*Vd - G2/Cg
                elif signal3 == 'signal1':
                    epsilon_P3 = 0.0 + G1/Cg
                    epsilon_N3 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P3 = 0.0 - signal3
                    epsilon_N3 = -1.5*Vd + signal3
                for P2i in range(2):
                    P2j = P2i
                    for P1i in range(2):
                        P1j = P1i
                        for N2i in range(2):
                            N2j = N2i
                            for N1i in range(2):
                                N1j = N1i
                                # Put in matrix element
                                if (G1j not in absorbing_inds1) and\
                                   (G2j not in absorbing_inds2) and\
                                   (G3j not in absorbing_inds3):
                                    # N3 -> P3
                                    left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, 0, 1, G3i))
                                    right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, 1, 0, G3j))
                                    matrix_entries.append(  gamma * ( 1. / (np.exp(beta*(abs(epsilon_P3-epsilon_N3)))-1.+1e-6)) )
                                    # P3 -> N3
                                    left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, 1, 0, G3i))
                                    right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, 0, 1, G3j))
                                    matrix_entries.append( gamma * (1. + 1. / (np.exp(beta*(abs(epsilon_P3-epsilon_N3)))-1.+1e-6)) )

    # P3 <-> D3 (Depends on Gate 2)
    print('\tP3<->D3')
    for G3i, G3 in enumerate(gate3_vals):
        G3j = G3i
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal3 == 'signal3':
                    epsilon_P3 = 0.0 + G3/Cg
                    epsilon_N3 = -1.5*Vd - G3/Cg
                elif signal3 == 'signal2':
                    epsilon_P3 = 0.0 + G2/Cg
                    epsilon_N3 = -1.5*Vd - G2/Cg
                elif signal3 == 'signal1':
                    epsilon_P3 = 0.0 + G1/Cg
                    epsilon_N3 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P3 = 0.0 - signal3
                    epsilon_N3 = -1.5*Vd + signal3
                for P2i in range(2):
                    P2j = P2i
                    for P1i in range(2):
                        P1j = P1i
                        for N3i in range(2):
                            N3j = N3i
                            for N2i in range(2):
                                N2j = N2i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # P3 -> D3
                                        left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, N3i, 0, G3i))
                                        right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, N3j, 1, G3j))
                                        matrix_entries.append( gamma * ( 1. / (np.exp(beta*(epsilon_P3-Vd))+1.)) )
                                        # D3 -> P3
                                        left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, N3i, 1, G3i))
                                        right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, N3j, 0, G3j))
                                        matrix_entries.append( gamma * (1. - 1. / (np.exp(beta*(epsilon_P3-Vd))+1.)) )

    # P3 <-> G3 (Depends on Gate 2)
    print('\tP3<->G3')
    for G3j, G3 in enumerate(gate3_vals[:-1]):
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal3 == 'signal3':
                    epsilon_P3 = 0.0 + G3/Cg
                    epsilon_N3 = -1.5*Vd - G3/Cg
                elif signal3 == 'signal2':
                    epsilon_P3 = 0.0 + G2/Cg
                    epsilon_N3 = -1.5*Vd - G2/Cg
                elif signal3 == 'signal1':
                    epsilon_P3 = 0.0 + G1/Cg
                    epsilon_N3 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P3 = 0.0 - signal3
                    epsilon_N3 = -1.5*Vd + signal3
                for P2i in range(2):
                    P2j = P2i
                    for P1i in range(2):
                        P1j = P1i
                        for N3i in range(2):
                            N3j = N3i
                            for N2i in range(2):
                                N2j = N2i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # P3 -> G3
                                        left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, N3i, 0, G3j+1))
                                        right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, N3j, 1, G3j  ))
                                        matrix_entries.append( gamma*1./(np.exp(beta*(epsilon_P3+G3/Cg))+1.) )
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j+1 not in absorbing_inds3):
                                        # G3 -> P3
                                        left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, N3i, 1, G3j  ))
                                        right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, N3j, 0, G3j+1))
                                        matrix_entries.append( gamma*(1-1./(np.exp(beta*(epsilon_P3+G3/Cg))+1.)) )

    # N3 -> G3 (Depends on Gate 2)
    print('\tN3<->G3')
    for G3j, G3 in enumerate(gate3_vals[:-1]):
        for G2i, G2 in enumerate(gate2_vals):
            G2j = G2i
            for G1i, G1 in enumerate(gate1_vals):
                G1j = G1i
                if signal3 == 'signal3':
                    epsilon_P3 = 0.0 + G3/Cg
                    epsilon_N3 = -1.5*Vd - G3/Cg
                elif signal3 == 'signal2':
                    epsilon_P3 = 0.0 + G2/Cg
                    epsilon_N3 = -1.5*Vd - G2/Cg
                elif signal3 == 'signal1':
                    epsilon_P3 = 0.0 + G1/Cg
                    epsilon_N3 = -1.5*Vd - G1/Cg
                else:
                    epsilon_P3 = 0.0 - signal3
                    epsilon_N3 = -1.5*Vd + signal3
                for P3i in range(2):
                    P3j = P3i
                    for P2i in range(2):
                        P2j = P2i
                        for P1i in range(2):
                            P1j = P1i
                            for N2i in range(2):
                                N2j = N2i
                                for N1i in range(2):
                                    N1j = N1i
                                    # Put in matrix element
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j not in absorbing_inds3):
                                        # N3 -> G3
                                        left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, 0, P3i, G3j+1))
                                        right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, 1, P3j, G3j  ))
                                        matrix_entries.append( gamma*1./(np.exp(beta*(epsilon_N3+G3/Cg))+1.) )
                                    if (G1j not in absorbing_inds1) and\
                                       (G2j not in absorbing_inds2) and\
                                       (G3j+1 not in absorbing_inds3):
                                        # G3 -> N3
                                        left_inds.append( (N1i, P1i, G1i, N2i, P2i, G2i, 1, P3i, G3j  ))
                                        right_inds.append((N1j, P1j, G1j, N2j, P2j, G2j, 0, P3j, G3j+1))
                                        matrix_entries.append( gamma*(1-1./(np.exp(beta*(epsilon_N3+G3/Cg))+1.)) )


    # Populate the diagonal
    print('Populating Diagonal')
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
    print('Converting to linear index')
    gen_shape = (2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3)
    for i in range(len(right_inds)):
        right_inds[i] = np.ravel_multi_index(right_inds[i], gen_shape)
        left_inds[i] = np.ravel_multi_index(left_inds[i], gen_shape)

    # Put into a matrix
    print('Putting into a sparse matrix')
    W = csc_matrix((matrix_entries, (left_inds, right_inds)),
                   shape=(Ngate1*Ngate2*Ngate3*2*2*2*2*2*2, Ngate1*Ngate2*Ngate3*2*2*2*2*2*2))

    # Print the memory size of the array
    print(f'Generator Number of elements: {len(matrix_entries)}')
    print(f'Generator Memory Entries: {W.data.nbytes/1000000} Mb')
    print(f'Generator Memory Entries: {W.indptr.nbytes/1000000} Mb')
    print(f'Generator Memory Entries: {W.indices.nbytes/1000000} Mb')
    print(f'Generator Memory: {(W.indices.nbytes+W.data.nbytes+W.indptr.nbytes)/1000000} Mb')

    # Convert to a dense matrix if wanted
    print('Converting to a dense matrix')
    if not sparse:
        W = W.toarray()

    # Reshape the initial tensor into a matrix for diagonalization
    if not square:
        W = np.reshape(W, (2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3,
                           2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3))

    return W

def run_ed(params, gate1_vals, gate2_vals, gate3_vals,
           absorbing_inds1 = [],
           absorbing_inds2 = [],
           absorbing_inds3 = [],
           vl0=None, vr0=None, nstate=1):
    """
    Run an exact diagonalization calculation for a set of
    coupled not gates
    """

    # Get the generator
    W = get_generator(params, gate1_vals, gate2_vals, gate3_vals,
                      absorbing_inds1=absorbing_inds1,
                      absorbing_inds2=absorbing_inds2,
                      absorbing_inds3=absorbing_inds3)

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

def measure_target_probability_density(vr, vl, gate1_vals, gate2_vals, gate3_vals,
                                       source_inds1, source_inds2, source_inds3):

    Ngate1 = len(gate1_vals)
    Ngate2 = len(gate2_vals)
    Ngate3 = len(gate3_vals)

    # Copy the states
    vl = vl.copy()
    vr = vr.copy()

    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3))
    vr = vr.reshape((2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3))

    # Normalize
    norm = np.einsum('ijklmn,ijklmn->', vl, vr)
    vr /= norm
    norm = np.einsum('ijklmn,ijklmn->', vl, vr)

    # Calculate occupation of either
    val = 0.
    for G1i in range(Ngate1):
        for G2i in range(Ngate2):
            for G3i in range(Ngate3):
                if (G1i in source_inds1) or (G2i in source_inds2) or (G3i in source_inds3):
                    val += np.sum(vr[:, :, G1i, :, :, G2i, :, :, G3i])
    return val

def measure_gate_occs(vr, vl, gate1_vals, gate2_vals, gate3_vals, plot=False):

    Ngate1 = len(gate1_vals)
    Ngate2 = len(gate2_vals)
    Ngate3 = len(gate3_vals)

    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3))
    vr = vr.reshape((2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3))

    # Put into quimb tensors
    vr = qtn.MatrixProductState.from_dense(vr, [2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3])
    vl = qtn.MatrixProductState.from_dense(vl, [2, 2, Ngate1, 2, 2, Ngate2, 2, 2, Ngate3])

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

    # Measure occupation of third gate
    opsG3 = get_ops(Ngate2)
    rhoG3 = np.zeros(Ngate2)
    for occi in range(Ngate2):
        rhoG3[occi] = nsite_operator(vr, vl, [8], [opsG3[f'n{occi}']])

    # Plot Densities
    if plot:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.semilogy(gate1_vals, rhoG1, 'b+')
        ax.semilogy(gate2_vals, rhoG2, 'gx')
        ax.semilogy(gate3_vals, rhoG3, 'r.')
        plt.show()

    return rhoG1, rhoG2, rhoG3

def get_time_evolution_operator(params, gate1_vals, gate2_vals, gate3_vals, dt,
                                W=None,
                                absorbing_inds1=[],
                                absorbing_inds2=[],
                                absorbing_inds3=[]):
    # Get the generator
    if W is None:
        W = get_generator(params, gate1_vals, gate2_vals, gate3_vals,
                          absorbing_inds1=absorbing_inds1,
                          absorbing_inds2=absorbing_inds2,
                          absorbing_inds3=absorbing_inds3)

    # Take the exponential of the generator
    print('Exponentiating Generator')
    U = expm(dt*W)

    #print(f'Evolution Operator Memory Entries: {U.data.nbytes/1000000} Mb')
    #print(f'Evolution Operator Memory Entries: {U.indptr.nbytes/1000000} Mb')
    #print(f'Evolution Operator Memory Entries: {U.indices.nbytes/1000000} Mb')
    #print(f'Evolution Operator Memory: {(U.indices.nbytes+U.data.nbytes+U.indptr.nbytes)/1000000} Mb')

    return U

def time_evolution_step(params, v0, gate1_vals, gate2_vals, gate3_vals, dt,
                        U=None, W=None, method='exact',
                        absorbing_inds1=[],
                        absorbing_inds2=[],
                        absorbing_inds3=[]):
    # Get the generator (needed for all evolution methods)
    if W is None:
        W = get_generator(params, gate1_vals, gate2_vals, gate2_vals,
                          absorbing_inds1 = absorbing_inds1,
                          absorbing_inds2 = absorbing_inds2,
                          absorbing_inds3 = absorbing_inds3)

    if method == 'exact':
        # Get the exact evolution operator
        if U is None:
            print('Exponentiating evolution matrix')
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
        if max(abs(v[np.where( v < 0 )[0]])) < 1e-32:
            pass
        else:
            v0 = time_evolution_step(params, v0,
                                     gate1_vals, gate2_vals, gate3_vals,
                                     0.5*dt, U=U, W=W, method=method,
                                     absorbing_inds1=absorbing_inds1,
                                     absorbing_inds2=absorbing_inds2,
                                     absorbing_inds3=absorbing_inds3)
            v = time_evolution_step(params, v0, gate1_vals, gate2_vals, gate3_vals,
                                    0.5*dt, U=U, W=W, method=method,
                                    absorbing_inds1=absorbing_inds1,
                                    absorbing_inds2=absorbing_inds2,
                                    absorbing_inds3=absorbing_inds3)
            print(f'Finished with step {dt}')

    # Renormalize
    v[np.where(v<0)[0]] = 0.
    v /= np.sum(v)

    # Return the time evolved state
    return v
