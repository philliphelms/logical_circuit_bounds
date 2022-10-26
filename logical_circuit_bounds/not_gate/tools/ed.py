import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from numpy.linalg import eig
import quimb.tensor as qtn
from logical_circuit_bounds.not_gate.tools.op_tools import *
cmap = plt.get_cmap('viridis')
blues_cmap = plt.get_cmap('Blues')

def get_absorbing_generator(params, gate_vals, absorbing_inds, square=True):
    # -----------------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------------
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin

    Ngate = len(gate_vals)

    # -----------------------------------------------------------
    # GENERATOR
    # -----------------------------------------------------------
    # Store the generator as
    # (G1, N1, P1, G2, N2, P2) x (G1, N1, P1, G2, N2, P2)
    W = np.zeros((Ngate, 2, 2, Ngate, 2, 2))

    # Put each term in the generator

    # N1 -> S1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            # Put in matrix element
            if Gj not in absorbing_inds:
                W[Gi, 0, Pi, Gj, 1, Pj] = gamma / (np.exp(beta*epsilonN)+1.)

    # S1 -> N1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            # Put in matrix element
            if Gj not in absorbing_inds:
                W[Gi, 1, Pi, Gj, 0, Pj] = gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.))

    # N1 -> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        if Gj not in absorbing_inds:
            W[Gi, 0, 1, Gj, 1, 0] = gamma * ( 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))

    # P1 -> N1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        if Gj not in absorbing_inds:
            W[Gi, 1, 0, Gj, 0, 1] = gamma * (1. + 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))

    # P1 -> D1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            # Put in matrix element
            if Gj not in absorbing_inds:
                W[Gi, Ni, 0, Gj, Nj, 1] = gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.))

    # D1 -> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            # Put in matrix element
            if Gj not in absorbing_inds:
                W[Gi, Ni, 1, Gj, Nj, 0] = gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.))

    # P1 -> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Ni in range(2):
            Nj = Ni
            if Gi not in absorbing_inds:
                W[Gi+1, Ni, 0, Gi, Nj, 1] = gamma*1./(np.exp(beta*(epsilonP+G/Cg))+1.)

    # G1 -> P1
    for Gi, G in enumerate(gate_vals[1:]):
        for Ni in range(2):
            Nj = Ni
            if Gi+1 not in absorbing_inds:
                W[Gi, Ni, 1, Gi+1, Nj, 0] = gamma*(1-1./(np.exp(beta*(epsilonP+G/Cg))+1.))

    # N1 -> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Pi in range(2):
            Pj = Pi
            if Gi not in absorbing_inds:
                W[Gi+1, 0, Pi, Gi, 1, Pj] = gamma*1./(np.exp(beta*(epsilonN+G/Cg))+1.)

    # G1 -> N1
    for Gi, G in enumerate(gate_vals[1:]):
        for Pi in range(2):
            Pj = Pi
            if Gi+1 not in absorbing_inds:
                W[Gi, 1, Pi, Gi+1, 0, Pj] = gamma*(1-1./(np.exp(beta*(epsilonN+G/Cg))+1.))

    # Populate the diagonal
    for Gi, G in enumerate(gate_vals):
        for Ni in range(2):
            for Pi in range(2):
                W[Gi, Ni, Pi, Gi, Ni, Pi] = -np.sum(W[:, :, :, Gi, Ni, Pi])

    # Do a transpose to match kmc simulation
    # (N, P, G) x (N, P, G)
    W = W.transpose([1, 2, 0, 4, 5, 3])

    # Reshape the initial tensor into a matrix for diagonalization
    if square:
        W = np.reshape(W, (Ngate*2*2, Ngate*2*2))

    return W

def get_generator(params, gate_vals, square=True):
    # -----------------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------------
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin

    Ngate = len(gate_vals)

    # -----------------------------------------------------------
    # GENERATOR
    # -----------------------------------------------------------
    # Store the generator as
    # (G1, N1, P1, G2, N2, P2) x (G1, N1, P1, G2, N2, P2)
    W = np.zeros((Ngate, 2, 2, Ngate, 2, 2))

    # Put each term in the generator

    # N1 -> S1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            # Put in matrix element
            W[Gi, 0, Pi, Gj, 1, Pj] = gamma / (np.exp(beta*epsilonN)+1.)

    # S1 -> N1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            # Put in matrix element
            W[Gi, 1, Pi, Gj, 0, Pj] = gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.))

    # N1 -> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        W[Gi, 0, 1, Gj, 1, 0] = gamma * ( 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))

    # P1 -> N1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        W[Gi, 1, 0, Gj, 0, 1] = gamma * (1. + 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))

    # P1 -> D1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            # Put in matrix element
            W[Gi, Ni, 0, Gj, Nj, 1] = gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.))

    # D1 -> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            # Put in matrix element
            W[Gi, Ni, 1, Gj, Nj, 0] = gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.))

    # P1 -> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Ni in range(2):
            Nj = Ni
            W[Gi+1, Ni, 0, Gi, Nj, 1] = gamma*1./(np.exp(beta*(epsilonP+G/Cg))+1.)

    # G1 -> P1
    for Gi, G in enumerate(gate_vals[1:]):
        for Ni in range(2):
            Nj = Ni
            W[Gi, Ni, 1, Gi+1, Nj, 0] = gamma*(1-1./(np.exp(beta*(epsilonP+G/Cg))+1.))

    # N1 -> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Pi in range(2):
            Pj = Pi
            W[Gi+1, 0, Pi, Gi, 1, Pj] = gamma*1./(np.exp(beta*(epsilonN+G/Cg))+1.)

    # G1 -> N1
    for Gi, G in enumerate(gate_vals[1:]):
        for Pi in range(2):
            Pj = Pi
            W[Gi, 1, Pi, Gi+1, 0, Pj] = gamma*(1-1./(np.exp(beta*(epsilonN+G/Cg))+1.))

    # Populate the diagonal
    for Gi, G in enumerate(gate_vals):
        for Ni in range(2):
            for Pi in range(2):
                W[Gi, Ni, Pi, Gi, Ni, Pi] = -np.sum(W[:, :, :, Gi, Ni, Pi])

    # Do a transpose to match kmc simulation
    # (N, P, G) x (N, P, G)
    W = W.transpose([1, 2, 0, 4, 5, 3])

    # Reshape the initial tensor into a matrix for diagonalization
    if square:
        W = np.reshape(W, (Ngate*2*2, Ngate*2*2))

    return W

def get_dQ_operator(params, gate_vals, square=True, tensor=False):
    """
    Get an operator that measures the instantaneous change
    in heat dissipation
    """
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin

    Ngate = len(gate_vals)

    # Store the operator as
    # (G1, N1, P1, G2, N2, P2) x (G1, N1, P1, G2, N2, P2)
    W = np.zeros((Ngate, 2, 2, Ngate, 2, 2))

    # Current between D and P
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            # P1 -> D1
            W[Gi, Ni, 0, Gj, Nj, 1] = - gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.)) * (-G/Cg-Vd)
            # D1 -> P1
            W[Gi, Ni, 1, Gj, Nj, 0] = + gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.)) * (-G/Cg-Vd)

    # Current between S and N
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            # N1 -> S1
            W[Gi, 0, Pi, Gj, 1, Pj] = + gamma / (np.exp(beta*epsilonN)+1.) * -G/Cg
            # S1 -> N1
            W[Gi, 1, Pi, Gj, 0, Pj] = - gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.)) * -G/Cg

    # Do a transpose to match kmc simulation
    # (N, P, G) x (N, P, G)
    W = W.transpose([1, 2, 0, 4, 5, 3])

    # Reshape the initial tensor into a matrix for diagonalization
    if square:
        W = np.reshape(W, (Ngate*2*2, Ngate*2*2))
    elif tensor:
        W = qtn.Tensor(W, inds=['b0', 'b1', 'b2', 'k0', 'k1', 'k2'])

    return W

def run_ed(params, gate_vals, plot=False,
           vl0=None, vr0=None, nstate=1):
    """
    Run an exact diagonalization calculation for a set of
    coupled not gates
    """

    # Get the generator
    W = get_generator(params, gate_vals)

    # Solve the right eigenproblem
    if True:
        if vr0 is not None:
            vr0 = vr0[:, 0]
        er, vr = eigs(W, k=nstate, which='LR', maxiter=100000, tol=1e-8, v0=vr0)
    else:
        er, vr = eig(W)
        inds = np.argsort(er)[::-1]
        er = er[inds]
        vr = vr[:, inds]

    for i in range(len(er)):
        vr[:, i] /= np.sum(vr[:, i])


    # Solve the left eigenproblem
    if True:
        if vl0 is not None:
            vl0 = vl0[:, 0]
        el, vl = eigs(W.T, k=nstate, which='LR', maxiter=100000, tol=1e-8, v0=vl0)
    else:
        el, vl = eig(W.T)
        inds = np.argsort(el)[::-1]
        el = el[inds]
        vl = vl[:, inds]

    for i in range(len(el)):
        vl[:, i] /= np.sum(vl[:, i])

    return er, el, vr, vl

def measure_current(vr, vl, params, gate_vals, thresh=1e-100, absorbing_inds=[]):
    vr = vr.copy()
    vl = vl.copy()
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin
    Ngate = len(gate_vals)

    # Normalize the provided states
    vl /= np.dot(vr, vl.T)

    # Put the state into correct shape
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))
    vl = vl.transpose(2, 0, 1)
    vr = vr.transpose(2, 0, 1)

    # Measure the entropy production
    current = dict()

    # N1 <-> S1
    current[('N1', 'S1')] = 0.
    current[('S1', 'N1')] = 0.
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            Wij = gamma / (np.exp(beta*epsilonN)+1.)
            Wji = gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.))
            probi = vr[Gi, 0, Pi]
            probj = vr[Gj, 1, Pj]
            # Current
            if Gi not in absorbing_inds:
                current[('N1', 'S1')] += Wji*probi
            if Gj not in absorbing_inds:
                current[('S1', 'N1')] += Wij*probj

    # N1 <-> P1
    current[('N1', 'P1')] = 0.
    current[('P1', 'N1')] = 0.
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        Wij = gamma * ( 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        Wji = gamma * (1. + 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        probi = vr[Gi, 0, 1]
        probj = vr[Gi, 1, 0]
        # Current
        if Gi not in absorbing_inds:
            current[('N1', 'P1')] += Wji*probi
        if Gj not in absorbing_inds:
            current[('P1', 'N1')] += Wij*probj

    # P1 <-> D1
    current[('P1', 'D1')] = 0.
    current[('D1', 'P1')] = 0.
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            Wij = gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            Wji = gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            probi = vr[Gi, Ni, 0]
            probj = vr[Gj, Nj, 1]
            # Current
            if Gi not in absorbing_inds:
                current[('P1', 'D1')] += Wji*probi
            if Gj not in absorbing_inds:
                current[('D1', 'P1')] += Wij*probj

    # P1 <-> G1
    current[('P1', 'G1')] = 0.
    current[('G1', 'P1')] = 0.
    for Gi, G in enumerate(gate_vals[:-1]):
        for Ni in range(2):
            Nj = Ni
            Wij = gamma*1./(np.exp(beta*(epsilonP+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonP+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, Ni, 0]
            probj = vr[Gi, Nj, 1]
            # Current
            if Gi+1 not in absorbing_inds:
                current[('P1', 'G1')] += Wji*probi
            if Gi not in absorbing_inds:
                current[('G1', 'P1')] += Wij*probj

    # N1 <-> G1
    current[('N1', 'G1')] = 0.
    current[('G1', 'N1')] = 0.
    for Gi, G in enumerate(gate_vals[:-1]):
        for Pi in range(2):
            Pj = Pi
            Wij = gamma*1./(np.exp(beta*(epsilonN+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonN+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, 0, Pi]
            probj = vr[Gi, 1, Pj]
            # Current
            if Gi+1 not in absorbing_inds:
                current[('N1', 'G1')] += Wji*probi
            if Gi not in absorbing_inds:
                current[('G1', 'N1')] += Wij*probj

    return current

def measure_entropy_production(vr, vl, params, gate_vals, thresh=1e-100, absorbing_inds=[], return_sum=True):
    vr = vr.copy()
    vl = vl.copy()
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin
    Ngate = len(gate_vals)

    # Normalize the provided states
    vr /= np.dot(vr, vl.T)

    # Put the state into correct shape
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))
    vl = vl.transpose(2, 0, 1)
    vr = vr.transpose(2, 0, 1)

    # Measure the entropy production
    dS_flow = dict()

    # N1 <-> S1
    dS_flow[('N1', 'S1')] = 0.
    dS_flow[('S1', 'N1')] = 0.
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            Wij = gamma / (np.exp(beta*epsilonN)+1.)
            Wji = gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.))
            probi = vr[Gi, 0, Pi]
            probj = vr[Gj, 1, Pj]
            # Entropy Flow
            if not (Gj in absorbing_inds):
                dS_flow[('S1', 'N1')] += (Wij*probj) * (-beta*epsilonN)
            if not (Gi in absorbing_inds):
                dS_flow[('N1', 'S1')] += (- Wji*probi) * (-beta*epsilonN)

    # N1 <-> P1
    dS_flow[('N1', 'P1')] = 0.
    dS_flow[('P1', 'N1')] = 0.
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        Wij = gamma * ( 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        Wji = gamma * (1. + 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        probi = vr[Gi, 0, 1]
        probj = vr[Gi, 1, 0]
        # Entropy Flow
        if not (Gj in absorbing_inds):
            dS_flow[('P1', 'N1')] += (Wij*probj) * (-beta*abs(epsilonP-epsilonN))
        if not (Gi in absorbing_inds):
            dS_flow[('N1', 'P1')] += (- Wji*probi) * (-beta*abs(epsilonP-epsilonN))

    # P1 <-> D1
    dS_flow[('P1', 'D1')] = 0.
    dS_flow[('D1', 'P1')] = 0.
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            Wij = gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            Wji = gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            probi = vr[Gi, Ni, 0]
            probj = vr[Gj, Nj, 1]
            # Entropy Flow
            if not (Gj in absorbing_inds):
                dS_flow[('D1', 'P1')] += (Wij*probj) * (-beta*(epsilonP-Vd))
            if not (Gi in absorbing_inds):
                dS_flow[('P1', 'D1')] += (- Wji*probi) * (-beta*(epsilonP-Vd))

    # P1 <-> G1
    dS_flow[('P1', 'G1')] = 0.
    dS_flow[('G1', 'P1')] = 0.
    for Gi, G in enumerate(gate_vals[:-1]):
        for Ni in range(2):
            Nj = Ni
            Wij = gamma*1./(np.exp(beta*(epsilonP+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonP+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, Ni, 0]
            probj = vr[Gi, Nj, 1]
            # Entropy Flow (NOTE - Could use the 'midpoint rule' instead)
            if not (Gi in absorbing_inds):
                dS_flow[('G1', 'P1')] += (Wij*probj) * (-beta*(epsilonP+G/Cg))
            if not (Gi+1 in absorbing_inds):
                dS_flow[('P1', 'G1')] += (- Wji*probi) * (-beta*(epsilonP+gate_vals[Gi+1]/Cg))

    # N1 <-> G1
    dS_flow[('N1', 'G1')] = 0.
    dS_flow[('G1', 'N1')] = 0.
    for Gi, G in enumerate(gate_vals[:-1]):
        for Pi in range(2):
            Pj = Pi
            Wij = gamma*1./(np.exp(beta*(epsilonN+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonN+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, 0, Pi]
            probj = vr[Gi, 1, Pj]
            # Entropy Flow (NOTE - Could use the 'midpoint rule' instead)
            if not (Gi in absorbing_inds):
                dS_flow[('G1', 'N1')] += (Wij*probj) * (-beta*(epsilonN+G/Cg))
            if not (Gi+1 in absorbing_inds):
                dS_flow[('N1', 'G1')] += (- Wji*probi) * (-beta*(epsilonN+gate_vals[Gi+1]/Cg))

    #print(f'(N,S) {dS_flow[0]} (N,P) {dS_flow[1]} (P,D) {dS_flow[2]} (P,G) {dS_flow[3]} (N,G) {dS_flow[4]}')
    if return_sum:
        return sum(dS_flow.values())
    else:
        return dS_flow

def measure_entropy_production_backup(vr, vl, params, gate_vals, thresh=1e-100):
    vr = vr.copy()
    vl = vl.copy()
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin
    Ngate = len(gate_vals)

    # Normalize the provided states
    vl /= np.dot(vr, vl.T)

    # Put the state into correct shape
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))
    vl = vl.transpose(2, 0, 1)
    vr = vr.transpose(2, 0, 1)

    # Measure the entropy production
    dS_flow = []

    # N1 <-> S1
    dSi_flow = 0.
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            Wij = gamma / (np.exp(beta*epsilonN)+1.)
            Wji = gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.))
            probi = vr[Gi, 0, Pi]
            probj = vr[Gj, 1, Pj]
            # Entropy Flow
            dSi_flow += (Wij*probj - Wji*probi) * (-beta*epsilonN)
    dS_flow.append(dSi_flow)

    dSi_flow = 0.
    # N1 <-> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        Wij = gamma * ( 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        Wji = gamma * (1. + 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        probi = vr[Gi, 0, 1]
        probj = vr[Gi, 1, 0]
        # Entropy Flow
        dSi_flow += (Wij*probj - Wji*probi) * (-beta*abs(epsilonP-epsilonN))
    dS_flow.append(dSi_flow)

    # P1 <-> D1
    dSi_flow = 0.
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            Wij = gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            Wji = gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            probi = vr[Gi, Ni, 0]
            probj = vr[Gj, Nj, 1]
            # Entropy Flow
            dSi_flow += (Wij*probj - Wji*probi) * (-beta*(epsilonP-Vd))
    dS_flow.append(dSi_flow)

    # P1 <-> G1
    dSi_flow = 0.
    for Gi, G in enumerate(gate_vals[:-1]):
        for Ni in range(2):
            Nj = Ni
            Wij = gamma*1./(np.exp(beta*(epsilonP+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonP+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, Ni, 0]
            probj = vr[Gi, Nj, 1]
            # Entropy Flow
            dSi_flow += (Wij*probj - Wji*probi) * (-beta*epsilonP+G/Cg)
    dS_flow.append(dSi_flow)

    # N1 <-> G1
    dSi_flow = 0.
    for Gi, G in enumerate(gate_vals[:-1]):
        for Pi in range(2):
            Pj = Pi
            Wij = gamma*1./(np.exp(beta*(epsilonN+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonN+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, 0, Pi]
            probj = vr[Gi, 1, Pj]
            # Entropy Flow
            dSi_flow += (Wij*probj - Wji*probi) * (-beta*(epsilonN+G/Cg))
    dS_flow.append(dSi_flow)

    print(sum(dS_flow), dS_flow)
    return sum(dS_flow)

def measure_entropy_production_old(vr, vl, params, gate_vals, thresh=1e-100):
    vr = vr.copy()
    vl = vl.copy()
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin
    Ngate = len(gate_vals)

    # Normalize the provided states
    vl /= np.dot(vr, vl.T)

    # Put the state into correct shape
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))
    vl = vl.transpose(2, 0, 1)
    vr = vr.transpose(2, 0, 1)

    # Measure the entropy production
    dS_flow = []
    dS_prod = []
    dS = []

    # N1 <-> S1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            Wij = gamma / (np.exp(beta*epsilonN)+1.)
            Wji = gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.))
            probi = vr[Gi, 0, Pi]
            probj = vr[Gj, 1, Pj]
            # Total entropy
            #dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
            #if np.isfinite(dSi):
            #    dS.append(dSi)
            # Entropy Production
            dSi_prod = (Wij*probj - Wji*probi) * np.log( (Wij*probj) / (Wji*probi) )
            if np.isfinite(dSi_prod):
                dS_prod.append(dSi_prod)
            # Entropy Flow
            #dSi_flow = (Wij*probj - Wji*probi) * np.log( Wji / Wij )
            dSi_flow = (Wij*probj - Wji*probi) * (-beta*epsilonN)
            if np.isfinite(dSi_flow):
                dS_flow.append(dSi_flow)

    # N1 <-> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        Wij = gamma * ( 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        Wji = gamma * (1. + 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        probi = vr[Gi, 0, 1]
        probj = vr[Gi, 1, 0]
        # Total Entropy
        #dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
        #if np.isfinite(dSi):
        #    dS.append(dSi)
        # Entropy Production
        dSi_prod = (Wij*probj - Wji*probi) * np.log( (Wij*probj) / (Wji*probi) )
        if np.isfinite(dSi_prod):
            dS_prod.append(dSi_prod)
        # Entropy Flow
        #dSi_flow = (Wij*probj - Wji*probi) * np.log( Wji / Wij )
        dSi_flow = (Wij*probj - Wji*probi) * (-beta*abs(epsilonP-epsilonN))
        if np.isfinite(dSi_flow):
            dS_flow.append(dSi_flow)

    # P1 <-> D1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            Wij = gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            Wji = gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            probi = vr[Gi, Ni, 0]
            probj = vr[Gj, Nj, 1]
            # Total Entropy
            #dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
            #if np.isfinite(dSi):
            #    dS.append(dSi)
            # Entropy Production
            dSi_prod = (Wij*probj - Wji*probi) * np.log( (Wij*probj) / (Wji*probi) )
            if np.isfinite(dSi_prod):
                dS_prod.append(dSi_prod)
            # Entropy Flow
            #dSi_flow = (Wij*probj - Wji*probi) * np.log( Wji / Wij )
            dSi_flow = (Wij*probj - Wji*probi) * (-beta*(epsilonP-Vd))
            if np.isfinite(dSi_flow):
                dS_flow.append(dSi_flow)

    # P1 <-> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Ni in range(2):
            Nj = Ni
            Wij = gamma*1./(np.exp(beta*(epsilonP+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonP+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, Ni, 0]
            probj = vr[Gi, Nj, 1]
            # Total Entropy
            #dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
            #if np.isfinite(dSi):
            #    dS.append(dSi)
            # Entropy Production
            dSi_prod = (Wij*probj - Wji*probi) * np.log( (Wij*probj) / (Wji*probi) )
            if np.isfinite(dSi_prod):
                dS_prod.append(dSi_prod)
            # Entropy Flow
            #dSi_flow = (Wij*probj - Wji*probi) * np.log( Wji / Wij )
            dSi_flow = (Wij*probj - Wji*probi) * (-beta*epsilonP+G/Cg)
            if np.isfinite(dSi_flow):
                dS_flow.append(dSi_flow)

    # N1 <-> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Pi in range(2):
            Pj = Pi
            Wij = gamma*1./(np.exp(beta*(epsilonN+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonN+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, 0, Pi]
            probj = vr[Gi, 1, Pj]
            # Total entropy
            #dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
            #if np.isfinite(dSi):
            #    dS.append(dSi)
            # Entropy Production
            dSi_prod = (Wij*probj - Wji*probi) * np.log( (Wij*probj) / (Wji*probi) )
            if np.isfinite(dSi_prod):
                dS_prod.append(dSi_prod)
            # Entropy Flow
            #dSi_flow = (Wij*probj - Wji*probi) * np.log( Wji / Wij )
            dSi_flow = (Wij*probj - Wji*probi) * (-beta*(epsilonN+G/Cg))
            if np.isfinite(dSi_flow):
                dS_flow.append(dSi_flow)

    return sum(dS_flow)#, sum(dS_prod), np.sum(dS)

def measure_entropy_production_v2(vr, vl, params, gate_vals, thresh=1e-100):
    vr = vr.copy()
    vl = vl.copy()
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin
    Ngate = len(gate_vals)

    # Normalize the provided states
    vl /= np.dot(vr, vl.T)

    # Put the state into correct shape
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))
    vl = vl.transpose(2, 0, 1)
    vr = vr.transpose(2, 0, 1)

    dS = []

    # Measure the entropy production
    dS = []

    # N1 <-> S1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            Wij = gamma / (np.exp(beta*epsilonN)+1.)
            Wji = gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.))
            probi = vr[Gi, 0, Pi]
            probj = vr[Gj, 1, Pj]
            dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
            if not np.isfinite(dSi):
                print(Wij, Wji, probj, probi, probj/probi, np.log(probj/probi), np.log(probj)-np.log(probi), Wij*probj - Wji*probi, dSi)
            else:
                dS.append(dSi)

    # N1 <-> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        Wij = gamma * ( 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        Wji = gamma * (1. + 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        probi = vr[Gi, 0, 1]
        probj = vr[Gi, 1, 0]
        dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
        if not np.isfinite(dSi):
            print(Wij, Wji, probj, probi, probj/probi, np.log(probj/probi), np.log(probj)-np.log(probi), Wij*probj - Wji*probi, dSi)
        else:
            dS.append(dSi)

    # P1 <-> D1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            Wij = gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            Wji = gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            probi = vr[Gi, Ni, 0]
            probj = vr[Gj, Nj, 1]
            dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
            if not np.isfinite(dSi):
                print(Wij, Wji, probj, probi, probj/probi, np.log(probj/probi), np.log(probj)-np.log(probi), Wij*probj - Wji*probi, dSi)
            else:
                dS.append(dSi)

    # P1 <-> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Ni in range(2):
            Nj = Ni
            Wij = gamma*1./(np.exp(beta*(epsilonP+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonP+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, Ni, 0]
            probj = vr[Gi, Nj, 1]
            dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
            if not np.isfinite(dSi):
                print(Wij, Wji, probj, probi, probj/probi, np.log(probj/probi), np.log(probj)-np.log(probi), Wij*probj - Wji*probi, dSi)
            else:
                dS.append(dSi)

    # N1 <-> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Pi in range(2):
            Pj = Pi
            Wij = gamma*1./(np.exp(beta*(epsilonN+G/Cg))+1.)
            Wji = gamma*(1-1./(np.exp(beta*(epsilonN+gate_vals[Gi+1]/Cg))+1.))
            probi = vr[Gi+1, 0, Pi]
            probj = vr[Gi, 1, Pj]
            dSi = (Wij*probj - Wji*probi) * np.log( probj / probi )
            if not np.isfinite(dSi):
                print(Wij, Wji, probj, probi, probj/probi, np.log(probj/probi), np.log(probj)-np.log(probi), Wij*probj - Wji*probi, dSi)
            else:
                dS.append(dSi)

    return sum(dS)

def measure_entropy_production_v1(vr, vl, params, gate_vals, thresh=1e-100):
    vr = vr.copy()
    vl = vl.copy()
    # Get the parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Vin = params['Vin']
    Cg = params['Cg']
    epsilonP = 0.0-Vin
    epsilonN = -1.5*Vd+Vin
    Ngate = len(gate_vals)

    # Put the state into correct shape
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))
    vl = vl.transpose(2, 0, 1)
    vr = vr.transpose(2, 0, 1)

    # Measure entropy production
    dS = []

    # N1 -> S1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            # Get entropy production from this transition
            Wij = gamma / (np.exp(beta*epsilonN)+1.)
            probi = vr[Gi, 0, Pi]
            probj = vr[Gj, 1, Pj]
            if probj > thresh:
                #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
                dSi = -Wij * probj * (np.log(probi/probj))
                dS.append(dSi)

    # S1 -> N1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Pi in range(2):
            Pj = Pi
            # Get entropy production from this transition
            Wij = gamma * (1. - 1. / (np.exp(beta*epsilonN)+1.))
            probi = vr[Gi, 1, Pi]
            probj = vr[Gj, 0, Pj]
            if probj > thresh:
                #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
                dSi = -Wij * probj * (np.log(probi/probj))
                dS.append(dSi)

    # N1 -> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        # Get entropy production from this transition
        Wij = gamma * ( 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        probi = vr[Gi, 0, 1]
        probj = vr[Gj, 1, 0]
        if probj > thresh:
            #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
            dSi = -Wij * probj * (np.log(probi/probj))
            dS.append(dSi)

    # P1 -> N1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        # Get entropy production from this transition
        Wij = gamma * (1. + 1. / (np.exp(beta*(abs(epsilonP-epsilonN)))-1.+1e-6))
        probi = vr[Gi, 1, 0]
        probj = vr[Gj, 0, 1]
        if probj > thresh:
            #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
            dSi = -Wij * probj * (np.log(probi/probj))
            dS.append(dSi)

    # P1 -> D1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            # Get entropy production from this transition
            Wij = gamma * ( 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            probi = vr[Gi, Ni, 0]
            probj = vr[Gj, Nj, 1]
            if probj > thresh:
                #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
                dSi = -Wij * probj * (np.log(probi/probj))
                dS.append(dSi)

    # D1 -> P1
    for Gi, G in enumerate(gate_vals):
        Gj = Gi
        for Ni in range(2):
            Nj = Ni
            # Get entropy production from this transition
            Wij = gamma * (1. - 1. / (np.exp(beta*(epsilonP-Vd))+1.))
            probi = vr[Gi, Ni, 1]
            probj = vr[Gj, Nj, 0]
            if probj > thresh:
                #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
                dSi = -Wij * probj * (np.log(probi/probj))
                dS.append(dSi)

    # P1 -> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Ni in range(2):
            Nj = Ni
            # Get entropy production from this transition
            Wij = gamma*1./(np.exp(beta*(epsilonP+G/Cg))+1.)
            probi = vr[Gi+1, Ni, 0]
            probj = vr[Gi, Nj, 1]
            if probj > thresh:
                #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
                dSi = -Wij * probj * (np.log(probi/probj))
            dS.append(dSi)

    # G1 -> P1
    for Gi, G in enumerate(gate_vals[1:]):
        for Ni in range(2):
            Nj = Ni
            # Get entropy production from this transition
            Wij = gamma*(1-1./(np.exp(beta*(epsilonP+G/Cg))+1.))
            probi = vr[Gi, Ni, 1]
            probj = vr[Gi+1, Nj, 0]
            if probj > thresh:
                #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
                dSi = -Wij * probj * (np.log(probi/probj))
                dS.append(dSi)

    # N1 -> G1
    for Gi, G in enumerate(gate_vals[:-1]):
        for Pi in range(2):
            Pj = Pi
            # Get entropy production from this transition
            Wij = gamma*1./(np.exp(beta*(epsilonN+G/Cg))+1.)
            probi = vr[Gi+1, 0, Pi]
            probj = vr[Gi, 1, Pj]
            if probj > thresh:
                #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
                dSi = -Wij * probj * (np.log(probi/probj))
                dS.append(dSi)

    # G1 -> N1
    for Gi, G in enumerate(gate_vals[1:]):
        for Pi in range(2):
            Pj = Pi
            # Get entropy production from this transition
            Wij = gamma*(1-1./(np.exp(beta*(epsilonN+G/Cg))+1.))
            probi = vr[Gi, 1, Pi]
            probj = vr[Gi+1, 0, Pj]
            if probj > thresh:
                #dSi = -Wij * probj * (np.log(probi) - np.log(probj))
                dSi = -Wij * probj * (np.log(probi/probj))
                dS.append(dSi)

    return sum(dS)

def measure_heat_generation(vr, vl, params, gate_vals, op=None):
    vr = vr.copy()
    vl = vl.copy()
    Ngate = len(gate_vals)
    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))

    # Put into quimb tensors
    vr = qtn.MatrixProductState.from_dense(vr, [2, 2, Ngate])
    vl = qtn.MatrixProductState.from_dense(vl, [2, 2, Ngate])

    # Normalize
    norm = vr & vl
    norm = norm ^ all
    vr /= norm

    # Get the heat operator
    if op is None:
        op = get_dQ_operator(params, gate_vals, square=False, tensor=True)

    # Do the measurement
    dQ = dense_state_operator(vr, vl, op).real
    return dQ

def measure_N_occ(vr, vl, gate_vals, plot=False):
    vr = vr.copy()
    vl = vl.copy()
    Ngate = len(gate_vals)
    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))

    # Put into quimb tensors
    vr = qtn.MatrixProductState.from_dense(vr, [2, 2, Ngate])
    vl = qtn.MatrixProductState.from_dense(vl, [2, 2, Ngate])

    # Normalize
    norm = vr & vl
    norm = norm ^ all
    vr /= norm

    # Measure occupation of first gate
    opsN = get_ops(2)
    rhoN = nsite_operator(vr, vl, [0], [opsN[f'n1']]).real
    return rhoN

def measure_P_occ(vr, vl, gate_vals, plot=False):
    vr = vr.copy()
    vl = vl.copy()
    Ngate = len(gate_vals)
    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))

    # Put into quimb tensors
    vr = qtn.MatrixProductState.from_dense(vr, [2, 2, Ngate])
    vl = qtn.MatrixProductState.from_dense(vl, [2, 2, Ngate])

    # Normalize
    norm = vr & vl
    norm = norm ^ all
    vr /= norm

    # Measure occupation of first gate
    opsP = get_ops(2)
    rhoP = nsite_operator(vr, vl, [1], [opsP[f'n1']]).real
    return rhoP

def measure_gate_occ(vr, vl, gate_vals, plot=False):
    vr = vr.copy()
    vl = vl.copy()
    Ngate = len(gate_vals)
    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))

    # Put into quimb tensors
    vr = qtn.MatrixProductState.from_dense(vr, [2, 2, Ngate])
    vl = qtn.MatrixProductState.from_dense(vl, [2, 2, Ngate])

    # Normalize
    norm = vr & vl
    norm = norm ^ all
    vr /= norm

    # Measure occupation of first gate
    opsG = get_ops(Ngate)
    rhoG = np.zeros(Ngate)
    for occi in range(Ngate):
        rhoG[occi] = nsite_operator(vr, vl, [2], [opsG[f'n{occi}']]).real

    # Plot Densities
    if plot:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.semilogy(gate_vals, rhoG, 'b+')
        plt.show()

    return rhoG

def measure_target_probability_density(vr, vl, gate_vals, source_inds):
    vr = vr.copy()
    vl = vl.copy()

    Ngate = len(gate_vals)
    # Put the eigenstate back into a tensor
    vl = vl.reshape((2, 2, Ngate))
    vr = vr.reshape((2, 2, Ngate))

    # Put into quimb tensors
    vr = qtn.MatrixProductState.from_dense(vr, [2, 2, Ngate])
    vl = qtn.MatrixProductState.from_dense(vl, [2, 2, Ngate])

    # Normalize
    norm = vr & vl
    norm = norm ^ all
    vr /= norm

    # Truncate the tensors to only include gates of interest
    data = vl[2].data
    ind = vl[2].inds.index('k2')
    if ind == 0:
        data = data[source_inds, :]
    else:
        data = data[:, source_inds]
    vl[2].modify(data=data)

    data = vr[2].data
    ind = vr[2].inds.index('k2')
    if ind == 0:
        data = data[source_inds, :]
    else:
        data = data[:, source_inds]
    vr[2].modify(data=data)

    # Do the measurement
    prob = vr & vl
    prob = prob ^ all
    return prob

def get_time_evolution_operator(params, gate_vals, dt):
    # Get the generator
    W = get_generator(params, gate_vals)

    # Take the exponential of the generator
    U = expm(dt*W)

    return U

def get_absorbing_time_evolution_operator(params, gate_vals, dt, absorbing_inds):
    # Get the generator
    W = get_absorbing_generator(params, gate_vals, absorbing_inds)

    # Take the exponential of the generator
    U = expm(dt*W)

    return U

def time_evolution_step(params, v0, gate_vals, dt, U=None, absorbing_inds=None):
    if U is None:
        # Get the generator
        if absorbing_inds is None:
            W = get_generator(params, gate_vals)
        else:
            W = get_absorbing_generator(params, gate_vals, absorbing_inds)

        # Take the exponential of the generator
        U = expm(dt*W)

    # Apply the time evolution operator to the state
    v = np.dot(U, v0)

    # Renormalize
    #v /= np.dot(v, v.T)
    v /= np.sum(v)

    # Return the time evolved state
    return v

def absorbing_time_evolution_step(params, v0, gate_vals, dt, absorbing_inds, U=None):
    if U is None:
        # Get the generator
        W = get_absorbing_generator(params, gate_vals, absorbing_inds)

        # Take the exponential of the generator
        U = expm(dt*W)

    # Apply the time evolution operator to the state
    v = np.dot(U, v0)

    # Renormalize
    v /= np.dot(v, v.T)

    # Return the time evolved state
    return v
