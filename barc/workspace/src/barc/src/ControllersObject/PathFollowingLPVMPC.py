#!/usr/bin/env python
"""
    File name: stateEstimator.py
    Author: Shuqi Xu and Ugo Rosolia
    Email: shuqixu@berkeley.edu (xushuqi8787@gmail.com)
    Modified: Eugenio Alcala
    Email: eugenio.alcala@upc.edu
    Python Version: 2.7.12
"""
from scipy import linalg, sparse
import numpy as np
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
from utilities import Curvature
import datetime
import numpy as np
from numpy import linalg as la
import pdb
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from numpy import tan, arctan, cos, sin, pi
import rospy

solvers.options['show_progress'] = False

class PathFollowingLPV_MPC:
    """Create the Path Following LMPC controller with LTV model
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, Q, R, dR, N, vt, dt, map, Solver, steeringDelay):
        """Initialization
        Q, R: weights to build the cost function h(x,u) = ||x||_Q + ||u||_R
        dR: weight to define the input rate cost h(x,u) = ||u_{k}-u_{k-1}||_dR
        N: horizon length
        vt: target velocity
        dt: discretization time
        map: map
        """
        self.A    = []
        self.B    = []
        self.C    = []
        self.N    = N
        self.n    = Q.shape[0]
        self.d    = R.shape[0]
        self.vt   = vt
        self.Q    = Q
        self.R    = R
        self.dR   = dR              # Slew rate
        self.LinPoints = np.zeros((self.N+2,self.n))
        self.dt = dt                # Sample time 33 ms
        self.map = map              # Used for getting the road curvature
        self.halfWidth = map.halfWidth

        self.first_it = 1

        self.steeringDelay = steeringDelay

        self.OldSteering = [0.0]*int(1 + steeringDelay)

        self.OldAccelera = [0.0]*int(1)        

        self.OldPredicted = [0.0]*int(1 + steeringDelay + N)
        #print np.shape(self.OldPredicted)

        self.Solver = Solver

        self.F, self.b = _buildMatIneqConst(self)

        self.G = []
        self.E = []

        # Vehicle parameters:
        self.lf = 0.125
        self.lr = 0.125
        self.m  = 1.98
        self.I  = 0.03
        self.Cf = 20
        self.Cr = 20




    def solve(self, x0, Last_xPredicted, uPred):
        """Computes control action
        Arguments:
            x0: current state position
            EA: Last_xPredicted: set of last predicted states used for updating matrix A LPV
            EA: uPred: set of last predicted control inputs used for updating matrix A LPV
        """

        PointAndTangent         = self.map.PointAndTangent
        s                       = Last_xPredicted[0,4]
        curv, vel               = Curvature(s, PointAndTangent)

        startTimer              = datetime.datetime.now()

        self.A, self.B          = _EstimateABC(self, Last_xPredicted, uPred)

#        self.G, self.E, self.Eu = _buildMatEqConst(self)
        self.G, self.E          = _buildMatEqConst(self)


        self.M, self.q          = _buildMatCost(self, uPred[0,:], vel)

        endTimer                = datetime.datetime.now()
        deltaTimer              = endTimer - startTimer
        self.linearizationTime  = deltaTimer

        M = self.M
        q = self.q
        G = self.G
        E = self.E
        F = self.F
        b = self.b
        n = self.n
        N = self.N
        d = self.d

        
        if self.Solver == "CVX":
            startTimer = datetime.datetime.now()
            sol = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0))
            endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
            self.solverTime = deltaTimer
            if sol['status'] == 'optimal':
                self.feasible = 1
            else:
                self.feasible = 0

            self.xPred = np.squeeze(np.transpose(np.reshape((np.squeeze(sol['x'])[np.arange(n * (N + 1))]), (N + 1, n))))
            self.uPred = np.squeeze(np.transpose(np.reshape((np.squeeze(sol['x'])[n * (N + 1) + np.arange(d * N)]), (N, d))))
        else:
            startTimer = datetime.datetime.now()
            # Adaptation for QSQP from https://github.com/alexbuyval/RacingLMPC/
            res_cons, feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F), b, sparse.csr_matrix(G), np.dot(E,x0) )
            #res_cons, feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F), b, sparse.csr_matrix(G), np.add( np.dot(E,x0), np.dot(Eu,uOld)) )
            Solution = res_cons.x

            endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
            self.solverTime = deltaTimer
            self.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n * (N + 1))]), (N + 1, n))))
            self.uPred = np.squeeze(np.transpose(np.reshape((Solution[n * (N + 1) + np.arange(d * N)]), (N, d))))

        self.LinPoints = np.concatenate( (self.xPred.T[1:,:], np.array([self.xPred.T[-1,:]])), axis=0 )
        self.xPred = self.xPred.T
        self.uPred = self.uPred.T



    def oneStepPrediction(self, x, u, UpdateModel = 0):
        """Propagate the model one step forward
        Arguments:
            x: current state
            u: current input
        """
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer

        if self.A == []:
            x_next = x
        else:
            x_next = np.dot(self.A[0], x) + np.dot(self.B[0], u) # EA: In discrete time
        return x_next, deltaTimer



    def LPVPrediction(self, x, u):
        """Propagate the model N steps forward
        Arguments:
            x: current state
            u: current set of input (predicted)
        """
        lf = self.lf
        lr = self.lr
        m = self.m
        I = self.I
        Cf = self.Cf
        Cr = self.Cr
        epss= 0.01
        STATES_vec = np.zeros((self.N+1, 6))

        for i in range(0, self.N):
            if i==0:
                states = np.reshape(x, (6,1))

            PointAndTangent = self.map.PointAndTangent
            vy = states[1] + 0.0001
            epsi = states[3]
            s = states[4]
            ey = states[5]
            cur, vel = Curvature(s, PointAndTangent)
            vx = vel + epss                # EA: This is using desired circuit velocity
            #vx = states[0] + epss         # EA: This is using past predictions
            delta = u[i,0]          # EA: steering angle at K-1

            A12 = (np.sin(delta) * Cf) / m
            A13 = (np.sin(delta) * Cf * lf) / m
            A22 = -(Cr + Cf * np.cos(delta)) / m
            A23 = -(lf * Cf * np.cos(delta) - lr * Cr) / m
            A32 = -(lf * Cf * np.cos(delta) - lr * Cr) / I
            A33 = -(lf * lf * Cf * np.cos(delta) + lr * lr * Cr) / I
            A5 = (1 / (1 - ey * cur)) * (-(vx * np.cos(epsi) - vy * np.sin(epsi))) * cur
            A6 = (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - ey * cur)
            A7 = vx * np.sin(epsi)
            A8 = vy * np.cos(epsi)
            B11 = -(np.sin(delta) * Cf) / m
            B21 = (np.cos(delta) * Cf) / m
            B31 = (lf * Cf * np.cos(delta)) / I
            
            Ai = np.array([[0., A12 / vx, (A13 / vx) + vy, 0., 0., 0.],  # [vx]
                           [0., A22 / vx, (A23 / vx) - vx, 0., 0., 0.],  # [vy]
                           [0., A32 / vx, A33 / vx, 0., 0., 0.],  # [wz]
                           [A5 / vx, 0., 1., 0., 0., 0.],  # [epsi]
                           [A6 / vx, 0., 0., 0., 0., 0.],  # [s]
                           [A7 / vx, A8 / vy, 0., 0., 0., 0.]])  # [ey]

            Bi = np.array([[B11, 1],  # [delta, a]
                           [B21, 0],
                           [B31, 0],
                           [0, 0],
                           [0, 0],
                           [0, 0]])

            Ai = np.eye(len(Ai)) + self.dt * Ai
            Bi = self.dt * Bi

            # print "\n", np.shape(np.transpose(u[i,:]))
            # print np.shape(Bi)            
            # print np.shape(np.dot(Bi, np.transpose(u[i,:])))
            # print np.shape(np.dot(Ai, states))
            # print np.shape(np.dot(Ai, states) + np.dot(Bi, np.transpose(u[i,:])))
            
            states_new = np.dot(Ai, states) + np.dot(Bi, np.transpose(u[i,:]))

            STATES_vec[i] = np.reshape(states_new, (6,))

            states = states_new

        return STATES_vec
        
# ======================================================================================================================
# ======================================================================================================================
# =============================== Internal functions for MPC reformulation to QP =======================================
# ======================================================================================================================
# ======================================================================================================================

def osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    # EA: P represents the quadratic weight composed by N times Q and R matrices.
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using OSQP <https://github.com/oxfordcontrol/osqp>.
    Parameters
    ----------
    P : scipy.sparse.csc_matrix Symmetric quadratic-cost matrix.
    q : numpy.array Quadratic cost vector.
    G : scipy.sparse.csc_matrix Linear inequality constraint matrix.
    h : numpy.array Linear inequality constraint vector.
    A : scipy.sparse.csc_matrix, optional Linear equality constraint matrix.
    b : numpy.array, optional Linear equality constraint vector.
    initvals : numpy.array, optional Warm-start guess vector.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
    """
    osqp = OSQP()
    if G is not None:
        l = -inf * ones(len(h))
        if A is not None:
            qp_A = vstack([G, A]).tocsc()
            qp_l = hstack([l, b])
            qp_u = hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l
            qp_u = h
        osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
    else:
        osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=True)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()

    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)
    feasible = 0
    if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
        feasible = 1
    return res, feasible



def _buildMatIneqConst(Controller):
    N = Controller.N
    n = Controller.n

    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    # Fx = np.array([[-1., 0., 0., 0., 0., 0.],
    #               [0., 0., 0., 0., 0., 1.],
    #               [0., 0., 0., 0., 0., -1.]])
    # # bx = np.array([[-0.01],  # vx max
    # #               [Controller.halfWidth],  # max ey
    # #               [Controller.halfWidth]])  # min ey
    # bx = np.array([[-0.01],  # vx max
    #               [5.5],  # max ey
    #               [5.5]])  # min ey

    Fx = np.array([[-1., 0., 0., 0., 0., 0.]])
    bx = np.array([[-0.01]]) # vx min

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[0.5],  # Max Steering
                   [0.5],  # Max Steering
                   [1.0],  # Max Acceleration
                   [1.0]])  # Max DesAcceleration


    # Now stuck the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    rep_a = [Fx] * (N)
    Mat = linalg.block_diag(*rep_a)
    NoTerminalConstr = np.zeros((np.shape(Mat)[0], n))  # No need to constraint also the terminal point
    Fxtot = np.hstack((Mat, NoTerminalConstr))
    bxtot = np.tile(np.squeeze(bx), N)

    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu] * (N)
    Futot = linalg.block_diag(*rep_b)
    butot = np.tile(np.squeeze(bu), N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack((Fxtot, np.zeros((rFxtot, cFutot))))
    Dummy2 = np.hstack((np.zeros((rFutot, cFxtot)), Futot))
    F = np.vstack((Dummy1, Dummy2))
    b = np.hstack((bxtot, butot))

    if Controller.Solver == "CVX":
        F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0].astype(int), np.nonzero(F)[1].astype(int), F.shape)
        F_return = F_sparse
    else:
        F_return = F
    
    return F_return, b



def _buildMatCost(Controller, uOld, vel):
    # EA: This represents to be: [(r-x)^T * Q * (r-x)] up to N+1
    # and [u^T * R * u] up to N

    Q  = Controller.Q
    n  = Q.shape[0]
    R  = Controller.R
    dR = Controller.dR
    P  = Controller.Q
    N  = Controller.N
    #vt = Controller.vt
    vt = vel

    uOld  = [Controller.OldSteering[0], Controller.OldAccelera[0]]

    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    #c = [R] * (N)
    c = [R + 2 * np.diag(dR)] * (N) # Need to add dR for the derivative input cost

    Mu = linalg.block_diag(*c)
    # Need to condider that the last input appears just once in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

    # Derivative Input Cost
    OffDiaf = -np.tile(dR, N-1)
    np.fill_diagonal(Mu[2:], OffDiaf)
    np.fill_diagonal(Mu[:, 2:], OffDiaf)

    # This is without slack lane:
    M0 = linalg.block_diag(Mx, P, Mu)

    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M0)

    # Derivative Input
    q[n*(N+1):n*(N+1)+2] = -2 * np.dot( uOld, np.diag(dR) )

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost


    if Controller.Solver == "CVX":
        M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0].astype(int), np.nonzero(M)[1].astype(int), M.shape)
        M_return = M_sparse
    else:
        M_return = M

    return M_return, q



def _buildMatEqConst(Controller):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon

    A = Controller.A
    B = Controller.B
    N = Controller.N
    n = Controller.n
    d = Controller.d

    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    #E = np.zeros((n * (N + 1), n))
    E = np.zeros((n * (N + 1) + Controller.steeringDelay, n)) #new
    E[np.arange(n)] = np.eye(n)

    #Eu = np.zeros((n * (N + 1) + Controller.steeringDelay, d))

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A[i]
        Gu[np.ix_(ind1, ind2u)] = -B[i]

    G = np.hstack((Gx, Gu))

    # Delay implementation:
    if Controller.steeringDelay > 0:
        xZerosMat = np.zeros((Controller.steeringDelay, n *(N+1)))
        uZerosMat = np.zeros((Controller.steeringDelay, d * N))
        for i in range(0, Controller.steeringDelay):
            ind2Steer = i * d
            uZerosMat[i, ind2Steer] = 1.0        
        Gdelay = np.hstack((xZerosMat, uZerosMat))
        G = np.vstack((G, Gdelay))
    
    # if Controller.Solver == "CVX":
    #     G_sparse = spmatrix(G[np.nonzero(G)], np.nonzero(G)[0].astype(int), np.nonzero(G)[1].astype(int), G.shape)
    #     E_sparse = spmatrix(E[np.nonzero(E)], np.nonzero(E)[0].astype(int), np.nonzero(E)[1].astype(int), E.shape)
    #     G_return = G_sparse
    #     E_return = E_sparse
    # else:
    #     G_return = G
    #     E_return = E

    #return G, E, Eu
    return G, E



def _EstimateABC(Controller,Last_xPredicted, uPredicted):

    N = Controller.N
    dt  = Controller.dt
    lf  = Controller.lf
    lr  = Controller.lr
    m   = Controller.m
    I   = Controller.I
    Cf  = Controller.Cf
    Cr  = Controller.Cr
    epss= 0.01  
    Atv = []
    Btv = []

    for i in range(0, N):

        #############################################
        # Here comes everything of Eugenio:
        #############################################
        ## States:
        ##   long velocity    [vx]
        ##   lateral velocity [vy]
        ##   angular velocity [wz]
        ##   theta error      [epsi]
        ##   distance traveled[s]
        ##   lateral error    [ey]
        ##
        ## Control actions:
        ##   Steering angle   [delta]
        ##   Acceleration     [a]
        ##
        ## Scheduling variables:
        ##   vx
        ##   vy
        ##   epsi
        ##   ey
        ##   cur
        #############################################

        PointAndTangent = Controller.map.PointAndTangent
        
        vy = Last_xPredicted[i,1] + 0.0001
        epsi = Last_xPredicted[i,3]
        s = Last_xPredicted[i,4]
        ey = Last_xPredicted[i, 5]
        cur, vel = Curvature(s, PointAndTangent)
        vx = vel + epss
        #vx = Last_xPredicted[i,0] + epss
        delta = uPredicted[i,0]             #EA: set of predicted steering angles

        A12 = (np.sin(delta)*Cf)/m
        A13 = (np.sin(delta)*Cf*lf)/m
        A22 = -(Cr + Cf*np.cos(delta))/ m
        A23 = -(lf*Cf*np.cos(delta) - lr*Cr) / m
        A32 = -(lf*Cf*np.cos(delta) - lr*Cr) / I
        A33 = -(lf*lf*Cf*np.cos(delta) + lr*lr*Cr)  / I
        A5 = (1/(1-ey*cur)) * (-(vx*np.cos(epsi) - vy*np.sin(epsi))) * cur
        A6 = (vx*np.cos(epsi) - vy*np.sin(epsi)) / (1-ey*cur)
        A7 = vx * np.sin(epsi)
        A8 = vy * np.cos(epsi)
        B11 = -(np.sin(delta)*Cf)/m
        B21 = (np.cos(delta)*Cf)/ m
        B31 = (lf*Cf*np.cos(delta)) / I

        Ai = np.array([[0., A12 / vx, (A13 / vx) + vy, 0., 0., 0.],  # [vx]
                       [0., A22 / vx, (A23 / vx) - vx, 0., 0., 0.],  # [vy]
                       [0., A32 / vx, A33 / vx, 0., 0., 0.],  # [wz]
                       [A5 / vx, 0., 1., 0., 0., 0.],  # [epsi]
                       [A6 / vx, 0., 0., 0., 0., 0.],  # [s]
                       [A7 / vx, A8 / vy, 0., 0., 0., 0.]])  # [ey]

        Bi  = np.array([[ B11, 1 ], #[delta, a]
                        [ B21, 0 ],
                        [ B31, 0 ],
                        [ 0,   0 ],
                        [ 0,   0 ],
                        [ 0,   0 ]])

        Ai = np.eye(len(Ai)) + dt * Ai
        Bi = dt * Bi

        #############################################
        Atv.append(Ai)
        Btv.append(Bi)

    return Atv, Btv


