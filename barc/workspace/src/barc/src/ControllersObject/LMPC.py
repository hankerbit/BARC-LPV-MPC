import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from utilities import Curvature
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP

solvers.options['show_progress'] = False

class ControllerLMPC():
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """

    def __init__(self, numSS_Points, numSS_it, N, Qslack, Qlane, Q, R, dR, n, d, shift, dt,  map, Laps, TimeLMPC, Solver, SysID_Solver, flag_LTV):
        """Initialization
        Arguments:
            numSS_Points: number of points selected from the previous trajectories to build SS
            numSS_it: number of previois trajectories selected to build SS
            N: horizon length
            Q,R: weight to define cost function h(x,u) = ||x||_Q + ||u||_R
            dR: weight to define the input rate cost h(x,u) = ||u_{k}-u_{k-1}||_dR
            n,d: state and input dimensiton
            shift: given the closest point x_t^j to x(t) the controller start selecting the point for SS from x_{t+shift}^j
            map: map
            Laps: maximum number of laps the controller can run (used to avoid dynamic allocation)
            TimeLMPC: maximum time [s] that an lap can last (used to avoid dynamic allocation)
            Solver: solver used in the reformulation of the LMPC as QP
        """
        self.numSS_Points = numSS_Points
        self.numSS_it     = numSS_it
        self.N = N
        self.Qslack = Qslack
        self.Qlane = Qlane
        self.Q = Q
        self.R = R
        self.dR = dR
        self.n = n
        self.d = d
        self.shift = shift
        self.dt = dt
        self.map = map
        self.Solver = Solver            
        self.SysID_Solver = SysID_Solver
        self.A = []
        self.B = []
        self.C = []
        self.flag_LTV = flag_LTV
        self.halfWidth = map.halfWidth

        self.OldInput = np.zeros((1,2))

        self.MaxNumPoint = 40
        self.itUsedSysID = 2

        # Initialize the following quantities to avoid dynamic allocation
        NumPoints = int(TimeLMPC / dt) + 1
        self.TimeSS  = 10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.SS      = 10000 * np.ones((NumPoints, 6, Laps))    # Sampled Safe SS
        self.uSS     = 10000 * np.ones((NumPoints, 2, Laps))    # Input associated with the points in SS
        self.Qfun    =     0 * np.ones((NumPoints, Laps))       # Qfun: cost-to-go from each point in SS
        self.SS_glob = 10000 * np.ones((NumPoints, 6, Laps))    # SS in global (X-Y) used for plotting

        # Initialize the controller iteration
        self.it      = 0

        # Initialize pool for parallel computing used in the internal function _LMPC_EstimateABC
        # self.p = Pool(4)

        # Build matrices for inequality constraints
        self.F, self.b = _LMPC_BuildMatIneqConst(self)

    def solve(self, x0, uOld = np.zeros([0, 0])):
        """Computes control action
        Arguments:
            x0: current state position
        """
        n = self.n;        d = self.d
        F = self.F;        b = self.b
        SS = self.SS;      Qfun = self.Qfun; SS_glob = self.SS_glob
        uSS = self.uSS;    TimeSS = self.TimeSS
        Q = self.Q;        R = self.R
        dR =self.dR;       OldInput = self.OldInput
        N = self.N; dt = self.dt
        it           = self.it
        numSS_Points = self.numSS_Points
        shift       = self.shift
        Qslack       = self.Qslack

        LinPoints = self.LinPoints
        LinInput  = self.LinInput
        map = self.map

        # Select Points from SS
        SS_PointSelectedTot      = np.empty((n, 0))
        SS_glob_PointSelectedTot = np.empty((n, 0))
        Qfun_SelectedTot         = np.empty((0))
        for jj in range(0, self.numSS_it):
            SS_PointSelected, SS_glob_PointSelected, Qfun_Selected = _SelectPoints(SS, SS_glob, Qfun, it - jj - 1, x0, numSS_Points / self.numSS_it, shift)
            SS_PointSelectedTot      =  np.append(SS_PointSelectedTot, SS_PointSelected, axis=1)
            SS_glob_PointSelectedTot =  np.append(SS_glob_PointSelectedTot, SS_glob_PointSelected, axis=1)
            Qfun_SelectedTot         =  np.append(Qfun_SelectedTot, Qfun_Selected, axis=0)

        self.SS_PointSelectedTot      = SS_PointSelectedTot
        self.SS_glob_PointSelectedTot = SS_glob_PointSelectedTot
        self.Qfun_SelectedTot         = Qfun_SelectedTot
        # Run System ID
        startTimer = datetime.datetime.now()
        self.A, self.B, self.C, indexUsed_list = _LMPC_EstimateABC(self)
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        L, npG, npE = _LMPC_BuildMatEqConst(self, self.A, self.B, self.C, N, n, d)
        self.linearizationTime = deltaTimer

        # Build Terminal cost and Constraint
        G, E = _LMPC_TermConstr(self, npG, npE, N, n, d, SS_PointSelectedTot)
        M, q = _LMPC_BuildMatCost(self, Qfun_SelectedTot, numSS_Points, N, Qslack, Q, R, dR, OldInput)

        # Solve QP
        startTimer = datetime.datetime.now()

        if self.Solver == "CVX":
            res_cons = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0) + L)
            if res_cons['status'] == 'optimal':
                feasible = 1
            else:
                feasible = 0
            Solution = np.squeeze(res_cons['x'])     

        elif self.Solver == "OSQP":
            # Adaptarion for QSQP from https://github.com/alexbuyval/RacingLMPC/
            res_cons, feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F), b, sparse.csr_matrix(G), np.add(np.dot(E,x0),L[:,0]))
            Solution = res_cons.x

        self.feasible = feasible

        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        # Extract solution and set linerizations points
        xPred, uPred, lambd, slack = _LMPC_GetPred(Solution, n, d, N, np)

        self.xPred = xPred.T
        if self.N == 1:
            self.uPred    = np.array([[uPred[0], uPred[1]]])
            self.LinInput =  np.array([[uPred[0], uPred[1]]])
        else:
            self.uPred = uPred.T
            self.LinInput = np.vstack((uPred.T[1:, :], uPred.T[-1, :]))

        self.LinPoints = np.vstack((xPred.T[1:,:], xPred.T[-1,:]))

        self.OldInput = uPred.T[0,:]

    def addTrajectory(self, ClosedLoopData):
        """update iteration index and construct SS, uSS and Qfun
        Arguments:
            ClosedLoopData: ClosedLoopData object
        """
        it = self.it

        self.TimeSS[it] = ClosedLoopData.SimTime
        self.SS[0:(self.TimeSS[it] + 1), :, it] = ClosedLoopData.x[0:(self.TimeSS[it] + 1), :]
        self.SS_glob[0:(self.TimeSS[it] + 1), :, it] = ClosedLoopData.x_glob[0:(self.TimeSS[it] + 1), :]
        self.uSS[0:self.TimeSS[it], :, it]      = ClosedLoopData.u[0:(self.TimeSS[it]), :]
        self.Qfun[0:(self.TimeSS[it] + 1), it]  = _ComputeCost(ClosedLoopData.x[0:(self.TimeSS[it] + 1), :],
                                                              ClosedLoopData.u[0:(self.TimeSS[it]), :], self.map.TrackLength)
        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, it] == 0:
                self.Qfun[i, it] = self.Qfun[i - 1, it] - 1

        if self.it == 0:
            self.LinPoints = self.SS[1:self.N + 2, :, it]
            self.LinInput  = self.uSS[1:self.N + 1, :, it]

        self.it = self.it + 1

    def addPoint(self, x, x_glob, u, i):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
            i: at the j-th iteration i is the time at which (x,u) are recorded
        """
        Counter = self.TimeSS[self.it - 1]
        self.SS[Counter + i + 1, :, self.it - 1] = x + np.array([0, 0, 0, 0, self.map.TrackLength, 0])
        self.SS_glob[Counter + i + 1, :, self.it - 1] = x_glob
        self.uSS[Counter + i + 1, :, self.it - 1] = u
        if self.Qfun[Counter + i + 1, self.it - 1] == 0:
            self.Qfun[Counter + i + 1, self.it - 1] = self.Qfun[Counter + i, self.it - 1] - 1

    def update(self, SS, uSS, Qfun, TimeSS, it, LinPoints, LinInput):
        """update controller parameters. This function is useful to transfer information among LMPC controller
           with different tuning
        Arguments:
            SS: sampled safe set
            uSS: input associated with the points in SS
            Qfun: Qfun: cost-to-go from each point in SS
            TimeSS: time at which each j-th iteration is completed
            it: current iteration
            LinPoints: points used in the linearization and system identification procedure
            LinInput: inputs associated with the points used in the linearization and system identification procedure
        """
        self.SS  = SS
        self.uSS = uSS
        self.Qfun  = Qfun
        self.TimeSS  = TimeSS
        self.it = it

        self.LinPoints = LinPoints
        self.LinInput  = LinInput

    def oneStepPrediction(self, x, u, UpdateModel=0):
        """Propagate the model one step foreward
        Arguments:
            x: current state
            u: current input
            UpdateModel: optional flag, if set to 1 updates the model used in the propagation
        """
        startTimer = datetime.datetime.now()
        if self.A == []:
            x_next = x
        elif UpdateModel == 1:
            usedIt = range(self.it-self.itUsedSysID, self.it)
            self.LinPoints[0, :] = x
            self.LinInput[0, :]  = u

            Ai, Bi, Ci, indexSelected = RegressionAndLinearization(self.LinPoints, self.LinInput, usedIt, self.SS, self.uSS, self.TimeSS,
                                                       self.MaxNumPoint, self.n, self.d, matrix, self.map.PointAndTangent, self.dt, 0, self.SysID_Solver)
            x_next = np.dot(Ai, x) + np.dot(Bi, u) + np.squeeze(Ci)
        else:
            x_next = np.dot(self.A[0], x) + np.dot(self.B[0], u) + np.squeeze(self.C[0])            
        
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer

        return x_next, deltaTimer

# ======================================================================================================================
# ======================================================================================================================
# =============================== Internal functions for LMPC reformulation to QP ======================================
# ======================================================================================================================
# ======================================================================================================================

def osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
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
        osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=False)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()
    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)
    feasible = 0
    if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
        feasible = 1
    return res, feasible



def _LMPC_BuildMatCost(LMPC, Sel_Qfun, numSS_Points, N, Qslack, Q, R, dR, uOld):
    n = Q.shape[0]
    P = Q
    vt = 2
    Qlane = LMPC.Qlane

    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R + 2 * np.diag(dR)] * (N) # Need to add dR for the derivative input cost

    Mu = linalg.block_diag(*c)
    # Need to condider that the last input appears just once in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

    # Derivative Input Cost
    OffDiaf = -np.tile(dR, N-1)
    np.fill_diagonal(Mu[2:], OffDiaf)
    np.fill_diagonal(Mu[:, 2:], OffDiaf)
    # np.savetxt('Mu.csv', Mu, delimiter=',', fmt='%f')

    M00 = linalg.block_diag(Mx, P, Mu)
    quadLaneSlack = Qlane[0] * np.eye(2*LMPC.N)
    M0 = linalg.block_diag(M00, np.zeros((numSS_Points, numSS_Points)), Qslack, quadLaneSlack)
    # np.savetxt('M0.csv', M0, delimiter=',', fmt='%f')

    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q0 = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M00)


    # Derivative Input
    q0[n*(N+1):n*(N+1)+2] = -2 * np.dot( uOld, np.diag(dR) )

    # np.savetxt('q0.csv', q0, delimiter=',', fmt='%f')
    linLaneSlack = Qlane[1] * np.ones(2*LMPC.N)

    q = np.append(np.append(np.append(q0, Sel_Qfun), np.zeros(Q.shape[0])), linLaneSlack)

    # np.savetxt('q.csv', q, delimiter=',', fmt='%f')

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    if LMPC.Solver == "CVX":
        M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0].astype(int), np.nonzero(M)[1].astype(int), M.shape)
        M_return = M_sparse
    else:
        M_return = M

    return M_return, q



def _LMPC_BuildMatIneqConst(LMPC):
    N = LMPC.N
    n = LMPC.n
    numSS_Points = LMPC.numSS_Points
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])

    bx = np.array([[LMPC.halfWidth],  # max ey
                   [LMPC.halfWidth]])  # max ey

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[0.5],  # Max Steering
                   [0.5],  # Max Steering
                   [1.],  # Max Acceleration
                   [1.]])  # Max Acceleration



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

    FDummy = np.vstack((Dummy1, Dummy2))
    I = -np.eye(numSS_Points)
    FDummy2 = linalg.block_diag(FDummy, I)
    Fslack = np.zeros((FDummy2.shape[0], n))
    F_hard = np.hstack((FDummy2, Fslack))

    LaneSlack = np.zeros((F_hard.shape[0], 2*N))
    colIndex = range(2*N)
    rowIndex = []
    for i in range(0, N):
        rowIndex.append(i*Fx.shape[0] +0) # Slack on second element of Fx
        rowIndex.append(i*Fx.shape[0] +1) # Slack on third element of Fx
    LaneSlack[rowIndex, colIndex] = 1.0

    F = np.hstack((F_hard, LaneSlack))
    # np.savetxt('F.csv', F, delimiter=',', fmt='%f')
    # pdb.set_trace()



    # np.savetxt('F.csv', F, delimiter=',', fmt='%f')
    b = np.hstack((bxtot, butot, np.zeros(numSS_Points)))
    if LMPC.Solver == "CVX":
        F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0].astype(int), np.nonzero(F)[1].astype(int), F.shape)
        F_return = F_sparse
    else:
        F_return = F

    return F_return, b


def _SelectPoints(SS, SS_glob, Qfun, it, x0, numSS_Points, shift):
    x = SS[:, :, it]
    x_glob = SS_glob[:, :, it]
    oneVec = np.ones((x.shape[0], 1))
    x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T
    diff = x - x0Vec
    norm = la.norm(diff, 1, axis=1)
    MinNorm = np.argmin(norm)

    if (MinNorm + shift >= 0):
        SS_Points = x[shift + MinNorm:shift + MinNorm + numSS_Points, :].T
        SS_glob_Points = x_glob[shift + MinNorm:shift + MinNorm + numSS_Points, :].T
        Sel_Qfun = Qfun[shift + MinNorm:shift + MinNorm + numSS_Points, it]
    else:
        SS_Points = x[MinNorm:MinNorm + numSS_Points, :].T
        SS_glob_Points = x_glob[MinNorm:MinNorm + numSS_Points, :].T
        Sel_Qfun = Qfun[MinNorm:MinNorm + numSS_Points, it]

    return SS_Points, SS_glob_Points, Sel_Qfun

def _ComputeCost(x, u, TrackLength):
    Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
    # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    # We start from the last element of the vector x and we sum the running cost
    for i in range(0, x.shape[0]):
        if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
            Cost[x.shape[0] - 1 - i] = 0
        elif x[x.shape[0] - 1 - i, 4]< TrackLength:
            Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
        else:
            Cost[x.shape[0] - 1 - i] = 0

    return Cost


def _LMPC_TermConstr(LMPC, G, E, N ,n ,d , SS_Points):
    # Update the matrices for the Equality constraint in the LMPC. Now we need an extra row to constraint the terminal point to be equal to a point in SS
    # The equality constraint has now the form: G_LMPC*z = E_LMPC*x0 + TermPoint.
    # Note that the vector TermPoint is updated to constraint the predicted trajectory into a point in SS. This is done in the FTOCP_LMPC function

    TermCons = np.zeros((n, (N + 1) * n + N * d))
    TermCons[:, N * n:(N + 1) * n] = np.eye(n)

    G_enlarged = np.vstack((G, TermCons))

    G_lambda = np.zeros(( G_enlarged.shape[0], SS_Points.shape[1] + n))
    G_lambda[G_enlarged.shape[0] - n:G_enlarged.shape[0], :] = np.hstack((-SS_Points, np.eye(n)))

    G_LMPC0 = np.hstack((G_enlarged, G_lambda))
    G_ConHull = np.zeros((1, G_LMPC0.shape[1]))
    G_ConHull[-1, G_ConHull.shape[1]-SS_Points.shape[1]-n:G_ConHull.shape[1]-n] = np.ones((1,SS_Points.shape[1]))

    G_LMPC_hard = np.vstack((G_LMPC0, G_ConHull))

    SlackLane = np.zeros((G_LMPC_hard.shape[0], 2*N))
    G_LMPC = np.hstack((G_LMPC_hard, SlackLane))

    E_LMPC = np.vstack((E, np.zeros((n + 1, n))))

    # np.savetxt('G.csv', G_LMPC, delimiter=',', fmt='%f')
    # np.savetxt('E.csv', E_LMPC, delimiter=',', fmt='%f')

    if LMPC.Solver == "CVX":
        G_LMPC_sparse = spmatrix(G_LMPC[np.nonzero(G_LMPC)], np.nonzero(G_LMPC)[0].astype(int), np.nonzero(G_LMPC)[1].astype(int), G_LMPC.shape)
        E_LMPC_sparse = spmatrix(E_LMPC[np.nonzero(E_LMPC)], np.nonzero(E_LMPC)[0].astype(int), np.nonzero(E_LMPC)[1].astype(int), E_LMPC.shape)
        G_LMPC_return = G_LMPC_sparse
        E_LMPC_return = E_LMPC_sparse
    else:
        G_LMPC_return = G_LMPC
        E_LMPC_return = E_LMPC

    return G_LMPC_return, E_LMPC_return

def _LMPC_BuildMatEqConst(LMPC, A, B, C, N, n, d):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    L = np.zeros((n * (N + 1) + n + 1, 1)) # n+1 for the terminal constraint
    L[-1] = 1 # Summmation of lamba must add up to 1

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A[i]
        Gu[np.ix_(ind1, ind2u)] = -B[i]
        L[ind1, :]              =  C[i]

    G = np.hstack((Gx, Gu))


    if LMPC.Solver == "CVX":
        L_sparse = spmatrix(L[np.nonzero(L)], np.nonzero(L)[0].astype(int), np.nonzero(L)[1].astype(int), L.shape)
        L_return = L_sparse
    else:
        L_return = L

    return L_return, G, E

def _LMPC_GetPred(Solution,n,d,N, np):
    xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d*N)]),(N, d))))
    lambd = Solution[n*(N+1)+d*N:Solution.shape[0]-n]
    slack = Solution[Solution.shape[0]-n:]
    return xPred, uPred, lambd, slack

# ======================================================================================================================
# ======================================================================================================================
# ========================= Internal functions for Local Regression and Linearization ==================================
# ======================================================================================================================
# ======================================================================================================================
def _LMPC_EstimateABC(ControllerLMPC):
    LinPoints       = ControllerLMPC.LinPoints
    LinInput        = ControllerLMPC.LinInput
    N               = ControllerLMPC.N
    n               = ControllerLMPC.n
    d               = ControllerLMPC.d
    SS              = ControllerLMPC.SS
    uSS             = ControllerLMPC.uSS
    TimeSS          = ControllerLMPC.TimeSS
    PointAndTangent = ControllerLMPC.map.PointAndTangent
    dt              = ControllerLMPC.dt
    it              = ControllerLMPC.it
    SysID_Solver    = ControllerLMPC.SysID_Solver
    flag_LTV        = ControllerLMPC.flag_LTV
    MaxNumPoint     = ControllerLMPC.MaxNumPoint  # Need to reason on how these points are selected

    ParallelComputation = 0
    Atv = []; Btv = []; Ctv = []; indexUsed_list = []

    usedIt = range(ControllerLMPC.it-ControllerLMPC.itUsedSysID, ControllerLMPC.it)

    for i in range(0, N):
        if (i > 0) and (flag_LTV == False):
            Ai, Bi, Ci = Linearization(LinPoints, PointAndTangent, dt, i, Atv[0], Btv[0], Ctv[0])            
        else:
            Ai, Bi, Ci, indexSelected = RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                                                           MaxNumPoint, n, d, matrix, PointAndTangent, dt, i, SysID_Solver)
        Atv.append(Ai)
        Btv.append(Bi)
        Ctv.append(Ci)
        indexUsed_list.append(indexSelected)

    return Atv, Btv, Ctv, indexUsed_list

def RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS, MaxNumPoint, n, d, matrix, PointAndTangent, dt, i, SysID_Solver):


    x0 = LinPoints[i, :]

    Ai = np.zeros((n, n))
    Bi = np.zeros((n, d))
    Ci = np.zeros((n, 1))

    # Compute Index to use
    h = 5
    lamb = 0.0
    stateFeatures = [0, 1, 2]
    ConsiderInput = 1

    if ConsiderInput == 1:
        scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])
        xLin = np.hstack((LinPoints[i, stateFeatures], LinInput[i, :]))
    else:
        scaling = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        xLin = LinPoints[i, stateFeatures] 

    # startTimer = datetime.datetime.now()
    indexSelected = []
    K = []
    for i in usedIt:
        indexSelected_i, K_i = ComputeIndex(h, SS, uSS, TimeSS, i, xLin, stateFeatures, scaling, MaxNumPoint,
                                            ConsiderInput)
        indexSelected.append(indexSelected_i)
        K.append(K_i)
    # endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
    # print "Time Select Indeces: ", deltaTimer.total_seconds()
    # =========================
    # ====== Identify vx ======
    inputFeatures = [1]
    # startTimer = datetime.datetime.now()
    Q_vx, M_vx = Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K, SysID_Solver)
    # endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
    # print "Compute Q: ", deltaTimer.total_seconds()

    yIndex = 0
    # startTimer = datetime.datetime.now()
    b = Compute_b(SS, yIndex, usedIt, matrix, M_vx, indexSelected, K, np, SysID_Solver)
    # endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
    # print "Compute b: ", deltaTimer.total_seconds()

    # startTimer = datetime.datetime.now()
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_vx, b, stateFeatures,
                                                                                      inputFeatures, qp, SysID_Solver)
    # endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
    # print "Solve QP: ", deltaTimer.total_seconds()

    # =======================================
    # ====== Identify Lateral Dynamics ======
    inputFeatures = [0]
    Q_lat, M_lat = Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K, SysID_Solver)

    yIndex = 1  # vy
    b = Compute_b(SS, yIndex, usedIt, matrix, M_lat, indexSelected, K, np, SysID_Solver)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_lat, b, stateFeatures,
                                                                                      inputFeatures, qp, SysID_Solver)

    yIndex = 2  # wz
    b = Compute_b(SS, yIndex, usedIt, matrix, M_lat, indexSelected, K, np, SysID_Solver)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_lat, b, stateFeatures,
                                                                                      inputFeatures, qp, SysID_Solver)

    # ===========================
    # ===== Linearization =======
    vx = x0[0]; vy   = x0[1]
    wz = x0[2]; epsi = x0[3]
    s  = x0[4]; ey   = x0[5]

    if s < 0:
        print "s is negative, here the state: \n", LinPoints

    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    cur = Curvature(s, PointAndTangent)
    cur = Curvature(s, PointAndTangent)
    den = 1 - cur * ey
    # ===========================
    # ===== Linearize epsi ======
    # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
    depsi_vx = -dt * np.cos(epsi) / den * cur
    depsi_vy = dt * np.sin(epsi) / den * cur
    depsi_wz = dt
    depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
    depsi_s = 0  # Because cur = constant
    depsi_ey = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

    Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
    Ci[3]    = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur ) - np.dot(Ai[3, :], x0)
    # ===========================
    # ===== Linearize s =========
    # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
    ds_vx = dt * (np.cos(epsi) / den)
    ds_vy = -dt * (np.sin(epsi) / den)
    ds_wz = 0
    ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
    ds_s = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
    ds_ey = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den * 2) * (-cur)

    Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
    Ci[4]    = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) ) - np.dot(Ai[4, :], x0)

    # ===========================
    # ===== Linearize ey ========
    # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
    dey_vx = dt * np.sin(epsi)
    dey_vy = dt * np.cos(epsi)
    dey_wz = 0
    dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
    dey_s = 0
    dey_ey = 1

    Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
    Ci[5]    = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x0)

    endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

    return Ai, Bi, Ci, indexSelected

def Linearization(LinPoints, PointAndTangent, dt, i, Ai, Bi, Ci):


    x0 = LinPoints[i, :]

    # ===========================
    # ===== Linearization =======
    vx = x0[0]; vy   = x0[1]
    wz = x0[2]; epsi = x0[3]
    s  = x0[4]; ey   = x0[5]

    if s < 0:
        print "s is negative, here the state: \n", LinPoints

    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    cur = Curvature(s, PointAndTangent)
    cur = Curvature(s, PointAndTangent)
    den = 1 - cur * ey
    # ===========================
    # ===== Linearize epsi ======
    # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
    depsi_vx = -dt * np.cos(epsi) / den * cur
    depsi_vy = dt * np.sin(epsi) / den * cur
    depsi_wz = dt
    depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
    depsi_s = 0  # Because cur = constant
    depsi_ey = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

    Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
    Ci[3]    = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur ) - np.dot(Ai[3, :], x0)
    # ===========================
    # ===== Linearize s =========
    # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
    ds_vx = dt * (np.cos(epsi) / den)
    ds_vy = -dt * (np.sin(epsi) / den)
    ds_wz = 0
    ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
    ds_s = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
    ds_ey = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den * 2) * (-cur)

    Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
    Ci[4]    = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) ) - np.dot(Ai[4, :], x0)

    # ===========================
    # ===== Linearize ey ========
    # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
    dey_vx = dt * np.sin(epsi)
    dey_vy = dt * np.cos(epsi)
    dey_wz = 0
    dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
    dey_s = 0
    dey_ey = 1

    Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
    Ci[5]    = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x0)

    endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

    return Ai, Bi, Ci

def Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K, SysID_Solver):
    Counter = 0
    it = 1
    X0   = np.empty((0,len(stateFeatures)+len(inputFeatures)))
    Ktot = np.empty((0))

    for it in usedIt:
        X0 = np.append( X0, np.hstack((np.squeeze(SS[np.ix_(indexSelected[Counter], stateFeatures, [it])]),
                            np.squeeze(uSS[np.ix_(indexSelected[Counter], inputFeatures, [it])], axis=2))), axis=0)
        Ktot = np.append(Ktot, K[Counter])
        Counter = Counter + 1

    M = np.hstack((X0, np.ones((X0.shape[0], 1))))
    Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
    
    if SysID_Solver == "CVX":
        Q = matrix(Q0 + lamb * np.eye(Q0.shape[0]))
    else:
        Q = Q0 + lamb * np.eye(Q0.shape[0])

    return Q, M

def Compute_b(SS, yIndex, usedIt, matrix, M, indexSelected, K, np, SysID_Solver):
    Counter = 0
    y = np.empty((0))
    Ktot = np.empty((0))

    for it in usedIt:
        y = np.append(y, np.squeeze(SS[np.ix_(indexSelected[Counter] + 1, [yIndex], [it])]))
        Ktot = np.append(Ktot, K[Counter])
        Counter = Counter + 1

    if SysID_Solver == "CVX":
        b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))
    else:
        b = -np.dot(np.dot(M.T, np.diag(Ktot)), y)

    return b

def LMPC_LocLinReg(Q, b, stateFeatures, inputFeatures, qp, SysID_Solver):
    if SysID_Solver == "CVX":
        res_cons = qp(Q, b) # This is ordered as [A B C]
        Result = np.squeeze(np.array(res_cons['x']))
    elif SysID_Solver == "OSQP":
        res_cons, _ = osqp_solve_qp(sparse.csr_matrix(Q), b)
        Result = res_cons.x
    elif SysID_Solver == "scipy":
        res_cons = linalg.solve(Q, -b, sym_pos=True)
        Result = res_cons

    A = Result[0:len(stateFeatures)]
    B = Result[len(stateFeatures):(len(stateFeatures)+len(inputFeatures))]
    C = Result[-1]

    return A, B, C

def ComputeIndex(h, SS, uSS, TimeSS, it, x0, stateFeatures, scaling, MaxNumPoint, ConsiderInput):
    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration

    # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
    oneVec = np.ones( (SS[0:TimeSS[it], :, it].shape[0]-1, 1) )

    x0Vec = (np.dot( np.array([x0]).T, oneVec.T )).T

    if ConsiderInput == 1:
        DataMatrix = np.hstack((SS[0:TimeSS[it]-1, stateFeatures, it], uSS[0:TimeSS[it]-1, :, it]))
    else:
        DataMatrix = SS[0:TimeSS[it]-1, stateFeatures, it]

    diff  = np.dot(( DataMatrix - x0Vec ), scaling)
    # print 'x0Vec \n',x0Vec
    norm = la.norm(diff, 1, axis=1)
    indexTot =  np.squeeze(np.where(norm < h))
    # print indexTot.shape, np.argmin(norm), norm, x0
    if (indexTot.shape[0] >= MaxNumPoint):
        index = np.argsort(norm)[0:MaxNumPoint]
        # MinNorm = np.argmin(norm)
        # if MinNorm+MaxNumPoint >= indexTot.shape[0]:
        #     index = indexTot[indexTot.shape[0]-MaxNumPoint:indexTot.shape[0]]
        # else:
        #     index = indexTot[MinNorm:MinNorm+MaxNumPoint]
    else:
        index = indexTot

    K  = ( 1 - ( norm[index] / h )**2 ) * 3/4
    # K = np.ones(len(index))

    return index, K