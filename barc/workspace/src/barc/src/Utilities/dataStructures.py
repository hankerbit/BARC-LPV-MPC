import numpy as np
import rospy
from barc.msg import pos_info, ECU, prediction, SafeSetGlob


class LMPCprediction():
    """Object collecting the predictions and SS at eath time step
    """
    def __init__(self, N, n, d, TimeLMPC, numSS_Points, Laps):
        """
        Initialization:
            N: horizon length
            n, d: input and state dimensions
            TimeLMPC: maximum simulation time length [s]
            num_SSpoints: number used to buils SS at each time step
        """
        self.oneStepPredictionError = np.zeros((n, TimeLMPC, Laps))
        self.PredictedStates        = np.zeros((N+1, n, TimeLMPC, Laps))
        self.PredictedInputs        = np.zeros((N, d, TimeLMPC, Laps))

        self.SSused   = np.zeros((n , numSS_Points, TimeLMPC, Laps))
        self.Qfunused = np.zeros((numSS_Points, TimeLMPC, Laps))

class EstimatorData(object):
    """Data from estimator"""
    def __init__(self):
        """Subscriber to estimator"""
        rospy.Subscriber("pos_info", pos_info, self.estimator_callback)
        self.CurrentState = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def estimator_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = [msg.v_x, msg.v_y, msg.psiDot, msg.psi, msg.x, msg.y]
        
class ClosedLoopDataObj():
    """Object collecting closed loop data points
    Attributes:
        updateInitialConditions: function which updates initial conditions and clear the memory
    """
    def __init__(self, dt, Time, v0):
        """Initialization
        Arguments:
            dt: discretization time
            Time: maximum time [s] which can be recorded
            v0: velocity initial condition
        """
        self.dt = dt
        self.Points = int(Time / dt)  # Number of points in the simulation
        self.u = np.zeros((self.Points, 2))  # Initialize the input vector
        self.x = np.zeros((self.Points + 1, 6))  # Initialize state vector (In curvilinear abscissas)
        self.x_glob = np.zeros((self.Points + 1, 6))  # Initialize the state vector in absolute reference frame
        #self.SimTime = 0
        self.SimTime = np.zeros((self.Points, 1))
        self.x[0,0] = v0
        self.x_glob[0,0] = v0
        self.CurrentState = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.iterations = 0
        self.elapsedTime = np.zeros((self.Points, 1))

    def updateInitialConditions(self, x, x_glob):
        """Clears memory and resets initial condition
        x: initial condition is the curvilinear reference frame
        x_glob: initial condition in the inertial reference frame
        """
        self.x[0, :] = x
        self.x_glob[0, :] = x_glob

        self.x[1:, :] = np.zeros((self.x.shape[0]-1, 6))
        self.x_glob[1:, :] = np.zeros((self.x.shape[0]-1, 6))
        self.SimTime[0,:] = 0

    def addMeasurement(self, xMeasuredGlob, xMeasuredLoc, uApplied, i, solverTime):
        """Add point to the object ClosedLoopData
        xMeasuredGlob: measured state in the inerial reference frame
        xMeasuredLoc: measured state in the curvilinear reference frame
        uApplied: input applied to the system
        """
        # self.x[self.SimTime, :]      = xMeasuredLoc
        # self.x_glob[self.SimTime, :] = xMeasuredGlob
        # self.u[self.SimTime, :]      = uApplied
        # self.SimTime = self.SimTime + 1

        self.iterations = i
        self.x[i, :]      = xMeasuredLoc
        self.x_glob[i, :] = xMeasuredGlob
        self.u[i, :]      = uApplied
        self.SimTime[i,:] = self.SimTime[i-1,:]+1*self.dt
        self.elapsedTime[i,:]= solverTime
