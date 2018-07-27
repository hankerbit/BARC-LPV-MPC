#!/usr/bin/env python

'''
    File name: LMPC.py
    Author: Ugo Rosolia
    Email: ugo.rosolia@berkeley.edu
    Python Version: 2.7.12
'''
import sys
sys.path.append(sys.path[0]+'/ControllersObject')
sys.path.append(sys.path[0]+'/Utilities')
import datetime
import rospy
from trackInitialization import Map
from barc.msg import pos_info, ECU, prediction, SafeSetGlob
import numpy as np
import pdb
import pickle
from utilities import Regression
from dataStructures import LMPCprediction, EstimatorData, ClosedLoopDataObj
from PathFollowingLTI_MPC import PathFollowingLTI_MPC
from PathFollowingLTVMPC import PathFollowingLTV_MPC
from PathFollowingLPVMPC import PathFollowingLPV_MPC
from LMPC import ControllerLMPC
import matplotlib.pyplot as plt
from utilities import Curvature

def main():
    # Initializa ROS node
    rospy.init_node("LMPC")
    input_commands  = rospy.Publisher('ecu', ECU, queue_size=1)
    pred_treajecto  = rospy.Publisher('OL_predictions', prediction, queue_size=1)
    sel_safe_set    = rospy.Publisher('SS', SafeSetGlob, queue_size=1)

    mode            = rospy.get_param("/control/mode")

    loop_rate       = 30.0
    dt              = 1.0/loop_rate
    rate            = rospy.Rate(loop_rate)

    Steering_Delay  = 1


    ### BUILDING THE GRID FOR TUNNING:
    Data_for_RMSE = np.zeros((2000,3))
    VQ_ve       = np.array([100.0, 150.0, 200.0, 250.0])
    VQ_ye       = np.array([50.0, 100.0, 150.0, 200.0])
    VQ_thetae   = np.array([150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0])
    grid_length = len(VQ_ve)*len(VQ_ye)*len(VQ_thetae)
    Q_matrix_grid = np.zeros((grid_length,3))

    counter = 0
    for i in range(0, len(VQ_ve)):
        Q_ve = VQ_ve[i]
        for j in range(0, len(VQ_ye)):
            Q_ye = VQ_ye[j]
            for k in range(0, len(VQ_thetae)):
                Q_thetae = VQ_thetae[k]
                Q_matrix_grid[counter,:] = [ Q_ve, Q_ye, Q_thetae ]
                counter += 1
    #print Q_matrix_grid





    # Objects initializations
    SS_glob_sel     = SafeSetGlob()
    OL_predictions  = prediction()
    cmd             = ECU()                                              # Command message
    cmd.servo       = 0.0
    cmd.motor       = 0.0
    ClosedLoopData  = ClosedLoopDataObj(dt, 6000, 0)         # Closed-Loop Data
    estimatorData   = EstimatorData()
    map             = Map()                                              # Map
    
    PickController  = "LPV_MPC"

    first_it        = 1
    NumberOfLaps    = grid_length
    vt              = 1.0
    PathFollowingLaps = 0 #EA: With this at 0 skips the first PID lap

    # Initialize variables for main loop
    GlobalState     = np.zeros(6)
    LocalState      = np.zeros(6)
    HalfTrack       = 0
    LapNumber       = 0
    RunController   = 1
    PlotingCounter  = 0
    uApplied        = np.array([0.0, 0.0])
    oldU            = np.array([0.0, 0.0])

    RMSE_ve         = np.zeros(grid_length)
    RMSE_ye         = np.zeros(grid_length)
    RMSE_thetae     = np.zeros(grid_length)
    RMSE_matrix     = np.zeros(grid_length)


    # Loop running at loop rate
    TimeCounter = 0
    KeyInput = raw_input("Press enter to start the controller... \n")
    oneStepPrediction = np.zeros(6)



    ### CONTROLLER INITIALIZATION:
    ### 33 ms - 10 Hp
    Q  = np.diag([100.0, 1.0, 1.0, 50.0, 0.0, 1000.0])
    R  = 1 * np.diag([1.0, 0.5])  # delta, a
    dR = 30 * np.array([2.0, 1.5])  # Input rate cost u
    N  = 14
    Controller = PathFollowingLPV_MPC(Q, R, dR, N, vt, dt, map, "OSQP", Steering_Delay)
    print "Q matrix: \n",Q, "\n"



    while (not rospy.is_shutdown()) and RunController == 1:    
        # Read Measurements
        GlobalState[:] = estimatorData.CurrentState
        LocalState[:]  = estimatorData.CurrentState #The current estimated state vector

        # EA: s, ey, epsi
        LocalState[4], LocalState[5], LocalState[3], insideTrack = map.getLocalPosition(GlobalState[4], GlobalState[5], GlobalState[3])
        if LocalState[0]<0.01:
            LocalState[0] = 0.01

        # Check if the lap has finished
        if LocalState[4] >= 3*map.TrackLength/4:
            HalfTrack = 1
            

        ### END OF THE LAP:
        if HalfTrack==1 and (LocalState[4] <= map.TrackLength/4):
            HalfTrack = 0
            LapNumber += 1 

            print "Lap completed starting lap:", LapNumber, ". Lap time: ", float(TimeCounter)/loop_rate

            if LapNumber > 1:
                print "Recording RMSE... \n"
                PointAndTangent = map.PointAndTangent
                cur, vel = Curvature(LocalState[4], PointAndTangent)
                RMSE_ve[LapNumber-2] = np.sqrt((( vel-Data_for_RMSE[0:TimeCounter,0] ) ** 2).mean())
                RMSE_ye[LapNumber-2] = np.sqrt((Data_for_RMSE[0:TimeCounter,1] ** 2).mean())
                RMSE_thetae[LapNumber-2] = np.sqrt((Data_for_RMSE[0:TimeCounter,2] ** 2).mean())
                Data_for_RMSE = np.zeros((2000,3))

            TimeCounter = 0
            first_it    = 1
            Q  = np.diag([Q_matrix_grid[LapNumber-1,0], 1.0, 1.0, Q_matrix_grid[LapNumber-1,1], 0.0, Q_matrix_grid[LapNumber-1,2]])
            R  = 1 * np.diag([1.0, 0.5])  # delta, a
            dR = 30 * np.array([2.0, 1.5])  # Input rate cost u
            N  = 14
            Controller = PathFollowingLPV_MPC(Q, R, dR, N, vt, dt, map, "OSQP", Steering_Delay)
            print "Q matrix: \n",Q, "\n"


            if LapNumber >= NumberOfLaps:
                RunController = 0


        # If inside the track publish input
        if (insideTrack == 1):
            startTimer = datetime.datetime.now()
                                  
            oldU = uApplied
            uApplied = np.array([cmd.servo, cmd.motor])
            #Controller.OldInput = uApplied

            #print "Insert steering in the last space of the vector...", "\n"
            Controller.OldSteering.append(cmd.servo) # meto al final del vector
            Controller.OldAccelera.append(cmd.motor)
            #print "1o-Controller.OldSteering: ",Controller.OldSteering, "\n"
            #print "Remove the first element of the vector...", "\n"
            Controller.OldSteering.pop(0)
            Controller.OldAccelera.pop(0)    
            #print "2o-Controller.OldSteering: ",Controller.OldSteering, "\n"


            ### Publish input ###
            input_commands.publish(cmd)
            #print "Publishing: ", cmd.servo, "\n"

            oneStepPredictionError = LocalState - oneStepPrediction # Subtract the local measurement to the previously predicted one step

            uAppliedDelay = [Controller.OldSteering[-1 - Controller.steeringDelay], Controller.OldAccelera[-1]]
            #print "uAppliedDelay: ",Controller.OldSteering[-1 - Controller.steeringDelay], "\n"
                
            oneStepPrediction, oneStepPredictionTime = Controller.oneStepPrediction(LocalState, uAppliedDelay, 1)

            if first_it < 10:  # EA: Starting mode:
               
                xx = np.array([[LocalState[0]+0.05, LocalState[1], 0., 0.000001, LocalState[4], 0.0001],
                               [LocalState[0]+0.2, LocalState[1], 0., 0.000001, LocalState[4]+0.01, 0.0001],
                               [LocalState[0]+0.4, LocalState[1], 0., 0.000001, LocalState[4]+0.02, 0.0001],
                               [LocalState[0]+0.6, LocalState[1], 0., 0.000001, LocalState[4]+0.04, 0.0001],
                               [LocalState[0]+0.7, LocalState[1], 0., 0.000001, LocalState[4]+0.07, 0.0001],
                               [LocalState[0]+0.8, LocalState[1], 0., 0.000001, LocalState[4]+0.1, 0.0001],
                               [LocalState[0]+0.8, LocalState[1], 0., 0.000001, LocalState[4]+0.14, 0.0001],
                               [LocalState[0]+1.0,  LocalState[1], 0., 0.000001, LocalState[4]+0.18, 0.0001],
                               [LocalState[0]+1.05, LocalState[1], 0., 0.000001, LocalState[4]+0.23, 0.0001],
                               [LocalState[0]+1.05, LocalState[1], 0., 0.000001, LocalState[4]+0.55, 0.0001],
                               [LocalState[0]+1.05, LocalState[1], 0., 0.000001, LocalState[4]+0.66, 0.0001],
                               [LocalState[0]+1.05, LocalState[1], 0., 0.000001, LocalState[4]+0.77, 0.0001],
                               [LocalState[0]+1.05, LocalState[1], 0., 0.000001, LocalState[4]+0.89, 0.0001],
                               [LocalState[0]+1.0,  LocalState[1], 0., 0.000001, LocalState[4]+0.999, 0.0001]])
                uu = np.array([[0., 0. ],
                               [0., 0.3],
                               [0., 0.5],
                               [0., 0.7],
                               [0., 0.9],
                               [0., 1.0],
                               [0., 1.0],
                               [0., 1.0],
                               [0., 1.0],
                               [0., 0.8],
                               [0., 0.8],
                               [0., 0.8],
                               [0., 0.8],
                               [0., 0.8]])                       

                Controller.solve(LocalState, xx, uu)
                first_it = first_it + 1

                #print "Resolving MPC...", "\n "
                #print "Steering predicted: ",Controller.uPred[:,0] , "\n "
                #Controller.OldPredicted = np.hstack((Controller.OldSteering, Controller.uPred[Controller.steeringDelay:Controller.N-1,0]))
                #Controller.OldPredicted = np.concatenate((np.matrix(Controller.OldPredicted).T, np.matrix(Controller.uPred[:,1]).T), axis=1)
                
                Controller.OldPredicted = np.hstack((Controller.OldSteering[0:len(Controller.OldSteering)-1], Controller.uPred[Controller.steeringDelay:Controller.N,0]))
                Controller.OldPredicted = np.concatenate((np.matrix(Controller.OldPredicted).T, np.matrix(Controller.uPred[:,1]).T), axis=1)
                
                #print "OldPredicted: ",Controller.OldPredicted[:,0], "\n"


                # We store the predictions since we are going to rebuild the controller object:
                U_pred = Controller.uPred    
                last_Hp = len(uu)
            else:
                
                # # Recomputamos el objeto controller con nuevo Hp:
                # PointAndTangent = map.PointAndTangent
                # cur, vel = Curvature(LocalState[4], PointAndTangent)
                # print "Curvature: ",cur, "\n"
                # if cur > 0.1:
                #     Q  = np.diag([100.0, 1.0, 1.0, 20.0, 0.0, 700.0])
                #     R  = 0 * np.diag([1.0, 1.0])  # delta, a
                #     dR = 40 * np.array([2.0, 1.5])  # Input rate cost u
                #     N  = 10
                #     Controller = PathFollowingLPV_MPC(Q, R, dR, N, vt, dt, map, "OSQP", Steering_Delay)
                #     print "Hp = ", N, "\n"
                #     if last_Hp == 4:
                #         last_input = U_pred[3,:]
                #         for i in range(last_Hp, N):
                #             U_pred = np.vstack((U_pred, last_input)) 
                #         LPV_States_Prediction = Controller.LPVPrediction(LocalState, U_pred)
                #         Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, U_pred)
                #         U_pred = Controller.uPred
                #         #print U_pred
                #         last_Hp = N    
                #     else:                                
                #         LPV_States_Prediction = Controller.LPVPrediction(LocalState, U_pred)
                #         Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, U_pred)
                #         U_pred = Controller.uPred
                #         #print U_pred
                #         last_Hp = N
                # else:
                #     Q  = np.diag([100.0, 1.0, 1.0, 20.0, 0.0, 700.0])
                #     R  = 0 * np.diag([1.0, 1.0])  # delta, a
                #     dR = 40 * np.array([2.0, 1.5])  # Input rate cost u
                #     N  = 4
                #     Controller = PathFollowingLPV_MPC(Q, R, dR, N, vt, dt, map, "OSQP", Steering_Delay)
                #     print "Hp = ", N, "\n"
                #     LPV_States_Prediction = Controller.LPVPrediction(LocalState, U_pred)
                #     Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, U_pred)
                #     U_pred = Controller.uPred
                #     #print "U predicted:",U_pred, "\n"
                #     last_Hp = N


                # Old version:
                #LPV_States_Prediction = Controller.LPVPrediction(LocalState, Controller.uPred)
                #Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, Controller.uPred)
                #print "Resolving MPC...", "\n "
                LPV_States_Prediction = Controller.LPVPrediction(LocalState, Controller.OldPredicted)
                Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, Controller.OldPredicted)
            ###################################################################################################
            ###################################################################################################

            cmd.servo = Controller.uPred[0 + Controller.steeringDelay, 0]
            cmd.motor = Controller.uPred[0, 1]
            #print "Steering que SE PUBLICARA teniendo en cuenta el delay: ","\n", "------>", cmd.servo, "\n \n \n \n \n"


            ##################################################################
            # GETTING DATA FOR IDENTIFICATION:
            # cmd.servo = 0.5*np.cos(50*TimeCounter)
            # if LocalState[0]>1.6:
            #     cmd.motor = 0.0
            # else:
            #     cmd.motor = 1.0
            ##################################################################

            #print Controller.uPred[steps_Delay,0], Controller.uPred[0,0]

            # if (Controller.solverTime.total_seconds() + Controller.linearizationTime.total_seconds() + oneStepPredictionTime.total_seconds() > dt):
            #     print "NOT REAL-TIME FEASIBLE!!"
            #     print "Solver time: ", Controller.solverTime.total_seconds(), " Linearization Time: ", Controller.linearizationTime.total_seconds() + oneStepPredictionTime.total_seconds()
            

            endTimer = datetime.datetime.now(); 
            deltaTimer = endTimer - startTimer
            #print "Tot Solver Time: ", deltaTimer.total_seconds()


        else:   # If car out of the track
            cmd.servo = 0
            cmd.motor = 0
            input_commands.publish(cmd)

            print " Current Input: ", cmd.servo, cmd.motor
            print " X, Y State: ", GlobalState
            print " Current State: ", LocalState


        # Record Prediction
        OL_predictions.s = Controller.xPred[:, 4]
        OL_predictions.ey = Controller.xPred[:, 5]
        OL_predictions.epsi = Controller.xPred[:, 3]
        pred_treajecto.publish(OL_predictions)
        ClosedLoopData.addMeasurement(GlobalState, LocalState, uApplied, PlotingCounter, deltaTimer.total_seconds())
        Data_for_RMSE[TimeCounter,:] = [ LocalState[0], LocalState[5], LocalState[3] ]

        # Increase time counter and ROS sleep()
        TimeCounter = TimeCounter + 1
        PlotingCounter += 1
        rate.sleep()

    # Save Data
    file_data = open(sys.path[0]+'/data/'+mode+'/ClosedLoopData'+"LPV_MPC"+'.obj', 'wb')
    pickle.dump(ClosedLoopData, file_data)
    pickle.dump(Controller, file_data)
    #pickle.dump(OpenLoopData, file_data)


    print " \n \n \n Root Mean Squared Vel Error Normalised: \n", np.divide(RMSE_ve[0:LapNumber-1], np.amax(RMSE_ve[0:LapNumber-1])), "\n"
    print "Root Mean Squared Lateral Error Normalised: \n", np.divide(RMSE_ye[0:LapNumber-1], np.amax(RMSE_ye[0:LapNumber-1])), "\n"
    print "Root Mean Squared Theta Error Normalised: \n", np.divide(RMSE_thetae[0:LapNumber-1], np.amax(RMSE_thetae[0:LapNumber-1])), "\n"

    print len(RMSE_thetae[0:LapNumber-1])
    for k in range(0, len(RMSE_thetae[0:LapNumber-1])-1):
        RMSE_matrix[k] = RMSE_ve[k] + RMSE_ye[k] + RMSE_thetae[k]

    print "Best tunning: ", Q_matrix_grid[np.argmin(RMSE_matrix)], "\n \n \n"

    #plotTrajectory(map, ClosedLoopData)

    file_data.close()



# ===============================================================================================================================
# ==================================================== END OF MAIN ==============================================================
# ===============================================================================================================================


# def Q_Grid(VQ_ve, VQ_ye, VQ_thetae):
# for i in range(0, len(VQ_ve)):
#     Q_ve = VQ_ve[i]
#     for j in range(0, len(VQ_ye)):
#         Q_ye = VQ_ye[j]
#         for k in range(0, len(VQ_thetae)):
#             Q_thetae = VQ_thetae[k]
#             Q_matrix_grid[i+j+k,:] = [ Q_ve, Q_ye, Q_thetae ]

#     return Q_matrix_grid


def plotTrajectory(map, ClosedLoop):
    x = ClosedLoop.x
    x_glob = ClosedLoop.x_glob
    u = ClosedLoop.u
    time = ClosedLoop.SimTime
    it = ClosedLoop.iterations
    elapsedTime = ClosedLoop.elapsedTime
    print elapsedTime

    plt.figure(3)
    plt.plot(time[0:it], elapsedTime[0:it, 0])
    plt.ylabel('Elapsed Time')
    ax = plt.gca()
    ax.grid(True)

    plt.figure(2)
    plt.subplot(711)
    plt.plot(time[0:it], x[0:it, 0])
    plt.ylabel('vx')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(712)
    plt.plot(time[0:it], x[0:it, 1])
    plt.ylabel('vy')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(713)
    plt.plot(time[0:it], x[0:it, 2])
    plt.ylabel('wz')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(714)
    plt.plot(time[0:it], x[0:it, 3],'k')
    plt.ylabel('epsi')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(715)
    plt.plot(time[0:it], x[0:it, 5],'k')
    plt.ylabel('ey')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(716)
    plt.plot(time[0:it], u[0:it, 0], 'r')
    plt.ylabel('steering')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(717)
    plt.plot(time[0:it], u[0:it, 1], 'r')
    plt.ylabel('acc')
    ax = plt.gca()
    ax.grid(True)
    plt.show()



        
class PID:
    """Create the PID controller used for path following at constant speed
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, vt):
        """Initialization
        Arguments:
            vt: target velocity
        """
        self.vt = vt
        self.uPred = np.zeros([1,2])

        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.feasible = 1

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state position
        """
        vt = self.vt
        Steering = - 0.5 * 2.0 * x0[5] - 0.5 * x0[3]
        Accelera = 0.5 * 1.5 * (vt - x0[0])
        self.uPred[0, 0] = np.maximum(-0.6, np.minimum(Steering, 0.6)) + np.maximum(-0.45, np.min(np.random.randn() * 0.25, 0.45))
        self.uPred[0, 1] = np.maximum(-2.5, np.minimum(Accelera, 2.5)) + np.maximum(-0.2, np.min(np.random.randn() * 0.10, 0.2))

if __name__ == "__main__":

    try:    
        main()
        
    except rospy.ROSInterruptException:
        pass
