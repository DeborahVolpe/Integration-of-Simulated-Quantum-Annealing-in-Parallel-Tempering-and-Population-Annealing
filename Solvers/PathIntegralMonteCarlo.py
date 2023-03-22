import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpmath import *
from matplotlib import rc
import time

"""
19/11/2021
@author: Deborah
"""

class PathIntegralMonteCarloAlgorithm:
    def __init__(self, spinsNumberIn, JMatrixIn, HVectorIn, offset_In):

        try:
            self.spinsNumber = int(spinsNumberIn)
        except:
            return None


        #Matrix J
        self.JMatrix = JMatrixIn

        #Matrix dimension check
        if not np.shape(self.JMatrix)[0] == self.spinsNumber or not np.shape(self.JMatrix)[1] == self.spinsNumber:
            return None

        #Vector H
        self.HVector = HVectorIn

        #Vector dimention check
        if not np.shape(self.HVector)[0] == self.spinsNumber:
            return None

        # offset
        self.offset = offset_In

        # Empty list of final value to plot the comulative
        self.finalValues = []

        # Starting Field Gamma parameter: default value
        self.Gamma0 = 1

        # Temperature parameter: default one
        self.T = 1

        # Number of Replicas parameter: default one
        self.NumberOfReplica = 4

        # Number of Iteration: default value
        self.NumberOfIteration = 15

        # Number of Time: default value
        self.NumberOfTime = 100

        #GlobalOptimalConfigurations
        self.GlobalOptimalConfiguration = np.ones(self.spinsNumber)

        #GlobalOptimalValue
        self.GlobalOptimalValue = 0
        
        #Average time for execution
        self.AverageTimeForExecution = 0
        
        #Average time for replica update 
        self.AverageTimeForReplicaUpdate = 0
        
        #Average time for System update 
        self.AverageTimeForSystemUpdate = 0
        
        #Average time for spin update 
        self.AverageTimeForSpinUpdate = 0
        
        #Total Execution time
        self.TotalExecutionTime = 0
        
        #List of value (First Time)
        self.ListOfValueFirstTime = [0]
        
        #List of best value (First Time)
        self.ListOfBestValueFirstTime = [0]
        
        #List of value Average Times
        self.ListOfValueAverageTime = [0]
        
        #List of best value Average Time
        self.ListOfBestValueAverageTime = [0]


    def DefineParameter(self, **kwargs):
        for key, value in kwargs.items():
            # setting the parameters
            # Starting field Gamma
            if key == "Gamma0":
                try:
                    self.Gamma0 = float(value)
                except:
                    return -1

            # Temperature parameter
            if key == "T":
                try:
                    self.T = float(value)
                except:
                    return -1

            # Number of replicas parameter
            if key == "NumberOfReplica":
                try:
                    self.NumberOfReplica = int(value)
                except:
                    return -1


    def Solve(self, numberOfIterationIn, numberOfTimeIn):
        try:
            self.NumberOfIteration = int(numberOfIterationIn)
        except:
            return -1

        try:
            self.NumberOfTime = int(numberOfTimeIn)
        except:
            return -1
            
        #List of value (first Time) initialization
        self.ListOfValueFirstTime = [0]*self.NumberOfIteration
        
        #List of best value (first Time) initialization
        self.ListOfBestValueFirstTime = [0]*self.NumberOfIteration
        
        #List of value Average Times initialization
        self.ListOfValueAverageTime = [0]*self.NumberOfIteration
        
        #List of best value Average Time initialization
        self.ListOfBestValueAverageTime = [0]*self.NumberOfIteration

        ###############################################
        # Path Integral Quantum Monte Carlo algorithm #
        ##############################################
        
        #Start Total Time Measurement
        Start = time.time()
        
        #Temporary variables initialization
        BestOfIteration = 0
        AverageValueOfIteration = 0

        for f in range(self.NumberOfTime):
            # Start the iteration time measurement 
            StartIterationTime = time.time()
            
            # Starting value of Gamma
            Gamma = self.Gamma0

            #Starting value of T
            Temp = self.T*self.NumberOfIteration/(1*(1-1/8))

            # Correlation factor between replica
            Jplus = Temp/2*math.log(coth(Gamma/(self.NumberOfReplica*Temp)))

            # New random intialization at each run
            SpinsMatrix = np.random.choice([-1,1], size =(self.NumberOfReplica, self.spinsNumber))

            # Optimal Value of each run initialization
            OptimalValue = 0


            # Empty vector of the starting values
            EnergyOfEachReplica = []

            # Evaluation of the starting condition of each replicas
            ##################
            # To parallelize #
            ##################
            for i in range(self.NumberOfReplica):
                val = (np.matmul(SpinsMatrix[i], np.transpose(np.matmul(self.JMatrix, np.transpose(SpinsMatrix[i])))) + np.matmul(SpinsMatrix[i], np.transpose(self.HVector))).item(0)
                EnergyOfEachReplica.append(val)
                if i == 0 or OptimalValue > val:
                    OptimalConfiguration  = SpinsMatrix[i].copy()
                    OptimalValue = val

                    if f == 0 or self.GlobalOptimalValue > val:
                        self.GlobalOptimalValue = val
                        self.GlobalOptimalConfiguration = SpinsMatrix[i].copy()
            ##############################
            # To parallelize as possible #
            ##############################
            for i in range(self.NumberOfIteration):
                # Start the system update time
                StartUpdateTime = time.time()
                
                # For each replica
                for j in range(self.NumberOfReplica):
                    # Start the replica update time 
                    StartTheReplicaUpdateTime = time.time()

                    # left neighbors replica individuation
                    if j != (self.NumberOfReplica - 1):
                        replica_sx = j + 1
                    else:
                        replica_sx = 0

                    # right neighbor replica individuation
                    if j != 0:
                        replica_dx = j-1
                    else:
                        replica_dx = self.NumberOfReplica - 1

                    # For each spin of the replica
                    for k in range(self.spinsNumber):
                        # Start the spin update time
                        StartTheSpinUpdateTime = time.time()
                        
                        u = random.uniform(0, 1)

                        # deltaH_potential_evaluation
                        sum = 0
                        for h in range(self.spinsNumber):
                            if h != k:
                                # I consider the matrix in the symmetrical form
                                sum += 2*SpinsMatrix.item((j, h))*self.JMatrix.item((k, h))
                        deltaH_potential = (sum+self.HVector.item(k))*((-1)*SpinsMatrix.item((j,k))-SpinsMatrix.item((j,k)))
                        deltaH_kin = Jplus*(SpinsMatrix.item((replica_sx, k))+SpinsMatrix.item((replica_dx, k)))*((-1)*SpinsMatrix.item((j,k))-SpinsMatrix.item((j,k)))
                        deltaH = deltaH_potential/self.NumberOfReplica+deltaH_kin
                        #deltaH = deltaH_potential/self.NumberOfReplica+deltaH_kin

                        if deltaH_potential < 0 or deltaH < 0 or u < math.exp(-deltaH*self.NumberOfReplica/Temp):
                        #if u < math.exp(-deltaH*self.NumberOfReplica):
                            # Flip accepted
                            SpinsMatrix.itemset((j, k), (-1)*SpinsMatrix.item(j, k))
                            EnergyOfEachReplica[j] = EnergyOfEachReplica [j] + deltaH_potential
                            #solution = SpinsMatrix[j].copy()
                            #value = (np.matmul(solution, np.transpose(np.matmul(self.JMatrix, np.transpose(solution)))) + np.matmul(solution, np.transpose(self.HVector))).item(0)
                            #if value != EnergyOfEachReplica[j]:
                            #    print("Error\n\n")
                            #EnergyOfEachReplica[j] = value
                        
                        # Stop the spin update time
                        StopTheSpinUpdateTime = time.time()
                        
                        # Compute Spin Update Time
                        SpinUpdateTime = StopTheSpinUpdateTime - StartTheSpinUpdateTime
                        
                        # Compute the sum of the spin update time
                        self.AverageTimeForSpinUpdate += SpinUpdateTime
                    # Evaluation of the flip effect
                    #solution = SpinsMatrix[j].copy()
                    #value = (np.matmul(solution, np.transpose(np.matmul(self.JMatrix, np.transpose(solution)))) + np.matmul(solution, np.transpose(self.HVector))).item(0)
                    #print(value.item(0,0))
                    value = EnergyOfEachReplica[j]
                    
                    #BestOfIteration
                    if j == 0:
                        BestOfIteration = value + self.offset
                    elif value + self.offset < BestOfIteration:
                        BestOfIteration = value + self.offset
                        
                    #AverageValueOfIteration
                    if j == 0:
                        AverageValueOfIteration = value + self.offset
                    else:
                        AverageValueOfIteration += value + self.offset

                    # New optimal value found?
                    if value < OptimalValue:
                        OptimalValue = value
                        OptimalConfiguration = SpinsMatrix[j].copy()
                      
                    # Stop the replica update time 
                    StopTheReplicaUpdateTime = time.time()
                    
                    # ReplicaUpdateTime
                    ReplicaUpdateTime = StopTheReplicaUpdateTime - StartTheReplicaUpdateTime
                    
                    # Sum the replica update times
                    self.AverageTimeForReplicaUpdate += ReplicaUpdateTime
                    
                # Stop the system update time
                StopUpdateTime = time.time()
                
                # System updateTime
                SystemUpdateTime = StopUpdateTime - StartUpdateTime
                
                # Sum the system update time
                self.AverageTimeForSystemUpdate += SystemUpdateTime
                
                if f == 0:
                    #List of value (First Time)
                    self.ListOfValueFirstTime[i] = AverageValueOfIteration/self.NumberOfReplica
                    #List of best value (First Time)
                    self.ListOfBestValueFirstTime[i] = BestOfIteration
                    #List of value Average Times
                    self.ListOfValueAverageTime[i] = (AverageValueOfIteration/self.NumberOfReplica)/self.NumberOfTime
                    #List of best value Average Time
                    self.ListOfBestValueAverageTime[i] = BestOfIteration/self.NumberOfTime
                else:
                    #List of value Average Times
                    self.ListOfValueAverageTime[i] += (AverageValueOfIteration/self.NumberOfReplica)/self.NumberOfTime
                    #List of best value Average Time
                    self.ListOfBestValueAverageTime[i] += BestOfIteration/self.NumberOfTime
     
                 #Temp update
                Temp = self.T*self.NumberOfIteration/((i+1)*(1-1/8))

                # Gamma Update
                Gamma = self.Gamma0*(1- i/(self.NumberOfIteration+1))
                Jplus = Temp/2*math.log(coth(Gamma/(self.NumberOfReplica*Temp)))

            self.finalValues.append(OptimalValue+self.offset)
                       
            # Stop the iteration time measurement 
            StopIterationTime = time.time()
            
            # Compute the iteration time
            IterationTime = StopIterationTime - StartIterationTime
            
            # Compute the sum of the time for execution
            self.AverageTimeForExecution += IterationTime

            if OptimalValue < self.GlobalOptimalValue:
                self.GlobalOptimalValue = OptimalValue
                self.GlobalOptimalConfiguration = OptimalConfiguration.copy()

        self.GlobalOptimalValue += self.offset
        
        # Stop the total time execution measurement
        Stop = time.time()
        
        # Compute the total execution time 
        self.TotalExecutionTime = Stop - Start
        
        # Compute the average time for iteration
        self.AverageTimeForExecution = self.AverageTimeForExecution/self.NumberOfTime
        
        # Compute the average time for system update
        self.AverageTimeForSystemUpdate = self.AverageTimeForSystemUpdate/(self.NumberOfTime*self.NumberOfIteration)
        
        # Compute the average time for replica update
        self.AverageTimeForReplicaUpdate = self.AverageTimeForReplicaUpdate/(self.NumberOfTime*self.NumberOfIteration*self.NumberOfReplica)
        
        # Compute the average time for spin update
        self.AverageTimeForSpinUpdate = self.AverageTimeForSpinUpdate/(self.NumberOfTime*self.NumberOfIteration*self.NumberOfReplica*self.spinsNumber)

        return self.GlobalOptimalConfiguration

    def Value(self):
        return self.GlobalOptimalValue

    def Plot(self, FileName="", Last=True, Latex=False, Save=False):
        if Latex == True:
            rc('text', usetex=True)
            plt.rc('text', usetex=True)
            n, bins, patches = plt.hist(self.finalValues, cumulative=True, histtype='step', linewidth=2,bins=100, label=r'\textit{Path Integral Quantum Monte Carlo}')
            plt.title(r'\textbf{Cumulative distribution}',fontsize=20)
            plt.xlabel(r'\textit{Values obtained}', fontsize=20)
            plt.ylabel(r'\textit{occurrence}', fontsize=20)
        else:
            n, bins, patches = plt.hist(self.finalValues, cumulative=True, histtype='step', linewidth=2,bins=100, label="Path Integral Quantum Monte Carlo")
            plt.title("Cumulative distribution",fontsize=20)
            plt.xlabel("Values obtained", fontsize=20)
            plt.ylabel("occurrence", fontsize=20)        
        leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
        leg.get_frame().set_facecolor('white')
        if Save == True:
            plt.savefig(FileName + ".eps", format='eps')
            plt.savefig(FileName + ".png", format='png')
            plt.savefig(FileName + ".pdf", format='pdf')
            if Last == True:
                plt.close()
        else:
            if Last == True:
                plt.show()

    def SuccessProbability(self, **kwargs):
        OptimalValue = self.GlobalOptimalValue
        for key, value in kwargs.items():
            # setting the parameters
            # Starting field OptimalValue
            if key == "OptimalValue":
                try:
                    OptimalValue = float(value)
                except:
                    return -1
        return self.finalValues.count(OptimalValue)/len(self.finalValues)

    def AverageValue(self):
        return sum(self.finalValues)/len(self.finalValues)

    def WriteResultsInAFile(self, nomeFileReport,  **kwargs):
        OptimalValue = self.GlobalOptimalValue
        for key, value in kwargs.items():
            # setting the parameters
            # Starting field OptimalValue
            if key == "OptimalValue":
                try:
                    OptimalValue = float(value)
                except:
                    return -1
        try:
            ReportFile = open(nomeFileReport, "w")
        except:
            return None

        ReportFile.write("----------Report of simulation of Path Integral Quantum Monte Carlo-----------\n")
        ReportFile.write("The number of considered Spin " + format(self.spinsNumber) + "\n")
        ReportFile.write("The considered Annealing Duration " + format(self.NumberOfIteration) + "\n")
        ReportFile.write("The considered Annealing Number " + format(self.NumberOfTime) + "\n")
        ReportFile.write("The considered T " + format(self.T) + "\n")
        ReportFile.write("The considered Gamma0 " + format(self.Gamma0) + "\n")
        ReportFile.write("The considered Number Of Replica " + format(self.NumberOfReplica) + "\n")
        ReportFile.write("The solution found " + format(self.GlobalOptimalConfiguration) + "\n")
        ReportFile.write("The value found " + format(self.GlobalOptimalValue) + "\n")
        ReportFile.write("The Success probability is " + format(self.SuccessProbability(OptimalValue=OptimalValue)) + "\n")
        ReportFile.write("The average value is " + format(self.AverageValue()) + "\n")
        ReportFile.write("The total execution time is " + format(self.TotalExecutionTime) + "\n")
        ReportFile.write("The average time for execution is " + format(self.AverageTimeForExecution) + "\n")  
        ReportFile.write("The average time for system update is " + format(self.AverageTimeForSystemUpdate) + "\n")
        ReportFile.write("The average time for replica update is " + format(self.AverageTimeForReplicaUpdate) + "\n")
        ReportFile.write("The average time for spin update is " + format(self.AverageTimeForSpinUpdate) + "\n")
        ReportFile.write("The list of values found \n\n")
        for i in range(len(self.finalValues)):
            ReportFile.write(format(self.finalValues[i]) + "\n")
        ReportFile.write("\n\n\nThe list of values first execution \n\n")
        for i in range(len(self.ListOfValueFirstTime)):
            ReportFile.write(format(self.ListOfValueFirstTime[i]) + "\n")
        ReportFile.write("\n\n\nThe list of best values first execution \n\n")
        for i in range(len(self.ListOfBestValueFirstTime)):
            ReportFile.write(format(self.ListOfBestValueFirstTime[i]) + "\n")
        ReportFile.write("\n\n\nThe list of values average \n\n")
        for i in range(len(self.ListOfValueAverageTime)):
            ReportFile.write(format(self.ListOfValueAverageTime[i]) + "\n")
        ReportFile.write("\n\n\nThe list of best values average \n\n")
        for i in range(len(self.ListOfBestValueAverageTime)):
            ReportFile.write(format(self.ListOfBestValueAverageTime[i]) + "\n")

        ReportFile.close()

    def TakeFinalValues(self):
        return self.finalValues
        
    def TakeMeasuredTimes(self):
        return self.TotalExecutionTime, self.AverageTimeForExecution, self.AverageTimeForSystemUpdate, self.AverageTimeForReplicaUpdate, self.AverageTimeForSpinUpdate 

    def TakeListOfValueFirstTime(self):
        #List of value (First Time)
        return self.ListOfValueFirstTime
        
    def TakeListOfBestValueFirstTime(self):
        #List of best value (First Time)
        return self.ListOfBestValueFirstTime
        
    def TakeListOfValueAverageTime(self):    
        #List of value Average Times
        return self.ListOfValueAverageTime
        
    def TakeListOfBestValueAverageTime(self):
        #List of best value Average Time
        return self.ListOfBestValueAverageTime  

    def PlotConvergence(self, FileName="", VFT=True, BVFT=True, VAT=True, BVAT=True, Last=True, Latex=False, Save=False):
        Iterations = range(self.NumberOfIteration)
        if Latex == True:
            rc('text', usetex=True)
            plt.rc('text', usetex=True)
            if VFT == True:
                plt.plot(Iterations, self.ListOfValueFirstTime, color='r', linewidth=2,label=r'\textit{Energy first time}')
            if BVFT == True:
                plt.plot(Iterations, self.ListOfBestValueFirstTime, color='m', linewidth=2,label=r'\textit{Best energy first time}')
            if VAT == True:
                plt.plot(Iterations, self.ListOfValueAverageTime, color='b', linewidth=2,label=r'\textit{Average energy}')
            if BVAT == True:
                plt.plot(Iterations, self.ListOfBestValueAverageTime, color='c', linewidth=2,label=r'\textit{Average best energy}')
            plt.title(r'\textbf{Energy evolution}',fontsize=20)
            plt.xlabel(r'\textit{Iteration}', fontsize=20)
            plt.ylabel(r'\textit{Energy}', fontsize=20)
        else:
            if VFT == True:
                plt.plot(Iterations, self.ListOfValueFirstTime, color='r', linewidth=2,label="Energy first time")
            if BVFT == True:
                plt.plot(Iterations, self.ListOfBestValueFirstTime, color='m', linewidth=2,label="Best energy first time")
            if VAT == True:
                plt.plot(Iterations, self.ListOfValueAverageTime, color='b', linewidth=2,label="Average energy")
            if BVAT == True:
                plt.plot(Iterations, self.ListOfBestValueAverageTime, color='c', linewidth=2,label="Average best energy")
            plt.title("Energy evolution",fontsize=20)
            plt.xlabel("Iteration", fontsize=20)
            plt.ylabel("Energy", fontsize=20)      
        leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
        leg.get_frame().set_facecolor('white')
        if Save == True:
            plt.savefig(FileName + ".eps", format='eps')
            plt.savefig(FileName+ ".png", format='png')
            plt.savefig(FileName + ".pdf", format='pdf')
            if Last == True:
                plt.close()
        else:
            if Last == True:
                plt.show()        


