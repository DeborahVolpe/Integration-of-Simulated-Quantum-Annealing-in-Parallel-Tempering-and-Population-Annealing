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

class PathIntegralMonteCarloAlgorithmPopulation:
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
        
        # Number of Replicas parameter: default one
        self.NumberOfReplica = 4
        
        # Numeber of System parameter: default one
        self.NumberOfSystem = 4

        # Temperature parameter: default one
        self.T = 10

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
        
        #Average time for All Systems update 
        self.AverageTimeForAllSystemsUpdate = 0
        
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
                    
            # Number of system parameter					
            if key == "NumberOfSystem":
                try:
                    self.NumberOfSystem = int(value)
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

        ###########################################
        # Quantum Population annealing algorithm  #
        ###########################################
        
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
            
            # Initialization of matrix spins
            SpinsMatrix = np.random.choice([-1, 1], size=(self.NumberOfSystem, self.NumberOfReplica, self.spinsNumber))
            BetterValueOfEachSystem = []
            for i in range(self.NumberOfSystem):
                BetterValueOfEachSystem.append(0)

            # Empty vector of the starting values

            #for i in range(self.NumberOfReplica):
                #zeros.append(0)
            EnergyOfEachSystemReplica = np.zeros((self.NumberOfSystem, self.NumberOfReplica))
            #for i in range(self.NumberOfSystem):
            #    EnergyOfEachSystemReplica.append(zeros)
            
            # Evaluation of the starting condition of each replicas
            ##################
            # To parallelize #
            ##################
            for i in range(self.NumberOfSystem):
                #EnergyOfEachSystemReplica.append(list())
                for j in range(self.NumberOfReplica):
                    val = (np.matmul( SpinsMatrix[i, j], np.transpose(np.matmul(self.JMatrix, np.transpose( SpinsMatrix[i, j])))) + np.matmul( SpinsMatrix[i, j], np.transpose(self.HVector))).item(0)
                    EnergyOfEachSystemReplica.itemset((i,j), val)
                    if (i == 0 and j==0) or OptimalValue > val:
                        OptimalConfiguration  = SpinsMatrix[i].copy()
                        OptimalValue = val
                    
                    #if (j==0) or BetterValueOfEachSystem[i] > val:
                        #BetterValueOfEachSystem[i] = val

                    if (f==0 and i==0 and j==0) or self.GlobalOptimalValue > val:
                        self.GlobalOptimalValue = val
                        self.GlobalOptimalConfiguration = SpinsMatrix[i].copy()
            ##############################
            # To parallelize as possible #
            ##############################                       
            for i in range(self.NumberOfIteration):
                # Start the system update time
                StartUpdateTime = time.time()
                
                # For each System
                for System in range(self.NumberOfSystem):
                   # Start the systems update time 
                    StartTheSystemsUpdateTime = time.time()
                    
                    tempEnergy = 0
                    # For each Replica
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
						
                        # For each Spin
                        for k in range(self.spinsNumber):
                           # Start the spin update time
                            StartTheSpinUpdateTime = time.time()
                            
                            u = random.uniform(0, 1)
                            # deltaH_potential_evaluation
                            sum = 0
                            for h in range(self.spinsNumber):
                                if h != k:
                                    sum += 2*SpinsMatrix.item((System, j, h)) * self.JMatrix.item((k, h))
                            deltaH_potential = (sum+self.HVector.item(k))*((-1)*SpinsMatrix.item((System,j,k))-SpinsMatrix.item((System,j,k)))
                            deltaH_kin = Jplus*(SpinsMatrix.item((System, replica_sx, k))+SpinsMatrix.item((System, replica_dx, k)))*((-1)*SpinsMatrix.item((System, j,k))-SpinsMatrix.item((System, j,k)))
                            deltaH = deltaH_potential/self.NumberOfReplica+deltaH_kin
                            #newVal = 2 * (self.HVector.item(k) + sum + Jplus * (SpinsMatrix.item((System, replica, k))+SpinsMatrix.item((System, replica2, k)))) - SpinsMatrix.item((System, j, k)) * self.T * (-math.log(u))

                            #if newVal > 0:
                                #SpinsMatrix.itemset((System, j, k), 1)
                            #else:
                                #SpinsMatrix.itemset((System, j, k), -1)

                            if deltaH_potential < 0 or deltaH < 0 or u < math.exp(-deltaH*self.NumberOfReplica/Temp):
                                # Flip accepted
                                SpinsMatrix.itemset((System, j, k), (-1)*SpinsMatrix.item(System, j, k))
                                prova = EnergyOfEachSystemReplica.item((System, j)) + deltaH_potential
                                EnergyOfEachSystemReplica.itemset((System, j), prova)
                                #solution = SpinsMatrix[System,j].copy()
                                #value = (np.matmul(solution, np.transpose(np.matmul(self.JMatrix, np.transpose(solution)))) + np.matmul(solution, np.transpose(self.HVector))).item(0)
                                #if abs(value-EnergyOfEachSystemReplica[System][j]) > 0.01:
                                #    print("Error\n\n")
                                    
                            # Stop the spin update time
                            StopTheSpinUpdateTime = time.time()

                            # Compute Spin Update Time
                            SpinUpdateTime = StopTheSpinUpdateTime - StartTheSpinUpdateTime
                            
                            # Compute the sum of the spin update time
                            self.AverageTimeForSpinUpdate += SpinUpdateTime
                        value = EnergyOfEachSystemReplica.item((System, j))
                        #solution = SpinsMatrix[System,j].copy()
                        #value = (np.matmul(solution, np.transpose(np.matmul(self.JMatrix, np.transpose(solution)))) + np.matmul(solution, np.transpose(self.HVector))).item(0)
                        #if ((j == 0)) or BetterValueOfEachSystem[System] > value.item(0):
                        #    BetterValueOfEachSystem[System] = value.item(0)
                        #if ValueOpt > value.item(0):
                        #    SolutionOpt = solution.copy()
                        #    ValueOpt = value.item(0)
                        #tempEnergy += value
                        if BetterValueOfEachSystem[System] > value or j == 0:
                            BetterValueOfEachSystem[System] = value     
                        #BestOfIteration
                        
                        if System == 0 and j == 0:
                            BestOfIteration = value + self.offset
                        elif value + self.offset < BestOfIteration:
                            BestOfIteration = value + self.offset
                            
                        #AverageValueOfIteration
                        if System == 0 and j == 0:
                            AverageValueOfIteration = value + self.offset
                        else:
                            AverageValueOfIteration += value + self.offset                                    
      
                        # New optimal value found?
                        if value < OptimalValue:
                            OptimalValue = value
                            OptimalConfiguration = SpinsMatrix[System,j].copy()
                            
                        # Stop the replica update time 
                        StopTheReplicaUpdateTime = time.time()
                        
                        # ReplicaUpdateTime
                        ReplicaUpdateTime = StopTheReplicaUpdateTime - StartTheReplicaUpdateTime
                        
                        # Sum the replica update times
                        self.AverageTimeForReplicaUpdate += ReplicaUpdateTime
                            
                    #BetterValueOfEachSystem[System] = tempEnergy/self.NumberOfSystem
                    
                    # Stop the system update time 
                    StopTheSystemsUpdateTime = time.time()
                    
                    # Compute the system update Time
                    SystemUpdateTime = StopTheSystemsUpdateTime - StartTheSystemsUpdateTime
                    
                    # Sum the replica update times
                    self.AverageTimeForSystemUpdate += SystemUpdateTime 

                # Check overflow
                m = max(list(map(abs, BetterValueOfEachSystem)))
                # Q factor accumulation
                Q = 0
                if i != self.NumberOfIteration-1:
                    for System in range(self.NumberOfSystem):
                        if m > 1000:
                            Q += math.exp((0.5*math.log(((math.sqrt(Gamma**2 +1)+1)/Gamma)**2)
                                       - 0.5*math.log(((math.sqrt((self.Gamma0*(1-(i+1)/(self.NumberOfIteration+1)))**2 +1)+1)
                                                       /(self.Gamma0*(1- (i+1)/(self.NumberOfIteration+1))))**2)) * (BetterValueOfEachSystem[System]/m))/self.NumberOfSystem
                        else:
                            Q += math.exp((0.5*math.log(((math.sqrt(Gamma**2 +1)+1)/Gamma)**2)
                                       - 0.5*math.log(((math.sqrt((self.Gamma0*(1-(i+1)/(self.NumberOfIteration+1)))**2 +1)+1)
                                                       /(self.Gamma0*(1- (i+1)/(self.NumberOfIteration+1))))**2)) * BetterValueOfEachSystem[System])/self.NumberOfSystem

                if Q != 0:
                    # Number of copy defined
                    systemCopy = 0
                    # Temporary copies and replicas configuration
                    temp = np.zeros((self.NumberOfSystem, self.NumberOfReplica, self.spinsNumber))
                    tempEnergyOfEachSystemReplica = np.zeros((self.NumberOfSystem, self.NumberOfReplica))
                    # Evaluation of number of copies to mantain for each system copy
                    for System in range(self.NumberOfSystem):
                        if m > 1000:
                            Nmean = (1/Q) * math.exp((0.5*math.log(((math.sqrt(Gamma**2 +1)+1)/Gamma)**2) - 0.5*math.log(((math.sqrt((self.Gamma0*(1- (i+1)/(self.NumberOfIteration+1)))**2 +1)+1)/(self.Gamma0*(1- (i+1)/(self.NumberOfIteration+1))))**2)) * BetterValueOfEachSystem[System]/m)
                        else:
                            Nmean = (1/Q) * math.exp((0.5*math.log(((math.sqrt(Gamma**2 +1)+1)/Gamma)**2) - 0.5*math.log(((math.sqrt((self.Gamma0*(1- (i+1)/(self.NumberOfIteration+1)))**2 +1)+1)/(self.Gamma0*(1- (i+1)/(self.NumberOfIteration+1))))**2)) * BetterValueOfEachSystem[System])
                        Number = np.random.poisson(Nmean, size=1)[0]
                        for g in range(Number):
                            if systemCopy < self.NumberOfSystem:
                                temp[systemCopy] = SpinsMatrix[System].copy()
                                for t in range(self.NumberOfReplica):
                                    tempEnergyOfEachSystemReplica.itemset((systemCopy, t), EnergyOfEachSystemReplica.item((System, t)))
                                systemCopy += 1
                    while systemCopy < self.NumberOfSystem:
                        temp[systemCopy] = SpinsMatrix[self.NumberOfSystem-1].copy()
                        for t in range(self.NumberOfReplica):
                            tempEnergyOfEachSystemReplica.itemset((systemCopy, t), EnergyOfEachSystemReplica.item((self.NumberOfSystem-1, t)))
                        systemCopy += 1

                    # Assign temporary copies and replicas configuration to copies and replica configuration
                    SpinsMatrix = temp.copy()
                    EnergyOfEachSystemReplica = tempEnergyOfEachSystemReplica.copy()
                    
                # Stop the system update time
                StopUpdateTime = time.time()
                
                # System updateTime
                SystemUpdateTime = StopUpdateTime - StartUpdateTime
                
                # Sum the system update time
                self.AverageTimeForAllSystemsUpdate += SystemUpdateTime

                if f == 0:
                    #List of value (First Time)
                    self.ListOfValueFirstTime[i] = AverageValueOfIteration/(self.NumberOfReplica*self.NumberOfSystem)
                    #List of best value (First Time)
                    self.ListOfBestValueFirstTime[i] = BestOfIteration
                    #List of value Average Times
                    self.ListOfValueAverageTime[i] = (AverageValueOfIteration/(self.NumberOfReplica*self.NumberOfSystem))/self.NumberOfTime
                    #List of best value Average Time
                    self.ListOfBestValueAverageTime[i] = BestOfIteration/self.NumberOfTime
                else:
                    #List of value Average Times
                    self.ListOfValueAverageTime[i] += (AverageValueOfIteration/(self.NumberOfReplica*self.NumberOfSystem))/self.NumberOfTime
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
        self.AverageTimeForAllSystemsUpdate = self.AverageTimeForAllSystemsUpdate/(self.NumberOfTime*self.NumberOfIteration)
        
        # Compute the average time for system update
        self.AverageTimeForSystemUpdate = self.AverageTimeForSystemUpdate/(self.NumberOfTime*self.NumberOfIteration*self.NumberOfSystem)
        
        # Compute the average time for replica update
        self.AverageTimeForReplicaUpdate = self.AverageTimeForReplicaUpdate/(self.NumberOfTime*self.NumberOfIteration*self.NumberOfReplica*self.NumberOfSystem)
        
        # Compute the average time for spin update
        self.AverageTimeForSpinUpdate = self.AverageTimeForSpinUpdate/(self.NumberOfTime*self.NumberOfIteration*self.NumberOfReplica*self.spinsNumber*self.NumberOfSystem)
        return self.GlobalOptimalConfiguration

    def Value(self):
        return self.GlobalOptimalValue

    def Plot(self, FileName="", Last=True, Latex=False, Save=False):
        if Latex == True:
            rc('text', usetex=True)
            plt.rc('text', usetex=True)
            n, bins, patches = plt.hist(self.finalValues, bins=100, cumulative=True, histtype='step', linewidth=2,label=r'\textit{Quantum Population Annealing}')#cumulative=True, histtype='step',
            plt.title(r'\textbf{Cumulative distribution}',fontsize=20)
            plt.xlabel(r'\textit{Values obtained}', fontsize=20)
            plt.ylabel(r'\textit{occurrence}', fontsize=20)
        else:
            n, bins, patches = plt.hist(self.finalValues, bins=100, cumulative=True, histtype='step', linewidth=2,label="Quantum Population Annealing")#cumulative=True, histtype='step',
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

        ReportFile.write("----------Report of simulation of Quantum Population Annealing-----------\n")
        ReportFile.write("The number of considered Spin " + format(self.spinsNumber) + "\n")
        ReportFile.write("The considered Annealing Duration " + format(self.NumberOfIteration) + "\n")
        ReportFile.write("The considered Annealing Number " + format(self.NumberOfTime) + "\n")
        ReportFile.write("The considered T " + format(self.T) + "\n")
        ReportFile.write("The considered Gamma0 " + format(self.Gamma0) + "\n")
        ReportFile.write("The considered Number Of Replica " + format(self.NumberOfReplica) + "\n")
        ReportFile.write("The considered Number Of System " + format(self.NumberOfSystem) + "\n")
        ReportFile.write("The solution found " + format(self.GlobalOptimalConfiguration) + "\n")
        ReportFile.write("The value found " + format(self.GlobalOptimalValue) + "\n")
        ReportFile.write("The Success probability is " + format(self.SuccessProbability(OptimalValue=OptimalValue)) + "\n")
        ReportFile.write("The average value is " + format(self.AverageValue()) + "\n")
        ReportFile.write("The total execution time is " + format(self.TotalExecutionTime) + "\n")
        ReportFile.write("The average time for execution is " + format(self.AverageTimeForExecution) + "\n")  
        ReportFile.write("The average time for all systems update is " + format(self.AverageTimeForAllSystemsUpdate) + "\n")
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
        return self.TotalExecutionTime, self.AverageTimeForExecution, self.AverageTimeForAllSystemsUpdate, self.AverageTimeForSystemUpdate, self.AverageTimeForReplicaUpdate, self.AverageTimeForSpinUpdate 
       
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