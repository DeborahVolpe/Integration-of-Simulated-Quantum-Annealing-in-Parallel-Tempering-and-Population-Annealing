import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpmath import *
from mpmath import *
from matplotlib import rc
import time


"""
19/11/2021
@author: Deborah
"""



class PathIntegralMonteCarloAlgorithmPTempering:
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

        # Number of Iteration: default value
        self.NumberOfIteration = 15
        
        # Temperature parameter: default one
        self.T = 1
        self.Temp = np.linspace(self.T*self.NumberOfIteration/(1*(1-1/8)),self.T*self.NumberOfIteration/((self.NumberOfIteration+1)*(1-1/8)) , num=self.NumberOfSystem )
        
        # Gamma for each replicas evaluation
        self.ReplicaTemp = []
        self.Gamma = np.linspace(self.Gamma0, self.Gamma0*(1-self.NumberOfIteration/(self.NumberOfIteration+1)), num=self.NumberOfSystem)
        for i in range(self.NumberOfReplica):
            self.ReplicaTemp.append(i)

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

    def DefineParameter(self, **kwargs):
        for key, value in kwargs.items():
            # setting the parameters
            # Starting field Gamma
            if key == "Gamma0":
                try:
                    self.Gamma0 = float(value)
                    self.Gamma = np.linspace(self.Gamma0, self.Gamma0*(1-self.NumberOfIteration/(self.NumberOfIteration+1)), num=self.NumberOfSystem)
                except:
                    return -1
                    
            # Temperature parameter
            if key == "T":
                try:
                    self.T = float(value)
                    self.Temp = np.linspace(self.T*self.NumberOfIteration/(1*(1-1/8)),self.T*self.NumberOfIteration/((self.NumberOfIteration+1)*(1-1/8)) , num=self.NumberOfSystem )
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
                    self.Gamma = np.linspace(self.Gamma0, self.Gamma0*(1-self.NumberOfIteration/(self.NumberOfIteration+1)), num=self.NumberOfSystem)
                    self.Temp = np.linspace(self.T*self.NumberOfIteration/(1*(1-1/8)),self.T*self.NumberOfIteration/((self.NumberOfIteration+1)*(1-1/8)) , num=self.NumberOfSystem )
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

        #########################################
        # Quantum Parallel Tempering algorithm  #
        #########################################
        
        #Start Total Time Measurement
        Start = time.time()
        
        #Temporary variables initialization
        BestOfIteration = 0
        AverageValueOfIteration = 0
        
        for f in range(self.NumberOfTime):
            # Start the iteration time measurement 
            StartIterationTime = time.time()
            
            # Starting value of Gamma
            self.s = np.arange(0,self.NumberOfSystem)
            
                        # Initialization Gamma and T
            self.Temp = np.linspace(self.T*self.NumberOfIteration/(1*(1-1/8)),self.T*self.NumberOfIteration/((self.NumberOfIteration+1)*(1-1/8)) , num=self.NumberOfSystem )
            self.Gamma = np.linspace(self.Gamma0, self.Gamma0*(1-self.NumberOfIteration/(self.NumberOfIteration+1)), num=self.NumberOfSystem)

            # Correlation factor between replica
            Jplus = np.ones((1, self.NumberOfSystem))
            # Correlation factor between replica
            for t in range(self.NumberOfSystem):
                #Jplus.itemset((0, t), (self.T / 2) * math.log(coth(self.Gamma[self.s[t]] / (self.NumberOfReplica * self.T))))
                Jplus.itemset((0, t), (self.Temp[self.s[t]] / 2) * math.log(coth(self.Gamma[self.s[t]] / (self.NumberOfReplica * self.Temp[self.s[t]] ))))

            # Initialization of matrix spins
            SpinsMatrix = np.random.choice([-1, 1], size=(self.NumberOfSystem, self.NumberOfReplica, self.spinsNumber))
            BetterValueOfEachSystem = []
            for i in range(self.NumberOfSystem):
                BetterValueOfEachSystem.append(0)
            
            # Empty vector of the starting values
            #EnergyOfEachSystemReplica = []
            EnergyOfEachSystemReplica = np.zeros((self.NumberOfSystem, self.NumberOfReplica))
     
            # Evaluation of the starting condition of each replicas
            ##################
            # To parallelize #
            ##################
            for i in range(self.NumberOfSystem):
                #EnergyOfEachSystemReplica.append([])
                for j in range(self.NumberOfReplica):
                    val = (np.matmul( SpinsMatrix[i, j], np.transpose(np.matmul(self.JMatrix, np.transpose( SpinsMatrix[i, j])))) + np.matmul( SpinsMatrix[i, j], np.transpose(self.HVector))).item(0)
                    EnergyOfEachSystemReplica.itemset((i,j), val)
                    if (i == 0 and j==0) or OptimalValue > val:
                        OptimalConfiguration  = SpinsMatrix[i].copy()
                        OptimalValue = val
                    
                    if (j==0) or BetterValueOfEachSystem[i] > val:
                        BetterValueOfEachSystem[i] = val

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
                    
                    # For each Replica
                    tempEnergy = 0
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
                            deltaH_kin = Jplus[0,System]*(SpinsMatrix.item((System, replica_sx, k))+SpinsMatrix.item((System, replica_dx, k)))*((-1)*SpinsMatrix.item((System, j,k))-SpinsMatrix.item((System, j,k)))
                            deltaH = deltaH_potential/self.NumberOfReplica+deltaH_kin
                            #newVal = 2 * (self.HVector.item(k) + sum + Jplus[0,System] * (SpinsMatrix.item((System, replica, k))+SpinsMatrix.item((System, replica2, k)))) - SpinsMatrix.item((System, j, k)) * self.T * (-math.log(u))

                            #if newVal > 0:
                                #SpinsMatrix.itemset((System, j, k), 1)
                            #else:
                                #SpinsMatrix.itemset((System, j, k), -1)
                                
                            if deltaH_potential < 0 or deltaH < 0 or u < math.exp(-deltaH*self.NumberOfReplica/self.Temp[self.s[System]]):
                                # Flip accepted
                                SpinsMatrix.itemset((System, j, k), (-1)*SpinsMatrix.item(System, j, k))
                                prova = EnergyOfEachSystemReplica.item((System, j)) + deltaH_potential
                                EnergyOfEachSystemReplica.itemset((System, j), prova)
                                solution = SpinsMatrix[System,j].copy()
                                #value = (np.matmul(solution, np.transpose(np.matmul(self.JMatrix, np.transpose(solution)))) + np.matmul(solution, np.transpose(self.HVector)))
                                #if int(value) != int(EnergyOfEachSystemReplica[System][j]):
                                #    print("Error" + format(value) + " " + format(EnergyOfEachSystemReplica[System][j]) + "\n\n")
                                    
                            # Stop the spin update time
                            StopTheSpinUpdateTime = time.time()
                            
                            # Compute Spin Update Time
                            SpinUpdateTime = StopTheSpinUpdateTime - StartTheSpinUpdateTime
                            
                            # Compute the sum of the spin update time
                            self.AverageTimeForSpinUpdate += SpinUpdateTime
                        
                        value = EnergyOfEachSystemReplica.item((System, j))
                        #tempEnergy += value
                        #solution = SpinsMatrix[System,j].copy()
                        #value = np.matmul(solution, np.transpose(np.matmul(self.JMatrix, np.transpose(solution)))) + np.matmul(solution, np.transpose(self.HVector))
                        if BetterValueOfEachSystem[System] > value or j==0:
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

                for System1 in range(self.NumberOfSystem):
                    for System2 in range(self.NumberOfSystem):
                        if System != System2:
                            if m > 1000:
                                delta = (0.5*math.log(((math.sqrt(self.Gamma[self.s[System1]]**2 + 1)+1)/self.Gamma[self.s[System1]])**2) - 0.5*math.log(((math.sqrt(self.Gamma[self.s[System2]]**2 +1)+1)/self.Gamma[self.s[System2]])**2)) * (BetterValueOfEachSystem[System1] - BetterValueOfEachSystem[System2])/m
                            else:
                                delta = (0.5*math.log(((math.sqrt(self.Gamma[self.s[System1]]**2 + 1)+1)/self.Gamma[self.s[System1]])**2) - 0.5*math.log(((math.sqrt(self.Gamma[self.s[System2]]**2 +1)+1)/self.Gamma[self.s[System2]])**2)) * (BetterValueOfEachSystem[System1] - BetterValueOfEachSystem[System2])
                            if delta > 0: 
                                temp = self.s[System1]
                                self.s[System1] = self.s[System2]
                                self.s[System2] = temp
                            elif random.uniform(0, 1) < math.exp(delta):
                                temp = self.s[System1]
                                self.s[System1] = self.s[System2]
                                self.s[System2] = temp
                                
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
							
                for t in range(self.NumberOfSystem):
                    #Jplus.itemset((0, t), (self.T / 2) * math.log(coth(self.Gamma[self.s[t]] / (self.NumberOfReplica * self.T))))
                    Jplus.itemset((0, t), (self.Temp[self.s[t]] / 2) * math.log(coth(self.Gamma[self.s[t]] / (self.NumberOfReplica * self.Temp[self.s[t]] ))))
							
                    
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
        
        print(self.GlobalOptimalConfiguration)
        return self.GlobalOptimalConfiguration

    def Value(self):
        return self.GlobalOptimalValue

    def Plot(self, FileName="", Last=True, Latex=False, Save=False):
        if Latex == True:
            rc('text', usetex=True)
            plt.rc('text', usetex=True)
            #plt.xlim(min(self.finalValues), max(self.finalValues))   
            n, bins, patches = plt.hist(self.finalValues, bins=100, cumulative=True, histtype='step', linewidth=2,label=r'\textit{Quantum Parallel Tempering}')#cumulative=True, histtype='step',
            plt.title(r'\textbf{Cumulative distribution}',fontsize=20)
            plt.xlabel(r'\textit{Values obtained}', fontsize=20)
            plt.ylabel(r'\textit{occurrence}', fontsize=20)
        else: 
            #plt.xlim(min(self.finalValues), max(self.finalValues))   
            n, bins, patches = plt.hist(self.finalValues, bins=100, cumulative=True, histtype='step', linewidth=2,label="Quantum Parallel Tempering")#cumulative=True, histtype='step',
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

        ReportFile.write("----------Report of simulation of Quantum Parallel Tempering-----------\n")
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
        leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
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

