import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc


"""
29/11/2021
@author: Deborah
"""


class SolversResults:
    def __init__(self, solverType):
    
        self.solverType = solverType
    
        if self.solverType == "ParallelTempering":
            self.NumberOfReplicas = 0
            self.NumberOfSystemCopy = 0
        elif self.solverType == "PathIntegral":
            self.NumberOfReplicas = 0
        elif self.solverType == "PopulationAnnealing":
            self.NumberOfReplicas = 0
            self.NumberOfSystemCopy = 0
        elif self.solverType == "TemperingPopulationv1":
            self.NumberOfReplicas = 0
            self.NumberOfSystemTempering = 0
            self.NumberOfSystemPopulation = 0
        elif self.solverType == "TemperingPopulationv2":
            self.NumberOfReplicas = 0
            self.NumberOfSystemTempering = 0
            self.NumberOfSystemPopulation = 0
        else:
            return None
            
        self.NumberOfIteration = 0
        self.NumberOfTimes = 0
        self.SuccessProbability = 0
        self.AverageValue = 0
        self.OptimalValueFound = 0
        self.ValuesFound = []
        self.ListOfBestValueAverage = []
        
    def set_NumberOfReplicas(self, NumberOfReplicasIn):
        
        try:
            self.NumberOfReplicas = int(NumberOfReplicasIn) 
        except:
            self.NumberOfReplicas = 0
            
           
    def set_systemCopy(self, NumberOfSystemCopyIn):
        
        try:
            self.NumberOfSystemCopy = int(NumberOfSystemCopyIn) 
        except:
            self.NumberOfSystemCopy = 0
            
        
    def set_systemTempering(self, NumberOfSystemTemperingIn):
        
        try:
            self.NumberOfSystemTempering = int(NumberOfSystemTemperingIn) 
        except:
            self.NumberOfSystemTempering = 0
            
            
    def set_systemPopulation(self, NumberOfSystemPopulationIn):
        
        try:
            self.NumberOfSystemPopulation = int(NumberOfSystemPopulationIn) 
        except:
            self.NumberOfSystemPopulation = 0
            
            
    def set_NumberOfIteration(self, NumberOfIterationIn):
        
        try:
            self.NumberOfIteration = int(NumberOfIterationIn) 
        except:
            self.NumberOfIteration = 0
            
            
    def set_NumberOfTimes(self, NumberOfTimesnIn):
        
        try:
            self.NumberOfTimes = int(NumberOfTimesIn) 
        except:
            self.NumberOfTimes = 0
            
            
    def set_SuccessProbability(self, SuccessProbabilityIn):
        
        try:
            self.SuccessProbability = float(SuccessProbabilityIn) 
        except:
            self.SuccessProbability = 0
            
            
    def set_AverageValue(self, AverageValueIn):
        
        try:
            self.AverageValue = float(AverageValueIn) 
        except:
            self.AverageValue = 0
            
            
    def set_OptimalValueFound(self, OptimalValueFoundIn):
        
        try:
            self.OptimalValueFound = float(OptimalValueFoundIn) 
        except:
            self.OptimalValueFound = 0
            
            
            
    def set_ValuesFound(self, ValuesFoundIn):
        
        self.ValuesFound = ValuesFoundIn
        
        
        
    def set_BestValueAverage(self, BestValueAverageIn):
        
        self.ListOfBestValueAverage = BestValueAverageIn
        
        
    def computeProbabilityOfBeLowerThenValue(self, value):
        
        count = 0
        for val in self.ValuesFound:
            if val <= value:
                count += 1
                
        p = count/len(self.ValuesFound)
        
        return p
        
    def computeTotalNumberOfCopy(self):         
    
        if self.solverType == "ParallelTempering":
            return self.NumberOfReplicas*self.NumberOfSystemCopy
        elif self.solverType == "PathIntegral":
            return self.NumberOfReplicas
        elif self.solverType == "PopulationAnnealing":
            return self.NumberOfReplicas*self.NumberOfSystemCopy
        elif self.solverType == "TemperingPopulationv1":
            return self.NumberOfReplicas*(self.NumberOfSystemTempering + self.NumberOfSystemPopulation)
        elif self.solverType == "TemperingPopulationv2":
            return self.NumberOfReplicas*(self.NumberOfSystemTempering + self.NumberOfSystemPopulation)
        
        
    def returnTTS(self, value, targetProbability, Parallelization):
        
        p = self.computeProbabilityOfBeLowerThenValue(value)
        TotalNumberOfCopy = self.computeTotalNumberOfCopy()
        if p > 0 and p < 1:
            if Parallelization == False:
                TTS = self.NumberOfIteration * (math.log(1- targetProbability)/math.log(1-p))
            else:
                TTS = self.NumberOfIteration * (math.log(1- targetProbability)/math.log(1-p))*TotalNumberOfCopy
        elif p == 0:
            TTS = self.NumberOfIteration * (math.log(1- targetProbability)/math.log(1-0.001))
        else:
            if Parallelization == False:
                TTS = self.NumberOfIteration*(math.log(1- targetProbability)/math.log(1-0.99))
            else:
                TTS = self.NumberOfIteration*(math.log(1- targetProbability)/math.log(1-0.99))*TotalNumberOfCopy
        return TTS

    def get_NumberOfReplicas(self):
        
        return self.NumberOfReplicas 

            
           
    def get_systemCopy(self):
        
        return self.NumberOfSystemCopy

            
        
    def get_systemTempering(self):
        
        return self.NumberOfSystemTempering
       
            
    def get_systemPopulation(self):
        
        return self.NumberOfSystemPopulation
            
            
    def get_NumberOfIteration(self):

        return self.NumberOfIteration 
            
            
    def get_NumberOfTimes(self):

        return self.NumberOfTimes

            
    def get_SuccessProbability(self):
        
        return self.SuccessProbability 
            
            
    def get_AverageValue(self):
        
        return self.AverageValue  
            
            
    def get_OptimalValueFound(self):
        
        return self.OptimalValueFound
            
            
            
    def get_ValuesFound(self):
        
        return self.ValuesFound
        
        
        
    def get_BestValueAverage(self):
        
        return self.ListOfBestValueAverage
        
        
    def get_SolverType(self):
        
        return self.solverType
        
        