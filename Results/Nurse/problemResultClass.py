import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from SolversClass import SolversResults


"""
29/11/2021
@author: Deborah
"""


class ProblemResultNurse:
    def __init__(self, NurseIn, DayIn):
        
        self.TypeOfProblem = "Nurse"
        
        try:
            self.NumberOfNurse = int(NurseIn)
        except: 
            return None
        
        try:
            self.NumberOfDay = int(DayIn)
        except: 
            return None
            
        self.NumberOfVariables = (self.NumberOfNurse*self.NumberOfDay)
            
        self.SolversResultsList = {}
        
        self.ExpectedOptimalValue = 0
        
   
    def addSolverResult(self, SolverType, SolverResult):
        
        self.SolversResultsList[SolverType] = SolverResult
        

    def set_ExpectedOptimalValue(self, value):
            
        try:
            self.ExpectedOptimalValue = float(value) 
        except:
            self.ExpectedOptimalValue = None  

                
    def get_ExpectedOptimalValue(self):

        return self.ExpectedOptimalValue
        

    def min_reachedValue(self):
        
        First = True
        minimum = 0
        solverName = ""
        
        for solver in self.SolversResultsList.values():
            if First or minimum > solver.get_OptimalValueFound():
                minimum = solver.get_OptimalValueFound()
                solverName = solver.get_SolverType()
            First = False
                
        return minimum, solverName

                
    def max_reachedValue(self):
        
        First = True
        maximum = 0
        solverName = ""
        
        for solver in self.SolversResultsList.values():
            if First or maximum < solver.get_OptimalValueFound():
                maximum = solver.get_OptimalValueFound()
                solverName = solver.get_SolverType()
            First = False     
            
        return maximum, solverName
        
        
    def best_p(self, value):
    
        First = True    
        BestP = 0
        solverName = ""
        for solver in self.SolversResultsList.values():
            if First or BestP < solver.computeProbabilityOfBeLowerThenValue(value)*100:
                BestP = solver.computeProbabilityOfBeLowerThenValue(value)*100
                solverName = solver.get_SolverType()
            First = False   
        return BestP, solverName
        
        
        
    def get_numberOfNurse(self):
        return self.NumberOfNurse
        

    def get_numberOfDay(self):
        return self.NumberOfDay
        
        
    def get_numberOfVar(self):
        return self.NumberOfVariables   
        
     