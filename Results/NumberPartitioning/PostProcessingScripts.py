import os
import sys
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import cm
import shutil
import numpy as np
from SolversClass import SolversResults
from problemResultClass import ProblemResultNumberPartitioning
import collections
from scipy.interpolate import interp1d


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

rc('text', usetex=True)
plt.rc('text', usetex=True)

FolderProblem = "NumberPartitioningProblems/"

FolderResults = "NumberPartitioningProblemsResults/"

print("Start")


SizeForPlots = 200

n_point = 50

shutil.rmtree("PostProcessingResults", ignore_errors = True)
path = '.'
os.mkdir("PostProcessingResults")

    

# Reads problems
path = FolderProblem    
Folders = os.listdir(path)

ProblemsResults = {}


for file in Folders: 
  
    if file.endswith("_sol.txt"):

        try:
            problemSize = int(file.split("_")[1])
        except:
            print("Errore0")
            sys.exit()

            
        try: 
            f = open(FolderProblem + file)
        except: 
            print("Errore1")
            sys.exit()

        
        lines = f.readlines()
        
        try:
            Energy = float(lines[1].split(" ")[1])
        except:
            print("No energy for problem size" + format(problemSize) + "\n")
            

        try:
            offset = float(lines[1].split(" ")[1])
        except:
            print("No offset for problem size" + format(problemSize) + "\n")
           
        f.close() 
        
        Problem = ProblemResultNumberPartitioning(problemSize)
        
        Problem.set_ExpectedOptimalValue(Energy)
        
        Problem.set_offset(offset)
        
        ProblemsResults[problemSize] = Problem
       
       

# Reads results
path = FolderResults    
Folders = os.listdir(path)   

for file in Folders:     

    if file.endswith("_100.txt"):
        
        solversType = file.split("_")[3]
        
        problemSize = int(file.split("_")[1])
        
        Solver = SolversResults(solversType)
        
        if solversType == "ParallelTempering":
        
            NumberOfReplica = int(file.split("_")[7])
            NumberOfSystemCopy = int(file.split("_")[8])
            NumberOfIteration = int(file.split("_")[9])
            NumberOfTimes = int(file.split("_")[10][:-4])
            Solver.set_NumberOfReplicas(NumberOfReplica)
            Solver.set_systemCopy(NumberOfSystemCopy)
            Solver.set_NumberOfIteration(NumberOfIteration)
            Solver.set_NumberOfTimes(NumberOfTimes)
        
        elif solversType == "PopulationAnnealing":
        
            NumberOfReplica = int(file.split("_")[7])
            NumberOfSystemCopy = int(file.split("_")[8]) 
            NumberOfIteration = int(file.split("_")[9])
            NumberOfTimes = int(file.split("_")[10][:-4])
            Solver.set_NumberOfReplicas(NumberOfReplica)
            Solver.set_systemCopy(NumberOfSystemCopy)
            Solver.set_NumberOfIteration(NumberOfIteration)
            Solver.set_NumberOfTimes(NumberOfTimes)            

        elif solversType == "PathIntegral":
        
            NumberOfReplica = int(file.split("_")[7])
            NumberOfIteration = int(file.split("_")[8])
            NumberOfTimes = int(file.split("_")[9][:-4])
            Solver.set_NumberOfReplicas(NumberOfReplica)
            Solver.set_NumberOfIteration(NumberOfIteration)
            Solver.set_NumberOfTimes(NumberOfTimes)
            
        elif solversType == "TemperingPopulationv1":
        
            NumberOfReplica = int(file.split("_")[7])
            NumberOfSystemTempering = int(file.split("_")[8]) 
            NumberOfSystemPopulation = int(file.split("_")[9]) 
            NumberOfIteration = int(file.split("_")[10])
            NumberOfTimes = int(file.split("_")[11][:-4])
            Solver.set_NumberOfReplicas(NumberOfReplica)
            Solver.set_systemTempering(NumberOfSystemTempering)
            Solver.set_systemPopulation(NumberOfSystemPopulation)
            Solver.set_NumberOfIteration(NumberOfIteration)
            Solver.set_NumberOfTimes(NumberOfTimes)

        elif solversType == "TemperingPopulationv2":
        
            NumberOfReplica = int(file.split("_")[7])
            NumberOfSystemTempering = int(file.split("_")[8]) 
            NumberOfSystemPopulation = int(file.split("_")[9])
            NumberOfIteration = int(file.split("_")[10])
            NumberOfTimes = int(file.split("_")[11][:-4])
            Solver.set_NumberOfReplicas(NumberOfReplica)
            Solver.set_systemTempering(NumberOfSystemTempering)
            Solver.set_systemPopulation(NumberOfSystemPopulation)
            Solver.set_NumberOfIteration(NumberOfIteration)
            Solver.set_NumberOfTimes(NumberOfTimes)
            
        else:
            print("Errore3")       
            sys.exit()

          
          
        try: 
            f = open(FolderResults + file)
        except: 
            sys.close()
        
        lines = f.readlines()

        i = 0
            
        while not lines[i].startswith("The value found"):
            
            i += 1
            
        ValueFound = float(lines[i].split(" ")[3])
        
        i += 1
        
        SuccessProbability = float(lines[i].split(" ")[4]) 

        i += 1
        
        AverageValue = float(lines[i].split(" ")[4])


        while not lines[i].startswith("The list of values found"):
            
            i += 1          

        i += 2
        
        ListOfValueFound = []
        
        for el in range(NumberOfTimes):
            
            ListOfValueFound.append(float(lines[i]))
            
            i +=1 
            
            
        if solversType == "TemperingPopulationv1": 
            while not lines[i].startswith("The list of best values first execution"):
                
                i += 1 

            i += 2
            
            ListOfBestValueAverage = []
            
            print(solversType)
            
            if problemSize == SizeForPlots:
                for el in range(NumberOfIteration):
                    
                    ListOfBestValueAverage.append(float(lines[i]))
                    
                    i +=1
        else:
            while not lines[i].startswith("The list of best values average"):
                
                i += 1 

            i += 2
            
            ListOfBestValueAverage = []
            
            print(solversType)
            
            if problemSize == SizeForPlots:
                for el in range(NumberOfIteration):
                    
                    ListOfBestValueAverage.append(float(lines[i]))
                    
                    i +=1

        f.close()
        
        Solver.set_SuccessProbability(SuccessProbability)
        Solver.set_AverageValue(AverageValue)
        Solver.set_OptimalValueFound(ValueFound)
        Solver.set_ValuesFound(ListOfValueFound)
        Solver.set_BestValueAverage(ListOfBestValueAverage)
        
        
        if solversType in ProblemsResults[problemSize].SolversResultsList.keys():
        
            if ProblemsResults[problemSize].SolversResultsList[solversType].get_NumberOfIteration() < NumberOfIteration or  ProblemsResults[problemSize].SolversResultsList[solversType].computeTotalNumberOfCopy() < Solver.computeTotalNumberOfCopy():
            
                ProblemsResults[problemSize].addSolverResult(solversType, Solver)

        else:
            ProblemsResults[problemSize].addSolverResult(solversType, Solver)
            
            
            
# Write Tables

print("Starts Writing Table")


fileName = "PostProcessingResults/TableSetUp.txt"

try:
    fSetUp = open(fileName, "w")
except:
    sys.exit()
    

fileNameEnergy = "PostProcessingResults/TableEnergy.txt"

try:
    fEnergy = open(fileNameEnergy, "w")
except:
    sys.exit()
    
    

fileNameProbabilities = "PostProcessingResults/TableProbabilities.txt"

try:
    fProbabilities = open(fileNameProbabilities, "w")
except:
    sys.exit()


# Table header in the setup
fSetUp.write(r'\begin{table*}' + "\n")
fSetUp.write(r'\begin{center}' + "\n")
fSetUp.write(r'\begin{tabular}{?c|c?c?c?c|c?c|c?c|c|c?c|c|c?}' + "\n")
fSetUp.write(r'\noalign{\hrule height 1.5pt}' + "\n")
fSetUp.write(r'\multicolumn{2}{?c?}{\textbf{Problem}} & \multirow{2}{*}{$N_{\textrm{tot\_copies}}$} & \textbf{SQA} & \multicolumn{2}{c?}{\textbf{SQPT}} &   \multicolumn{2}{c?}{\textbf{SQPA}} &   \multicolumn{3}{c?}{\textbf{SQPTPA1}} & \multicolumn{3}{c?}{\textbf{SQPTPA2}} \\' + "\n")
fSetUp.write(r'\cline{1-2}\cline{4-14}'+ "\n")
fSetUp.write(r'\textbf{Name} & $v_{q}$ & & $M$ & $M$ & $SYS$ &  $M$ & $SYS$  &  $M$ & $SYS_{\textrm{temp}}$ & $SYS_{\textrm{pop}}$ &  $M$ & $SYS_{\textrm{temp}}$ & $SYS_{\textrm{pop}}$ \\' + "\n")
fSetUp.write(r'\noalign{\hrule height 1.5pt}'+ "\n")



# Table header in the Energy
fEnergy.write(r'\begin{table*}' + "\n")
fEnergy.write(r'\begin{center}' + "\n")
fEnergy.write(r'\begin{tabular}{?c|c?c|c?c|c?c|c?c|c?c|c?}' + "\n")
fEnergy.write(r'\noalign{\hrule height 1.5pt}' + "\n")
fEnergy.write(r'\multicolumn{2}{?c?}{\textbf{Problem}} &  \multicolumn{2}{c?}{\textbf{SQA}} & \multicolumn{2}{c?}{\textbf{SQPT}} &   \multicolumn{2}{c?}{\textbf{SQPA}} &   \multicolumn{2}{c?}{\textbf{SQPTPA1}} & \multicolumn{2}{c?}{\textbf{SQPTPA2}} \\' + "\n")
fEnergy.write(r'\hline'+ "\n")
fEnergy.write(r'\textbf{Name} & \textbf{MC\_step} & \textbf{opt} & \textbf{avg} & \textbf{opt} & \textbf{avg} & \textbf{opt} & \textbf{avg} & \textbf{opt} & \textbf{avg} & \textbf{opt} & \textbf{avg} \\' + "\n")
fEnergy.write(r'\noalign{\hrule height 1.5pt}'+ "\n")

# Table header in the Probabilities
fProbabilities.write(r'\begin{table*}' + "\n")
fProbabilities.write(r'\begin{center}' + "\n")
fProbabilities.write(r'\begin{tabular}{?c|c?c|c?c|c?c|c?c|c?c|c?}' + "\n")
fProbabilities.write(r'\noalign{\hrule height 1.5pt}' + "\n")
fProbabilities.write(r'\multicolumn{2}{?c?}{\textbf{Problem}} &  \multicolumn{2}{c?}{\textbf{SQA}} & \multicolumn{2}{c?}{\textbf{SQPT}} &   \multicolumn{2}{c?}{\textbf{SQPA}} &   \multicolumn{2}{c?}{\textbf{SQPTPA1}} & \multicolumn{2}{c?}{\textbf{SQPTPA2}} \\' + "\n")
fProbabilities.write(r'\hline'+ "\n")
fProbabilities.write(r'\textbf{Name} & $p_{\textrm{cons}}$[\%] & $p_{\textrm{range}}$[\%] & \textbf{TTS} &  $p_{\textrm{range}}$[\%] & \textbf{TTS} & $p_{\textrm{range}}$[\%] & \textbf{TTS} &  $p_{\textrm{range}}$[\%] & \textbf{TTS} &  $p_{\textrm{range}}$[\%] & \textbf{TTS} \\' + "\n")
fProbabilities.write(r'\noalign{\hrule height 1.5pt}'+ "\n")


SolversList = ["PathIntegral", "ParallelTempering", "PopulationAnnealing", "TemperingPopulationv1", "TemperingPopulationv2"]
SolversName = ["SQA", "SQPT", "SQPA", "SQPTPA1", "SQPTPA2"]

ProblemsResults =  collections.OrderedDict(sorted(ProblemsResults.items()))

pList = {}
TTSList = {}
Sizes = []

for Solver in SolversList:
    pList[Solver] = []
    TTSList[Solver] = []
    

for Problemkey in ProblemsResults.keys(): 

    fSetupLine = ""
    fEnergyLine = ""
    fProbabilitiesLine = ""
    
    Problem = ProblemsResults[Problemkey]
    numVar = Problem.get_numberOfNodes()
    ExpectedOptimum = Problem.get_ExpectedOptimalValue()
    minReach, SolverName = Problem.min_reachedValue()
    maxReach, SolverName = Problem.max_reachedValue()
    offset = Problem.get_offset()
    
    if maxReach == minReach:
        if minReach == 0: 
            percentage = 0.001
            value = minReach+abs(minReach*percentage)
        else:
            percentage = 0.001
            value = ((minReach+offset)+abs((minReach+offset)*percentage))-offset
    else:
        value = maxReach
        percentage = 1 - minReach/maxReach
        if percentage > 0.2: 
            percentage = 0.2
            value = minReach+abs(minReach*percentage)
        if percentage < 0.0001: 
            percentage = 0.0001
            value = minReach+abs(minReach*percentage)
        
    fSetupLine = r'NumberPartitioning\_' + format(numVar) + r'&' + format(numVar) 
    
    fEnergyLine = r'NumberPartitioning\_' + format(numVar) 
    
    fProbabilitiesLine = r'NumberPartitioning\_' + format(numVar) + r' &' +  "{:.2f}".format(percentage*100)

    if len(Problem.SolversResultsList.keys()) == 5:
    
        Best_p, SolverName = Problem.best_p(value)
    
        for Solver in SolversList: 
            if Solver == "PathIntegral":

                fSetupLine += r'&' + format(Problem.SolversResultsList[Solver].computeTotalNumberOfCopy()) + r' & ' + format(Problem.SolversResultsList[Solver].get_NumberOfReplicas()) 
                fEnergyLine += r'&' + format(Problem.SolversResultsList[Solver].get_NumberOfIteration()) 
                
                #if numVar >= 100 and numVar%5 == 0:
                
                Sizes.append(numVar)
            
            elif Solver == "ParallelTempering" or Solver == "PopulationAnnealing": 
                
                fSetupLine += r'&' + format(Problem.SolversResultsList[Solver].get_NumberOfReplicas()) + r' & ' + format(Problem.SolversResultsList[Solver].get_systemCopy()) 
                
            elif Solver == "TemperingPopulationv1" or Solver == "TemperingPopulationv2": 
                
                fSetupLine += r'&' + format(Problem.SolversResultsList[Solver].get_NumberOfReplicas()) + r' & ' + format(Problem.SolversResultsList[Solver].get_systemTempering()) + "&" + format(Problem.SolversResultsList[Solver].get_systemPopulation()) 

            perc = Problem.SolversResultsList[Solver].computeProbabilityOfBeLowerThenValue(value)*100

            
            if perc == Best_p:
                
                fEnergyLine += r'&' + r'\textbf{'+ "{:.2f}".format(Problem.SolversResultsList[Solver].get_OptimalValueFound()) + r'}'+ r' & ' + r'\textbf{'+ "{:.2f}".format(Problem.SolversResultsList[Solver].get_AverageValue())+ r'}'
            
            
                fProbabilitiesLine += r'&' +r'\textbf{'+ "{:.2f}".format(Problem.SolversResultsList[Solver].computeProbabilityOfBeLowerThenValue(value)*100) + r'}'+ r' & ' + r'\textbf{'+"{:.2f}".format(Problem.SolversResultsList[Solver].returnTTS(value, 0.99, False)) + r'}'
                
            else: 
            
                fEnergyLine += r'&' + "{:.2f}".format(Problem.SolversResultsList[Solver].get_OptimalValueFound()) + r' & ' + "{:.2f}".format(Problem.SolversResultsList[Solver].get_AverageValue())
            
            
                fProbabilitiesLine += r'&' + "{:.2f}".format(Problem.SolversResultsList[Solver].computeProbabilityOfBeLowerThenValue(value)*100) + r' & ' + "{:.2f}".format(Problem.SolversResultsList[Solver].returnTTS(value, 0.99, False))
                
            #if numVar >= 100 and numVar%5 == 0:

            pList[Solver].append(Problem.SolversResultsList[Solver].computeProbabilityOfBeLowerThenValue(value)*100/Problem.SolversResultsList[Solver].get_NumberOfIteration())
            TTSList[Solver].append(Problem.SolversResultsList[Solver].returnTTS(value, 0.99, False))
        
            
        fSetupLine += r'\\' + "\n" + r' \hline' + " \n"
        fEnergyLine += r'\\' + "\n" + r'\hline' + "\n"
        fProbabilitiesLine += r'\\' + "\n" + r'\hline' + "\n"
        
        fSetUp.write(fSetupLine)
        fEnergy.write(fEnergyLine)
        fProbabilities.write(fProbabilitiesLine)
        
        if numVar == SizeForPlots:
            Iterations = range(Problem.SolversResultsList[Solver].get_NumberOfIteration())
            for i in range(len(SolversList)):
                plt.plot(Iterations, Problem.SolversResultsList[SolversList[i]].get_BestValueAverage(),  linewidth=2,label=r'\textit{'+ SolversName[i] + '}')
            plt.title(r'\textbf{Energy evolution}',fontsize=20)
            plt.xlabel(r'\textit{Iteration}', fontsize=20)
            plt.ylabel(r'\textit{Energy}', fontsize=20)
            leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
            leg.get_frame().set_facecolor('white')
            plt.savefig("PostProcessingResults/NumberPartitioningConvergence.eps", format='eps', bbox_inches='tight')
            plt.savefig("PostProcessingResults/NumberPartitioningConvergence.png", format='png', bbox_inches='tight')
            plt.savefig("PostProcessingResults/NumberPartitioningConvergence.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            
            for i in range(len(SolversList)):
                n, bins, patches = plt.hist(Problem.SolversResultsList[SolversList[i]].get_ValuesFound(), cumulative=True, histtype='step', linewidth=2,bins=100, label=r'\textit{'+ SolversName[i] + '}')
            plt.title(r'\textbf{Cumulative distribution}',fontsize=20)
            plt.xlabel(r'\textit{Values obtained}', fontsize=20)
            plt.ylabel(r'\textit{occurrence}', fontsize=20)            
            leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
            leg.get_frame().set_facecolor('white')
            plt.savefig("PostProcessingResults/NumberPartitioningCumulative.eps", format='eps', bbox_inches='tight')
            plt.savefig("PostProcessingResults/NumberPartitioningCumulative.png", format='png', bbox_inches='tight')
            plt.savefig("PostProcessingResults/NumberPartitioningCumulative.pdf", format='pdf', bbox_inches='tight')
            plt.close()


plt.yscale("log")
for i in range(len(SolversList)):
    plt.plot(Sizes, pList[SolversList[i]],  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{Percentage for iteration vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'$\frac{p_{r}}{\textrm{MC\_step}}$', fontsize=20)
leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/NumberPartitioningPercentage.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/NumberPartitioningPercentage.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/NumberPartitioningPercentage.pdf", format='pdf', bbox_inches='tight')
plt.close()

plt.yscale("log")
for i in range(len(SolversList)):
    plt.plot(Sizes, TTSList[SolversList[i]],  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{TTS vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'\textit{TTS}', fontsize=20)
leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/NumberPartitioningTTS.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/NumberPartitioningTTS.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/NumberPartitioningTTS.pdf", format='pdf', bbox_inches='tight')
plt.close()

plt.yscale("log")
for i in range(len(SolversList)):
    temp = np.array([pList[SolversList[i]][0]]* (n_point-1) + pList[SolversList[i]])
    Val = list(moving_average(temp, n_point))
    plt.plot(Sizes, Val,  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{Percentage for iteration vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'$\frac{p_{\textrm{range}}}{\textrm{MC\_step}}$', fontsize=20)
leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/NumberPartitioningPercentage_mediamobile.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/NumberPartitioningPercentage_mediamobile.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/NumberPartitioningPercentage_mediamobile.pdf", format='pdf', bbox_inches='tight')
plt.close()

plt.yscale("log")
for i in range(len(SolversList)):
    temp = np.array([TTSList[SolversList[i]][0]]* (n_point-1) + TTSList[SolversList[i]])
    Val = list(moving_average(temp, n_point))
    plt.plot(Sizes, Val,  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{TTS vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'\textit{TTS}', fontsize=20)
leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/NumberPartitioningTTS_mediamobile.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/NumberPartitioningTTS_mediamobile.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/NumberPartitioningTTS_mediamobile.pdf", format='pdf', bbox_inches='tight')
plt.close()



fSetUp.write(r'\noalign{\hrule height 1.5pt}'+ "\n")
fSetUp.write(r'\hline'+ "\n")
fSetUp.write(r'\end{tabular}' + "\n") 
fSetUp.write(r'\end{center}' + "\n")
fSetUp.write(r'\label{tab:TableResults1}' + "\n")
fSetUp.write(r'\end{table*}' + "\n")
fSetUp.close()
fEnergy.write(r'\noalign{\hrule height 1.5pt}' + "\n")
fEnergy.write(r'\hline' + "\n")
fEnergy.write(r'\end{tabular}' + "\n") 
fEnergy.write(r'\end{center}' + "\n")
fEnergy.write(r'\label{tab:TableResults1}' + "\n")
fEnergy.write(r'\end{table*}' + "\n")
fEnergy.close()
fProbabilities.write(r'\noalign{\hrule height 1.5pt}' + "\n")
fProbabilities.write(r'\hline' + "\n")
fProbabilities.write(r'\end{tabular}' + "\n") 
fProbabilities.write(r'\end{center}' + "\n")
fProbabilities.write(r'\label{tab:TableResults1}' + "\n")
fProbabilities.write(r'\end{table*}' + "\n")
fProbabilities.close()