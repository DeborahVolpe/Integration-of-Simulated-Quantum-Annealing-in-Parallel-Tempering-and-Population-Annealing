import os
import sys
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import cm
import shutil
import numpy as np
from SolversClass import SolversResults
from problemResultClass import ProblemResultMaxCut
import collections
from scipy.interpolate import interp1d


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

rc('text', usetex=True)
plt.rc('text', usetex=True)

FolderProblem = "MaxCutProblems/"

FolderResults = "MaxCutProblemsResults/"

print("Start")


SizeForPlots = 200

n_point = 50

shutil.rmtree("PostProcessingResults", ignore_errors = True)
path = '.'
os.mkdir("PostProcessingResults")

FirstIterationPathIntergral = []

    

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
            Energy = float(lines[0].split(" ")[1])
        except:
            sys.close()
           
        f.close() 
        
        Problem = ProblemResultMaxCut(problemSize)
        
        Problem.set_ExpectedOptimalValue(Energy)
        
        
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
            
        while not lines[i].startswith("The list of best values first execution"):
                        
            i += 1 

        i += 2
        
        if problemSize == SizeForPlots and solversType == "PathIntegral":
            for el in range(NumberOfIteration):
                
                FirstIterationPathIntergral.append(float(lines[i]))
                
                i +=1            
            
        while not lines[i].startswith("The list of best values average"):
            
            i += 1 

        i += 2
        
        ListOfBestValueAverage = []
        
        print(solversType)
        
        if problemSize == SizeForPlots:
            if solversType != "TemperingPopulationv1":
                for el in range(NumberOfIteration):
                    
                    ListOfBestValueAverage.append(float(lines[i]))
                    
                    i +=1
            else: 
            
                TempList = lines[i].split("-")[1:]
                #print(TempList)
                for el in TempList:
                   ListOfBestValueAverage.append(float("-"+ el)) 
                  
                

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
    
    if maxReach == minReach:
        percentage = 0.001
        value = minReach+abs(minReach*percentage)
    else:
        value = maxReach
        percentage = abs(1 - maxReach/minReach)
        if percentage < 0.0001: 
            percentage = 0.0001
            value = minReach+abs(minReach*percentage)
        
    fSetupLine = r'MaxCut\_' + format(numVar) + r'&' + format(numVar) 
    
    fEnergyLine = r'MaxCut\_' + format(numVar) 
    
    fProbabilitiesLine = r'MaxCut\_' + format(numVar) + r' &' +  "{:.2f}".format(percentage*100)

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
            plt.savefig("PostProcessingResults/Convergence.eps", format='eps', bbox_inches='tight')
            plt.savefig("PostProcessingResults/Convergence.png", format='png', bbox_inches='tight')
            plt.savefig("PostProcessingResults/Convergence.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            
            plt.plot(Iterations, Problem.SolversResultsList[SolversList[0]].get_BestValueAverage(), color="b", linewidth=2,label=r'\textit{Average}')
            plt.plot(Iterations, FirstIterationPathIntergral,  color="g", linewidth=2,label=r'\textit{Single run}')
            plt.title(r'\textbf{Energy evolution}',fontsize=20)
            plt.xlabel(r'\textit{Iteration}', fontsize=20)
            plt.ylabel(r'\textit{Energy}', fontsize=20)
            leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
            leg.get_frame().set_facecolor('white')
            plt.savefig("PostProcessingResults/Explain.eps", format='eps', bbox_inches='tight')
            plt.savefig("PostProcessingResults/Explain.png", format='png', bbox_inches='tight')
            plt.savefig("PostProcessingResults/Explain.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            
            for i in range(len(SolversList)):
                n, bins, patches = plt.hist(Problem.SolversResultsList[SolversList[i]].get_ValuesFound(), cumulative=True, histtype='step', linewidth=2,bins=100, label=r'\textit{'+ SolversName[i] + '}')
            plt.title(r'\textbf{Cumulative distribution}',fontsize=20)
            plt.xlabel(r'\textit{Values obtained}', fontsize=20)
            plt.ylabel(r'\textit{occurrence}', fontsize=20)            
            leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
            leg.get_frame().set_facecolor('white')
            plt.savefig("PostProcessingResults/Cumulative.eps", format='eps', bbox_inches='tight')
            plt.savefig("PostProcessingResults/Cumulative.png", format='png', bbox_inches='tight')
            plt.savefig("PostProcessingResults/Cumulative.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            
            n, bins, patches = plt.hist(Problem.SolversResultsList[SolversList[0]].get_ValuesFound(), cumulative=True, histtype='step', linewidth=2,bins=100)            
            plt.title(r'\textbf{Cumulative distribution}',fontsize=20)
            plt.xlabel(r'\textit{Values obtained}', fontsize=20)
            plt.ylabel(r'\textit{occurrence}', fontsize=20)            
            plt.savefig("PostProcessingResults/CumulativeExplain.eps", format='eps', bbox_inches='tight')
            plt.savefig("PostProcessingResults/CumulativeExplain.png", format='png', bbox_inches='tight')
            plt.savefig("PostProcessingResults/CumulativeExplain.pdf", format='pdf', bbox_inches='tight')
            plt.close()

plt.yscale("log")
for i in range(len(SolversList)):
    plt.plot(Sizes, pList[SolversList[i]],  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{Percentage for iteration vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'$\frac{p_{r}}{\textrm{MC\_step}}$', fontsize=20)
leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/MaxCutPercentage.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutPercentage.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutPercentage.pdf", format='pdf', bbox_inches='tight')
plt.close()


plt.yscale("log")
for i in range(len(SolversList)):
    plt.plot(Sizes, TTSList[SolversList[i]],  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{TTS vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'\textit{TTS}', fontsize=20)
leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/MaxCutTTS.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutTTS.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutTTS.pdf", format='pdf', bbox_inches='tight')
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
plt.savefig("PostProcessingResults/MaxCutPercentage_mediamobile.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutPercentage_mediamobile.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutPercentage_mediamobile.pdf", format='pdf', bbox_inches='tight')
plt.close()

i=0
plt.yscale("log")
temp = np.array([pList[SolversList[i]][0]]* (n_point-1) + pList[SolversList[i]])
Val = list(moving_average(temp, n_point))
plt.plot(Sizes, pList[SolversList[i]],  linewidth=1, color = "b", label=r'\textit{effective samples}')
plt.plot(Sizes, Val,  linewidth=1, color= "c", label=r'\textit{moving average}')
plt.title(r'\textbf{Percentage for iteration vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'$\frac{p_{\textrm{range}}}{\textrm{MC\_step}}$', fontsize=20)
leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/ExplainationPercentage_mediamobile.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/ExplainationPercentage_mediamobile.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/ExplainationPercentage_mediamobile.pdf", format='pdf', bbox_inches='tight')
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
plt.savefig("PostProcessingResults/MaxCutTTS_mediamobile.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutTTS_mediamobile.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutTTS_mediamobile.pdf", format='pdf', bbox_inches='tight')
plt.close()

i=0
plt.yscale("log")
temp = np.array([TTSList[SolversList[i]][0]]* (n_point-1) + TTSList[SolversList[i]])
Val = list(moving_average(temp, n_point))
plt.plot(Sizes, TTSList[SolversList[i]],  linewidth=1, color = "b", label=r'\textit{effective samples}')
plt.plot(Sizes, Val,  linewidth=1, color= "c", label=r'\textit{moving average}')
plt.title(r'\textbf{TTS vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'\textit{TTS}', fontsize=20)
leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/ExplainationTTS_mediamobile.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/ExplainationTTS_mediamobile.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/ExplainationTTS_mediamobile.pdf", format='pdf', bbox_inches='tight')
plt.close()


fileNameInterpolation= "PostProcessingResults/InterpolationValues.txt"

try:
    fInterp = open(fileNameInterpolation, "w")
except:
    sys.exit()

fInterp.write("Linear\n\n")
fInterp.write("percentage \n\n")
plt.yscale("log")
SizesN = list(np.linspace(1, 200, 199))
for i in range(len(SolversList)):
    f = interp1d(Sizes[30:], pList[SolversList[i]][30:], fill_value="extrapolate")
    fInterp.write(SolversName[i] + "\n")
    fInterp.write(format(f) + "\n")
    plt.plot(SizesN, f(SizesN),  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{Percentage for iteration vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'$\frac{p_{r}}{\textrm{MC\_step}}$', fontsize=20)
leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/MaxCutPercentage_linearinterpolation.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutPercentage_linearinterpolation.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutPercentage_linearinterpolation.pdf", format='pdf', bbox_inches='tight')
plt.close()

fInterp.write("TTS \n\n")
plt.yscale("log")
for i in range(len(SolversList)):
    f = interp1d(Sizes[30:],TTSList[SolversList[i]][30:], fill_value="extrapolate")
    fInterp.write(SolversName[i] + "\n")
    fInterp.write(format(f) + "\n")
    plt.plot(SizesN, f(SizesN),  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{TTS vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'\textit{TTS}', fontsize=20)
leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/MaxCutTTS_linearinterpolation.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutTTS_linearinterpolation.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutTTS_linearinterpolation.pdf", format='pdf', bbox_inches='tight')
plt.close()

fInterp.write("Cubic\n\n")
fInterp.write("percentage \n\n")
plt.yscale("log")
for i in range(len(SolversList)):
    f = interp1d(Sizes, pList[SolversList[i]], kind='cubic', fill_value="extrapolate" )
    fInterp.write(SolversName[i] + "\n")
    fInterp.write(format(f) + "\n")
    plt.plot(SizesN, f(SizesN),  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{Percentage for iteration vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'$\frac{p_{r}}{\textrm{MC\_step}}$', fontsize=20)
leg = plt.legend(loc='upper right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/MaxCutPercentage_cubicinterpolations.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutPercentage_cubicinterpolations.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutPercentage_cubicinterpolations.pdf", format='pdf', bbox_inches='tight')
plt.close()

fInterp.write("TTS \n\n")
plt.yscale("log")
for i in range(len(SolversList)):
    f = interp1d(Sizes,TTSList[SolversList[i]], kind='cubic', fill_value="extrapolate")
    fInterp.write(SolversName[i] + "\n")
    fInterp.write(format(f) + "\n")
    plt.plot(SizesN, f(SizesN),  linewidth=1,label=r'\textit{'+ SolversName[i] + '}')
plt.title(r'\textbf{TTS vs Problem Size}',fontsize=20)
plt.xlabel(r'\textit{number of nodes}', fontsize=20)
plt.ylabel(r'\textit{TTS}', fontsize=20)
leg = plt.legend(loc='lower right', frameon=True, fontsize=15)
leg.get_frame().set_facecolor('white')
plt.savefig("PostProcessingResults/MaxCutTTS_cubicinterpolations.eps", format='eps', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutTTS_cubicinterpolations.png", format='png', bbox_inches='tight')
plt.savefig("PostProcessingResults/MaxCutTTS_cubicinterpolations.pdf", format='pdf', bbox_inches='tight')
plt.close()

fInterp.close()


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