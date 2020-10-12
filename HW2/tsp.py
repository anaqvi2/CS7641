import time
from array import array
from itertools import product
from time import clock

import sys

sys.path.append("./ABAGAIL.jar")
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction

N = 50
random = Random()
maxIters = 5000
numTrials = 5
samples = 100
keep = 50
m = 0.1
CE = 0.95
population = 200
mate = 100
mutate = 10
iteration_marks = [10, 100, 250, 500, 1000, 2500, 5000]
fname = "TSP_ALL.csv"
with open(fname,'w') as f:
    f.write('algorithm,numtrial,iterations,fitness,time\n')

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
ef = TravelingSalesmanRouteEvaluationFunction(points)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
ranges = array('i', [N] * N)
odd = DiscreteUniformDistribution(ranges)
df = DiscreteDependencyTree(m, ranges)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

def get_values(fit, mod, algo):
    for i in iteration_marks:
        start = clock()
        fit.train()
        times.append(times[-1] + (time.clock() - start))
        score = ef.value(mod.getOptimal())
        with open(fname,'a') as f:
            f.write('{},{},{},{},{}\n'.format(algo, t, i, score, times[-1]))

for t in range(numTrials):
    mimic = MIMIC(samples, keep, pop)
    fit = FixedIterationTrainer(mimic, 10)
    times = [0]
    get_values(fit, mimic, "Mimic")

for t in range(numTrials):
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 10)
    times = [0]
    get_values(fit, rhc, "RHC")

for t in range(numTrials):
    sa = SimulatedAnnealing(1E10, CE, hcp)
    fit = FixedIterationTrainer(sa, 10)
    times = [0]
    get_values(fit, sa, "SA")

for t in range(numTrials):
    ga = StandardGeneticAlgorithm(population, mate, mutate, gap)
    fit = FixedIterationTrainer(ga, 10)
    times = [0]
    get_values(fit, ga, "GA")
