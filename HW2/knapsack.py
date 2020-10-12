import time
import sys
import java.util.Random as Random

sys.path.append("./ABAGAIL.jar")

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array
from time import clock
from itertools import product

random = Random()
N = 40
KNAPSACK_VOLUME = 50 * N * 4 * .4
fill = [4] * N
copies = array('i', fill)
fill = [0] * N
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, N):
    weights[i] = random.nextDouble() * 50
    volumes[i] = random.nextDouble() * 50
fill = [4 + 1] * N
ranges = array('i', fill)
numTrials = 5
samples = 100
keep = 50
m = 0.1
CE = 0.95
population = 200
mate = 100
mutate = 10
iteration_marks = [10, 100, 250, 500, 1000, 2500, 5000]
fname = "KP_ALL.csv"
with open(fname,'w') as f:
    f.write('algorithm,numtrial,iterations,fitness,time\n')


ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

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
    get_values(fit, mimic, "MIMIC")

for t in range(numTrials):
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 10)
    times = [0]
    get_values(fit, rhc, "RHC")

for t in range(numTrials):
    sa = SimulatedAnnealing(1E10, CE, hcp)
    fit = FixedIterationTrainer(sa, 10)
    get_values(fit, sa, "SA")

for t in range(numTrials):
    ga = StandardGeneticAlgorithm(population, mate, mutate, gap)
    fit = FixedIterationTrainer(ga, 10)
    times = [0]
    get_values(fit, ga, "GA")