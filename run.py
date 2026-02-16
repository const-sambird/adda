import argparse
import pickle
import time
from dimod import BinaryQuadraticModel, make_quadratic

from replica import Replica
from parser import WorkloadParser
from cost_estimator import CostEstimator
from optim import QAOAOptimiser
from qubo import Algorithm
from anneal import make_qubo, anneal, apply_penalty_lagrangian, create_slack_varibles
from util import square_bqm_to_binary_polynomial

def get_replicas(path = './replicas.csv') -> list[Replica]:
    replicas = []
    with open(path, 'r') as infile:
        lines = infile.readlines()
        for config in lines:
            fields = config.split(',')
            replicas.append(
                Replica(
                    id=fields[0],
                    hostname=fields[1],
                    port=fields[2],
                    dbname=fields[3],
                    user=fields[4],
                    password=fields[5]
                )
            )
    return replicas

def create_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', '--quantum', action='store_true', help='use a real quantum computer')
    parser.add_argument('-r', '--qaoa-reps', type=int, default=1, help='number of repetitions to use in the QAOA ansatz')

    parser.add_argument('-A', '--penalty-term-A', type=int, default=100, help='penalty term A in the QUBO')
    parser.add_argument('-C', '--penalty-term-C', type=int, default=100, help='penalty term C in the QUBO')

    parser.add_argument('-w', '--storage-budget', type=int, help='storage budget')
    parser.add_argument('--cost-normalisation-factor', type=float, default=80000000)
    parser.add_argument('--benefit-normalisation-factor', type=float, default=100000)
    parser.add_argument('-n', '--num-reads', type=int, default=100, help='number of annealer reads')
    parser.add_argument('-d', '--dry-run', action='store_true', help='don\'t actually run the annealer')
    #parser.add_argument('problem')

    return parser.parse_args()

def optimise(args):
    replicas = get_replicas()
    parser = WorkloadParser(replicas[0])
    parser.read_queries('./workload')
    parser.get_all_columns()
    parser.extract_candidates()

    workload = parser.get_workload()
    templates = parser.get_templates()
    candidates = parser.get_candidates()
    n_templates = max(templates) + 1

    print('+++ workload parsing complete')
    print(len(workload), 'queries')
    print(n_templates, 'templates')
    print(len(candidates), 'candidates')
    print(len(replicas), 'replicas')
    print()
    print('index candidates:')
    for candidate in candidates:
        print('\t', candidate)

    print('+++ starting cost/benefit estimation')
    estimator = CostEstimator(replicas, candidates, workload, templates, n_templates)
    benefits = estimator.get_benefits()
    costs = estimator.get_storage_costs()
    baseline = estimator.get_baseline()
    print('+++ cost/benefit estimation complete')

    print('+++ starting optimisation!')
    for i in range(len(benefits)):
        for j in range(len(benefits[i])):
            benefits[i][j] = benefits[i][j] // args.benefit_normalisation_factor
    costs = [c // args.cost_normalisation_factor for c in costs]
    STORAGE_BUDGET = args.storage_budget // args.cost_normalisation_factor

    print('- benefits')
    print(benefits)
    print('- costs')
    print(costs)
    print('- storage budget:', STORAGE_BUDGET)

    # -- slack variable constraint for qubo
    # W_max - \sum_i w_i x_i - s
    storage_constraints = []
    for r in range(len(replicas)):
        constraint_model = BinaryQuadraticModel(vartype='BINARY')
        for i in range(len(candidates)):
            constraint_model.add_linear(f'x-i{i}-r{r}', -costs[i])
        for var, bias in create_slack_varibles(f's-wmax-r{r}', STORAGE_BUDGET).iter_linear():
            constraint_model.add_linear(var, -bias)
        constraint_model.offset = STORAGE_BUDGET
        hubo = square_bqm_to_binary_polynomial(constraint_model)
        qubo = make_quadratic(hubo, 10000, 'BINARY')
        storage_constraints.append(qubo)
    
    print('+++ creating QUBO')
    qubo = make_qubo(500, len(replicas), [i for i in range(n_templates)], [], [i for i in range(len(candidates))], baseline, [1 for _ in candidates], benefits, 1, storage_constraints)
    
    if args.dry_run:
        print('!!! stop due to user request')
        return
    
    print('+++ starting annealing')
    tic = time.time()
    result = anneal(qubo, 'anneal' if args.quantum else 'exact', args.num_reads).first
    toc = time.time()
    print(f'+++ ! annealing complete in {round(toc - tic, 2)}s')
    print('energy', result.energy)
    indexes = []
    routes = [-1 for _ in range(n_templates)]

    for r in range(len(replicas)):
        indexes.append([])
        print('- Replica', r)
        for i in range(len(candidates)):
            if result.sample[f'x-i{i}-r{r}'] == 1:
                indexes[r].append(candidates[i])
                print('\t', candidates[i])
        for q in range(n_templates):
            if result.sample[f't-q{q}-r{r}'] == 1:
                if routes[q] != -1:
                    print('!! warn: query', q, 'routed to multiple replicas. inspect output!')
                routes[q] = r

    print('- Index output for benchmarking module')
    idx_string = []
    for i_r, config in enumerate(indexes):
        for index in config:
            idx_string.append(f'{i_r},{index.column}')
    print(' '.join(idx_string))
    
    print('- Routing table')
    print(routes)

    with open('output.log', 'w') as outfile:
        outfile.write(str(result))

if __name__ == '__main__':
    args = create_arguments()
    optimise(args)
