import argparse
import pickle
import time
from dimod import BinaryQuadraticModel, make_quadratic

from replica import Replica
from parser import WorkloadParser
from cost_estimator import CostEstimator
from optim import QAOAOptimiser
from qubo import Algorithm
from anneal import (
    make_max_cost_qubo, make_total_cost_qubo, anneal,
    make_storage_constraint
)


def get_replicas(path='./replicas.csv') -> list[Replica]:
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


def get_objective_value(sample):
    """
    Decode the z slack variable from a sample to get the encoded objective value.
    z = sum_{k} 2^k * z_k, where variable names are 'z-{k}'.
    """
    keys = [key for key in sample if key.startswith('z-')]
    objective = 0
    for key in keys:
        k = int(key.split('-')[1])
        objective += (2 ** k) * int(sample[key])
    return objective


def get_slack_value(sample, replica=0):
    """
    Decode the per-replica slack variable s^(r) from a sample.
    s^(r) = sum_{k} 2^k * s_k, where variable names are 's-r{r}-{k}'.
    """
    prefix = f's-r{replica}-'
    keys = [key for key in sample if key.startswith(prefix)]
    value = 0
    for key in keys:
        k = int(key.split('-')[-1])
        value += (2 ** k) * int(sample[key])
    return value


def get_cost(sample, replica, baseline, benefits, n_queries, n_candidates):
    cost = 0
    for query in range(n_queries):
        if sample[f't-q{query}-r{replica}'] == 0:
            continue
        cost += baseline[query]
        for index in range(n_candidates):
            if sample[f'x-i{index}-r{replica}'] == 1:
                cost -= benefits[index][query]
    return cost


def create_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', '--quantum', action='store_true',
                        help='use a real quantum computer')
    parser.add_argument('-r', '--qaoa-reps', type=int, default=1,
                        help='number of repetitions to use in the QAOA ansatz')

    parser.add_argument('-A', '--penalty-term-A', type=int, default=100,
                        help='penalty term A in the QUBO')
    parser.add_argument('-C', '--penalty-term-C', type=int, default=100,
                        help='penalty term C in the QUBO')

    parser.add_argument('-w', '--storage-budget', type=int,
                        help='storage budget')
    parser.add_argument('--cost-normalisation-factor', type=float,
                        default=80000000)
    parser.add_argument('--benefit-normalisation-factor', type=float,
                        default=100000)
    parser.add_argument('-n', '--num-reads', type=int, default=100,
                        help='number of annealer reads')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help="don't actually run the annealer")
    parser.add_argument('basis', type=str, choices=['total', 'max'],
                        help='cost basis for objective function')

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
    true_costs = costs.copy()
    baseline = estimator.get_baseline()
    print('+++ cost/benefit estimation complete')

    print('+++ starting optimisation!')
    for i in range(len(benefits)):
        for j in range(len(benefits[i])):
            benefits[i][j] = benefits[i][j] // args.benefit_normalisation_factor
    baseline = [max(0, b // args.benefit_normalisation_factor) for b in baseline]
    costs = [max(0, c // args.cost_normalisation_factor) for c in costs]
    STORAGE_BUDGET = args.storage_budget // args.cost_normalisation_factor

    # Z_max: upper bound on the maximum possible replica workload cost.
    # Using sum(baseline) is conservative (all queries on one replica, no indexes).
    Z_max = sum(baseline)

    print('- baseline:', baseline)
    print('- benefits:', benefits)
    print('- costs:', costs)
    print('- storage budget:', STORAGE_BUDGET)
    print('- Z_max:', Z_max)

    # Storage budget constraints, one per replica.
    # Lambda is calibrated relative to Z_max via make_storage_constraint.
    storage_constraints = []
    if args.storage_budget:
        for r in range(len(replicas)):
            storage_constraints.append(
                make_storage_constraint(r, candidates, costs, STORAGE_BUDGET, Z_max)
            )

    print('+++ creating QUBO')
    if args.basis == 'max':
        qubo = make_max_cost_qubo(
            Z_max,
            len(replicas),
            list(range(n_templates)),
            [],
            list(range(len(candidates))),
            baseline,
            [1 for _ in candidates],
            benefits,
            1,
            storage_constraints
        )
    else:
        qubo = make_total_cost_qubo(
            len(replicas),
            list(range(n_templates)),
            [],
            list(range(len(candidates))),
            baseline,
            [1 for _ in candidates],
            benefits,
            1,
            storage_constraints
        )
    print(f'- created {args.basis} cost QUBO '
          f'({qubo.num_variables} variables, {qubo.num_interactions} interactions)')

    if args.dry_run:
        print('!!! stop due to user request')
        print('- indexes for export:', candidates)
        with open('model.pkl', 'wb') as outfile:
            pickle.dump(qubo.to_qubo(), outfile)
        return

    print('+++ starting annealing')
    tic = time.time()
    reads = anneal(qubo, 'anneal' if args.quantum else 'exact', args.num_reads)
    toc = time.time()
    result = reads.first
    print(f'+++ ! annealing complete in {round(toc - tic, 2)}s')
    print('energy', result.energy)
    print('objective (z)', get_objective_value(result.sample))
    for r in range(len(replicas)):
        print(f'slack s^({r})', get_slack_value(result.sample, r))

    indexes = []
    routes = [-1 for _ in range(n_templates)]
    pred_costs = []

    for r in range(len(replicas)):
        indexes.append([])
        space = 0
        coeff_space = 0
        pred_cost = get_cost(result.sample, r, baseline, benefits, n_templates, len(candidates))
        pred_costs.append(pred_cost)
        print(f'- Replica {r}')
        print(f'-- Predicted query cost: {pred_cost}')
        for i in range(len(candidates)):
            if result.sample[f'x-i{i}-r{r}'] == 1:
                indexes[r].append(candidates[i])
                print('\t', candidates[i])
                space += true_costs[i]
                coeff_space += costs[i]
        for q in range(n_templates):
            if result.sample[f't-q{q}-r{r}'] == 1:
                if routes[q] != -1:
                    print(f'!! warn: query {q} routed to multiple replicas. inspect output!')
                routes[q] = r
        print(f'-- Space used: {space}/{args.storage_budget} '
              f'({round(space / args.storage_budget, 4) * 100}%) '
              f'({coeff_space} / {STORAGE_BUDGET})')

    print('- Index output for benchmarking module')
    idx_string = []
    for i_r, config in enumerate(indexes):
        for index in config:
            idx_string.append(f'{i_r},{index.column}')
    print(' '.join(idx_string))

    print('- Routing table')
    print(','.join([str(r) for r in routes]))
    print('- Objective value:', max(pred_costs))

    with open('output.log', 'w') as outfile:
        outfile.write(str(result))

    # Diagnostic: print energy vs true cost for all reads to assess correlation.
    print('\n+++ read diagnostics (index, energy, z-objective, true max cost)')
    for i, read in enumerate(reads.data()):
        read_pred_costs = []
        for r in range(len(replicas)):
            read_pred_costs.append(
                get_cost(read.sample, r, baseline, benefits, n_templates, len(candidates))
            )
        print(f'{i}\tenergy {read.energy:.4f}\t'
              f'objective {get_objective_value(read.sample)}\t'
              f'cost {max(read_pred_costs)}')


if __name__ == '__main__':
    args = create_arguments()
    optimise(args)