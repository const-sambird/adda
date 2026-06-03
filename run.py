import argparse
import pickle
import time
import math
from dimod import BinaryQuadraticModel, make_quadratic, quicksum

from replica import Replica
from parser import WorkloadParser
from cost_estimator import CostEstimator
from anneal import (
    create_slack_variables,
    make_max_cost_qubo, make_total_cost_qubo, anneal,
    make_storage_constraint, omega
)
from problem import PROBLEMS
from index_candidate import DummyIndexCandidate


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


def get_cost(sample, replica, baseline, benefits, n_queries, n_candidates, queries, failed=-1):
    cost = 0
    def t(q, r):
        if failed == -1:
            return f't-q{q}-r{r}'
        return f't-q{q}-r{r}-j{failed}'
    for query in range(n_queries):
        if query in queries and sample[t(query, replica)] == 0:
            continue
        cost += baseline[query]
        for index in range(n_candidates):
            if sample[f'x-i{index}-r{replica}'] == 1:
                cost -= benefits[index][query]
    return cost

def is_feasible(sample, n_queries):
    for q in range(n_queries):
        ts = [v for k, v in sample.items() if f't-q{q}-' in k]
        if sum(ts) == 0:
            return False
    return True

def decompose_energy(sample, qubo, objective_bqm, components: dict, full_qubo_offset: float = 0.0):
    print('\n+++ energy decomposition')
    total = 0  # start with the full QUBO's offset (usually 0)

    obj_energy = objective_bqm.energy(sample.sample)# - objective_bqm.offset
    print(f'  {"objective":<22} {obj_energy:>20.2f}')
    total += obj_energy

    for name, value in components.items():
        if name.startswith('lam_'):
            print(f'  {"lambda " + name[4:]:<22} {value:>20.6g}')
            continue
        e = value.energy(sample.sample)# - value.offset
        print(f'  {name:<22} {e:>20.2f}')
        total += e

    print(f'  {"─" * 44}')
    print(f'  {"total (recomputed)":<22} {total:>20.2f}')
    print(f'  {"reported energy":<22} {sample.energy:>20.2f}')
    discrepancy = abs(total - sample.energy)
    if discrepancy > 1.0:
        print(f'  !! discrepancy of {discrepancy:.2f} — check for missing components')
    
    replica_load_combined = components.get('replica_load')
    aux_vars = [v for v in replica_load_combined.variables 
                if '*' in str(v) or 'aux' in str(v).lower()]
    print(f'auxiliary variables in replica_load: {len(aux_vars)}')

def create_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', '--quantum', action='store_true',
                        help='use a real quantum computer')
    parser.add_argument('-a', '--qaoa', action='store_true',
                        help='use the quantum approximate optimisation algorithm instead of annealing')

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
    parser.add_argument('-p', '--problem', choices=PROBLEMS.keys(), type=str)
    parser.add_argument('--workload-path', type=str, default='./workload')
    parser.add_argument('--alpha', type=float, default=0.0, help='per-node failure probability')
    parser.add_argument('basis', type=str, choices=['total', 'max'],
                        help='cost basis for objective function')

    return parser.parse_args()


def optimise(args):
    replicas = get_replicas()
    parser = WorkloadParser(replicas[0])
    parser.read_queries(args.workload_path)
    parser.get_all_columns()
    parser.extract_candidates()

    if args.problem:
        problem = PROBLEMS[args.problem]
        benefits = problem.benefits
        costs = problem.weights
        true_costs = costs.copy()
        baseline = problem.baseline
        STORAGE_BUDGET = problem.budget
        queries = [i for i in range(len(baseline))]
        updates = []
        candidates = [DummyIndexCandidate(i) for i in range(len(costs))]
        n_templates = len(baseline)
        n_replicas = len(replicas)
        print('+++ loaded problem', problem.name)
        print(n_templates, 'queries')
        print(n_templates, 'templates')
        print(len(candidates), 'candidates')
        print(len(replicas), 'replicas')
        print()
        print('index candidates:')
        for candidate in candidates:
            print('\t', candidate)
    else:
        workload = parser.get_workload()
        templates = parser.get_templates()
        queries = parser.get_queries()
        updates = parser.get_updates()
        candidates = parser.get_candidates()
        n_templates = parser.get_num_templates()
        n_replicas = len(replicas)

        print('+++ workload parsing complete')
        print(len(workload), 'statements')
        print(queries)
        print(updates)
        print(n_templates, 'templates', f'({len(queries)} queries, {len(updates)} updates)')
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
    Z_max = omega(queries, updates, [i for i in range(len(candidates))], baseline, [1 for _ in range(n_templates)], n_replicas)

    print('- baseline:', baseline)
    print('- benefits:', benefits)
    print('- costs:', costs)
    print('- storage budget:', STORAGE_BUDGET)
    print('- Z_max:', Z_max)

    print('+++ creating QUBO')
    # Storage constraints are built AFTER the main QUBO so that lam_storage
    # can be derived from replica_load_combined (available in components),
    # keeping lam_storage in the same tier as lam_routing.
    objective_bqm = create_slack_variables('z', max(1, Z_max))  # kept for decomposition
    if args.basis == 'max':
        qubo, components = make_max_cost_qubo(
            Z_max,
            len(replicas),
            queries,
            updates,
            list(range(len(candidates))),
            baseline,
            [1 for _ in range(n_templates)],
            benefits,
            1,
            args.alpha
        )
    else:
        qubo, components = make_total_cost_qubo(
            len(replicas),
            queries,
            updates,
            list(range(len(candidates))),
            baseline,
            [1 for _ in range(n_templates)],
            benefits,
            1,
        )
        objective_bqm = components['objective']

    # Now build storage constraints calibrated from the assembled replica_load BQM.
    if args.storage_budget or args.problem:
        calibration_bqm = objective_bqm
        storage_bqms = []
        for r in range(len(replicas)):
            sc, lam = make_storage_constraint(
                r, candidates, costs, STORAGE_BUDGET, calibration_bqm, list(range(n_templates)), [], n_replicas, baseline
            )
            storage_bqms.append(sc)
            qubo.update(sc)
        components['storage'] = quicksum(storage_bqms)
        components['lam_storage'] = lam
    print(f'- created {args.basis} cost QUBO '
          f'({qubo.num_variables} variables, {qubo.num_interactions} interactions)')
    print('+++ lambda values used')
    for k, v in components.items():
        if k.startswith('lam_'):
            print(f'  {k}: {v:.6g}')
    
    #qubo.offset = 0.0

    if args.dry_run:
        print('!!! stop due to user request')
        print('- indexes for export:', candidates)
        with open('model.pkl', 'wb') as outfile:
            pickle.dump(qubo.to_qubo(), outfile)
        return

    print('+++ starting annealing')
    tic = time.time()
    reads = anneal(qubo, 'qaoa' if args.qaoa else 'anneal', 'quantum' if args.quantum else 'simulate', args.num_reads)
    toc = time.time()

    best_cost = float('inf')
    result = None
    for i, read in enumerate(reads.data()):
        read_pred_costs = []
        for r in range(len(replicas)):
            read_pred_costs.append(
                get_cost(read.sample, r, baseline, benefits, n_templates, len(candidates), queries)
            )
        if args.basis == 'max':
            this_cost = max(read_pred_costs)
        else:
            this_cost = sum(read_pred_costs)
        
        if this_cost < best_cost:
            best_cost = this_cost
            result = read
    
    if args.qaoa:
        result = reads.lowest().first

    print(f'+++ ! annealing complete in {round(toc - tic, 2)}s')
    print('energy', result.energy)
    print('objective (z)', get_objective_value(result.sample))
    if args.basis == 'max':
        for r in range(len(replicas)):
            print(f'slack s^({r})', get_slack_value(result.sample, r))
        
        for r in range(len(replicas)):
            z_val = sum(2**k * int(result.sample[f'z-{k}']) 
                        for k in range(math.floor(math.log2(Z_max)) + 1))
            load_val = sum(
                result.sample[f't-q{q}-r{r}'] * (
                    1 * baseline[q] / 1 - 
                    sum(1 * benefits[i][q] / 1 * int(result.sample[f'x-i{i}-r{r}']) 
                        for i in range(len(candidates)))
                )
                for q in queries
            )
            slack_val = sum(2**k * int(result.sample[f's-r{r}-{k}']) 
                        for k in range(math.floor(math.log2(Z_max)) + 1))
            residual = z_val - load_val - slack_val
            print(f'replica {r}: z={z_val:.3f}, load={load_val:.3f}, '
                f'slack={slack_val:.3f}, residual={residual:.3f}')

    indexes, routes, pred_costs = extract_configuration(result, replicas, queries, updates, baseline, benefits, candidates, costs, true_costs, n_templates, STORAGE_BUDGET)

    if args.alpha > 0:
        with open('./fail-out.log', 'w') as outfile:
            for r in range(len(replicas)):
                outfile.write(f'replica-{r}-failed\n')
                f_indexes, f_routes, f_pred_costs = extract_configuration(result, replicas, queries, updates, baseline, benefits, candidates, costs, true_costs, n_templates, STORAGE_BUDGET, r)
                idx_string = []
                for i_r, config in enumerate(f_indexes):
                    for index in config:
                        idx_string.append(f'{i_r},{index.column}')
                outfile.write(' '.join(idx_string))
                outfile.write('\n')
                outfile.write(','.join([str(r) for r in f_routes]))
                outfile.write('\n')
                basis_fn = max if args.basis == 'max' else sum
                outfile.write('objective,')
                outfile.write(str(basis_fn(f_pred_costs)))
                outfile.write('\n')

    # Energy decomposition: shows relative scale of objective vs each penalty term
    if args.basis == 'max':
        decompose_energy(result, qubo, objective_bqm, components, qubo.offset)

    print('- Index output for benchmarking module')
    idx_string = []
    for i_r, config in enumerate(indexes):
        for index in config:
            idx_string.append(f'{i_r},{index.column}')
    print(' '.join(idx_string))

    print('- Routing table')
    print(','.join([str(r) for r in routes]))
    basis_fn = max if args.basis == 'max' else sum
    print('- Objective value:', basis_fn(pred_costs))

    with open('output.log', 'w') as outfile:
        outfile.write(str(result)) 
    exit(0)
    # Diagnostic: print energy vs true cost for all reads to assess correlation.
    print('\n+++ read diagnostics (index, energy, z-objective, true max cost)')
    for i, read in enumerate(reads.data()):
        read_pred_costs = []
        for r in range(len(replicas)):
            read_pred_costs.append(
                get_cost(read.sample, r, baseline, benefits, n_templates, len(candidates), queries)
            )
        print(f'{i}\tenergy {read.energy:.4f}\t'
              f'objective {get_objective_value(read.sample)}\t'
              f'cost {basis_fn(read_pred_costs)}')

def extract_configuration(result, replicas, queries, updates, baseline, benefits, candidates, costs, true_costs, n_templates, STORAGE_BUDGET, failed=-1):
    indexes = []
    routes = [-1 for _ in range(len(queries))]
    pred_costs = []

    def t(q, r):
        if failed == -1:
            return f't-q{q}-r{r}'
        return f't-q{q}-r{r}-j{failed}'

    if failed == -1:
        print('================= BASELINE CASE =================')
    else:
        print(f'================= NODE {failed} FAILURE ==================')

    for r in range(len(replicas)):
        indexes.append([])
        if r == failed: continue
        space = 0
        coeff_space = 0
        pred_cost = get_cost(result.sample, r, baseline, benefits, len(queries), len(candidates), queries, failed)
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
            if q in updates:
                routes[q] = -1
                continue
            if result.sample[t(q, r)] == 1:
                if routes[q] != -1:
                    print(f'!! warn: query {q} routed to multiple replicas. inspect output!')
                routes[q] = r
        if not args.problem:
            print(f'-- Space used: {space}/{args.storage_budget} '
                f'({round(space / args.storage_budget, 4) * 100}%) '
                f'({coeff_space} / {STORAGE_BUDGET})')
        else:
            print(f'-- Space used: {coeff_space} / {STORAGE_BUDGET} '
                  f'({round(coeff_space / STORAGE_BUDGET, 4) * 100}%)')
    
    return indexes, routes, pred_costs

if __name__ == '__main__':
    args = create_arguments()
    optimise(args)