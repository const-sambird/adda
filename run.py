import argparse
import pickle
import time
from dimod import BinaryQuadraticModel, make_quadratic, quicksum

from replica import Replica
from parser import WorkloadParser
from cost_estimator import CostEstimator
from optim import QAOAOptimiser
from qubo import Algorithm
from anneal import (
    create_slack_variables,
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
    
        print('\n+++ offset audit')
        print(f'  full_qubo offset:         {qubo.offset:.2f}')
        print(f'  objective offset:         {objective_bqm.offset:.2f}')
        for name, value in components.items():
            if not name.startswith('lam_'):
                print(f'  {name} offset: {value.offset:.2f}')
        sum_offsets = objective_bqm.offset + sum(
            v.offset for k, v in components.items() if not k.startswith('lam_')
        )
        print(f'  sum of component offsets: {sum_offsets:.2f}')
        print(f'  discrepancy in offsets:   {qubo.offset - sum_offsets:.2f}')

        print('\n+++ variable bias audit')
        # Rebuild the sum of all component BQMs
        from dimod import BinaryQuadraticModel, quicksum as qs
        all_components = [objective_bqm] + [
            v for k, v in components.items() if not k.startswith('lam_')
        ]
        summed = qs(all_components)

        # Compare coefficients
        print(f'  full_qubo num_variables:    {qubo.num_variables}')
        print(f'  summed components vars:     {summed.num_variables}')
        print(f'  full_qubo num_interactions: {qubo.num_interactions}')
        print(f'  summed interactions:        {summed.num_interactions}')

        # Find variables/interactions in full_qubo not in summed
        missing_linear = {}
        for v, bias in qubo.iter_linear():
            diff = bias - summed.get_linear(v) if v in summed.variables else bias
            if abs(diff) > 1e-6:
                missing_linear[v] = diff
        print(f'  linear coefficient mismatches: {len(missing_linear)}')

        missing_quad = {}
        for u, v, bias in qubo.iter_quadratic():
            try:
                diff = bias - summed.get_quadratic(u, v)
            except Exception:
                diff = bias
            if abs(diff) > 1e-6:
                missing_quad[(u,v)] = diff
        print(f'  quadratic coefficient mismatches: {len(missing_quad)}')
        if missing_quad:
            # Show a sample of the missing interactions
            items = list(missing_quad.items())[:5]
            for (u,v), d in items:
                print(f'    ({u}, {v}): diff={d:.4g}')
    replica_load_combined = components.get('replica_load')
    aux_vars = [v for v in replica_load_combined.variables 
                if '*' in str(v) or 'aux' in str(v).lower()]
    print(f'auxiliary variables in replica_load: {len(aux_vars)}')
    aux_energy = sum(
        replica_load_combined.get_linear(v) * sample.sample[v]
        for v in aux_vars
    ) + sum(
        replica_load_combined.get_quadratic(u, v) * sample.sample[u] * sample.sample[v]
        for u, v in replica_load_combined.quadratic
        if '*' in str(u)
    )
    print(f'auxiliary variable energy contribution: {aux_energy:.2f}')
    original_vars = [v for v in replica_load_combined.variables 
                 if '*' not in str(v) and 'aux' not in str(v).lower()]

    linear_energy = sum(
        replica_load_combined.get_linear(v) * sample.sample[v]
        for v in original_vars
    )
    quadratic_energy = sum(
        replica_load_combined.get_quadratic(u, v) * sample.sample[u] * sample.sample[v]
        for u, v in replica_load_combined.quadratic
        if '*' not in str(u) and '*' not in str(v)
        and 'aux' not in str(u).lower() and 'aux' not in str(v).lower()
    )
    print(f'original variable linear energy:     {linear_energy:.2f}')
    print(f'original variable quadratic energy:  {quadratic_energy:.2f}')
    print(f'auxiliary variable energy:           {aux_energy:.2f}')
    print(f'offset:                              {replica_load_combined.offset:.2f}')
    print(f'sum:                                 {linear_energy + quadratic_energy + aux_energy + replica_load_combined.offset:.2f}')

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

    print('+++ creating QUBO')
    # Storage constraints are built AFTER the main QUBO so that lam_storage
    # can be derived from replica_load_combined (available in components),
    # keeping lam_storage in the same tier as lam_routing.
    objective_bqm = create_slack_variables('z', max(1, Z_max))  # kept for decomposition
    if args.basis == 'max':
        qubo, components = make_max_cost_qubo(
            Z_max,
            len(replicas),
            list(range(n_templates)),
            [],
            list(range(len(candidates))),
            baseline,
            [1 for _ in candidates],
            benefits,
            1,
        )
    else:
        objective_bqm = None   # total-cost objective has no z slack; use None
        qubo, components = make_total_cost_qubo(
            len(replicas),
            list(range(n_templates)),
            [],
            list(range(len(candidates))),
            baseline,
            [1 for _ in candidates],
            benefits,
            1,
        )

    # Now build storage constraints calibrated from the assembled replica_load BQM.
    if args.storage_budget:
        calibration_bqm = objective_bqm
        storage_bqms = []
        for r in range(len(replicas)):
            sc = make_storage_constraint(
                r, candidates, costs, STORAGE_BUDGET, calibration_bqm
            )
            storage_bqms.append(sc)
            qubo.update(sc)
        components['storage'] = quicksum(storage_bqms)
        from anneal import penalty_lambda_from_objective
        components['lam_storage'] = penalty_lambda_from_objective(calibration_bqm, multiplier=2.0)
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
    reads = anneal(qubo, 'anneal' if args.quantum else 'exact', args.num_reads)
    toc = time.time()

    best_cost = float('inf')
    result = None
    for i, read in enumerate(reads.data()):
        read_pred_costs = []
        for r in range(len(replicas)):
            read_pred_costs.append(
                get_cost(read.sample, r, baseline, benefits, n_templates, len(candidates))
            )
        if args.basis == 'max':
            this_cost = max(read_pred_costs)
        else:
            this_cost = sum(read_pred_costs)
        
        if this_cost < best_cost:
            best_cost = this_cost
            result = read

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

    # Energy decomposition: shows relative scale of objective vs each penalty term
    if objective_bqm is not None:
        print(f'  manual energy check: {qubo.energy(result.sample):.2f}')
        Q, qubo_offset = qubo.to_qubo()
        from dimod import BinaryQuadraticModel
        qubo_from_matrix = BinaryQuadraticModel.from_qubo(Q)
        print(f'  qubo_offset: {qubo_offset:.2f}')
        print(f'  energy via from_qubo (no offset): {qubo_from_matrix.energy(result.sample):.2f}')
        print(f'  energy via from_qubo + offset:    {qubo_from_matrix.energy(result.sample) + qubo_offset:.2f}')
        decompose_energy(result, qubo, objective_bqm, components, qubo.offset)
    else:
        # For total-cost basis, reconstruct a minimal objective proxy
        from dimod import BinaryQuadraticModel as _ObjBQM
        decompose_energy(result, qubo, _ObjBQM('BINARY'), components)

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