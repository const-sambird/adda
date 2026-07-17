import math
from dimod import BinaryQuadraticModel, make_quadratic, quicksum
from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler
from qiskit_optimization import QuadraticProgram
from qaoa import QAOAOptimiser
from util import square_bqm_to_binary_polynomial

BINARY_VARTYPE = 'BINARY'
SAFETY_FACTOR = 1

def create_slack_variables(name, S_max):
    """
    Create a BQM representing a binary-encoded slack variable with value in [0, S_max].
    The variable decomposes as sum_{k=0}^{floor(log2(S_max))} 2^k * s_k.
    """
    if S_max < 1:
        raise ValueError(f"S_max must be >= 1, got {S_max}")
    model = BinaryQuadraticModel('BINARY')
    for j in range(0, math.floor(math.log2(S_max)) + 1):
        model.add_linear(f'{name}-{j}', 2**j)
    return model


def max_abs_coefficient(bqm: BinaryQuadraticModel) -> float:
    """Return the largest absolute value of any coefficient in a BQM."""
    linear_max = max((abs(b) for _, b in bqm.iter_linear()), default=0.0)
    quad_max = max((abs(b) for _, _, b in bqm.iter_quadratic()), default=0.0)
    return max(linear_max, quad_max)

def max_sum_coefficient(bqm: BinaryQuadraticModel) -> float:
    linear_sum = sum((max(b, 0) for _, b in bqm.iter_linear()))
    quad_sum = sum((max(b, 0) for _, _, b in bqm.iter_quadratic()))

    return linear_sum + quad_sum

def penalty_lambda_from_objective(objective_bqm: BinaryQuadraticModel,
                                  multiplier: float = 1.0) -> float:
    """
    Compute a penalty lambda guaranteed to dominate the objective BQM.

    The sufficient condition for lambda*(violation)^2 to enforce a constraint
    over an objective H is:  lambda > |H_max - H_min|

    For a BQM with n variables and max coefficient magnitude M, the objective
    range is bounded by M * (n + n*(n-1)/2). Computing this from the assembled
    objective ensures correct scaling regardless of cost normalisation.

    Parameters
    ----------
    objective_bqm : the objective portion of the QUBO (before constraints)
    multiplier    : safety factor; use >1 for hard constraints, ~1 for soft ones
    """
    objective_max = max_sum_coefficient(objective_bqm)

    return objective_max ** 2

def omega(Q, U, I, c, f, n_replicas):
    cost = 0
    for q in Q:
        cost += f[q] * c[q]
    for u in U:
        cost += n_replicas * f[u] * c[u]
    return cost


def make_max_cost_qubo(Z_max, n_replicas, Q, U, I, c, f, v, m, alpha,
                       additional_constraints=None):
    """
    Build a QUBO for the divergent design tuning problem on a maximum cost basis.

    Lambda calibration strategy
    ---------------------------
    All penalty lambdas are derived from the assembled objective BQM's
    coefficient structure AFTER it is built, not from the raw problem
    parameters. This ensures correct scaling regardless of normalisation.

    lam_replica  : enforces z^(0) >= load_r for each replica r.
                   Multiplier=2 allows near-optimal feasible solutions to
                   remain distinguishable while enforcing feasibility.

    lam_routing  : hard constraint enforcing sum_r t_q^r = m.
                   Multiplier scales with n_replicas * |Q| so it dominates
                   the replica-load terms regardless of problem size.
                   This is the constraint that must never be violated.
    """
    assert alpha <= 1, 'the convex combination over all probabilities must total 1'
    # Sanity check: Z_max should be achievable by a single replica
    max_single_replica_load = sum(f[q] * c[q] / m for q in Q)
    if Z_max > max_single_replica_load * 2:
        import warnings
        warnings.warn(
            f'Z_max={Z_max} is more than 2x the maximum possible single-replica '
            f'load ({max_single_replica_load:.0f}). This inflates slack variable '
            f'coefficients and worsens the energy landscape. Consider tightening Z_max.'
        )
    if additional_constraints is None:
        additional_constraints = []

    # Build the objective BQM: min z^(0) = sum_k 2^k z_k
    objective = create_slack_variables('z', Z_max)

    # build the failure terms z^(j)
    subobjectives = []
    if alpha > 0:
        for r in range(n_replicas):
            subobjective = create_slack_variables(f'z^({r})', Z_max)
            subobjectives.append(subobjective)

    # lam_replica: must dominate the objective range.
    lam_replica = omega(Q, U, I, c, f, n_replicas) * SAFETY_FACTOR

    replica_load_bqms = []
    routing_bqms = []

    # Per-replica load constraints.
    # Encodes: z^(0) - sum_q qcost(q, I_r) - sum_u ucost(u, I_r) - s^(r) = 0
    for r in range(n_replicas):
        constraint_model = BinaryQuadraticModel('BINARY')

        # z^(0) terms
        for var, bias in objective.iter_linear():
            constraint_model.add_linear(var, bias)

        # Query cost: -f(q)/m * c_q * t_q^r  and  -f(q)/m * v_i^q * x_i^r * t_q^r
        for q in Q:
            constraint_model.add_linear(f't-q{q}-r{r}', -f[q] * c[q] / m)
            for i in I:
                constraint_model.add_quadratic(
                    f'x-i{i}-r{r}', f't-q{q}-r{r}',
                    f[q] * v[i][q] / m
                )

        # Update cost: constant offset and -f(u)*v_i^u * x_i^r linear terms
        for u in U:
            constraint_model.offset += -f[u] * c[u]
            for i in I:
                constraint_model.add_linear(f'x-i{i}-r{r}', -f[u] * v[i][u])

        # Slack s^(r) in [0, Z_max] absorbs z^(0) - load_r
        for var, bias in create_slack_variables(f's-r{r}', Z_max).iter_linear():
            constraint_model.add_linear(var, -bias)

        # The make_quadratic reduction strength must exceed the largest
        # coefficient in the expression being squared.
        max_coeff = max(
            (abs(b) for _, b in constraint_model.iter_linear()), default=1.0
        )
        hubo = square_bqm_to_binary_polynomial(constraint_model)
        qubo = make_quadratic(hubo, 2.0 * max_coeff, 'BINARY')
        qubo.scale(lam_replica)
        replica_load_bqms.append(qubo)

    # lam_routing must dominate the replica_load terms, not just the objective.
    # The replica_load BQMs are already scaled by lam_replica and involve
    # O(|I|*|Q|) variables whose squared interactions can be enormous.
    # Deriving lam_routing from the assembled replica_load BQM ensures it
    # genuinely outweighs whatever energy the annealer gains by violating routing.
    replica_load_combined = quicksum(replica_load_bqms) if replica_load_bqms else BinaryQuadraticModel('BINARY')
    lam_routing = (omega(Q, U, I, c, f, n_replicas) ** 3) + 1 * SAFETY_FACTOR

    # failure-aware replica load terms
    failure_bqms = []
    failure_routing_bqms = []
    if alpha > 0:
        for j in range(n_replicas):
            for r in range(n_replicas):
                if j == r: continue

                constraint_model = BinaryQuadraticModel('BINARY')
                # z^(j) terms
                for var, bias in subobjectives[j].iter_linear():
                    constraint_model.add_linear(var, bias)
                
                # Query cost: -f(q)/min{m, |R|-1} * c_q * t_q^(r,j)  and  -f(q)/min{m, |R|-1} * v_i^q * x_i^r * t_q^(r,j)
                for q in Q:
                    constraint_model.add_linear(f't-q{q}-r{r}-j{j}', -f[q] * c[q] / min(m, n_replicas - 1))
                    for i in I:
                        constraint_model.add_quadratic(
                            f'x-i{i}-r{r}', f't-q{q}-r{r}-j{j}',
                            f[q] * v[i][q] / min(m, n_replicas - 1)
                        )

                # Update cost: constant offset and -f(u)*v_i^u * x_i^r linear terms
                for u in U:
                    constraint_model.offset += -f[u] * c[u]
                    for i in I:
                        constraint_model.add_linear(f'x-i{i}-r{r}', -f[u] * v[i][u])

                # Slack s^(r) in [0, Z_max] absorbs z^(0) - load_r
                for var, bias in create_slack_variables(f's-j{j}-r{r}', Z_max).iter_linear():
                    constraint_model.add_linear(var, -bias)

                # The make_quadratic reduction strength must exceed the largest
                # coefficient in the expression being squared.
                max_coeff = max(
                    (abs(b) for _, b in constraint_model.iter_linear()), default=1.0
                )
                hubo = square_bqm_to_binary_polynomial(constraint_model)
                qubo = make_quadratic(hubo, 2.0 * max_coeff, 'BINARY')
                qubo.scale(lam_replica)
                failure_bqms.append(qubo)
        
        # failure routing constraints
        for q in Q:
            for j in range(n_replicas):
                constraint_model = BinaryQuadraticModel('BINARY')
                for r in range(n_replicas):
                    if j == r: continue
                    constraint_model.add_linear(f't-q{q}-r{r}-j{j}', 1)
                constraint_model.offset = -min(m, n_replicas - 1)
                hubo = square_bqm_to_binary_polynomial(constraint_model)
                qubo = make_quadratic(hubo, 1.0, 'BINARY')
                qubo.scale(lam_routing)
                failure_routing_bqms.append(qubo)

    # Hard routing constraint: lambda_routing * (sum_r t_q^r - m)^2
    for q in Q:
        constraint_model = BinaryQuadraticModel('BINARY')
        for r in range(n_replicas):
            constraint_model.add_linear(f't-q{q}-r{r}', 1)
        constraint_model.offset = -m
        hubo = square_bqm_to_binary_polynomial(constraint_model)
        qubo = make_quadratic(hubo, 1.0, 'BINARY')
        qubo.scale(lam_routing)
        routing_bqms.append(qubo)

    # Merge routing BQMs into one for decomposition reporting
    routing_combined = quicksum(routing_bqms) if routing_bqms else BinaryQuadraticModel('BINARY')

    # scale the objective according to the failure probability
    objective.scale(1 - alpha)
    for subobjective in subobjectives:
        subobjective.scale(alpha / n_replicas)

    full_qubo = quicksum([objective, *subobjectives, *replica_load_bqms, *routing_bqms, *failure_bqms, *failure_routing_bqms, *additional_constraints])

    components = {
        'replica_load': replica_load_combined,
        'routing':      routing_combined,
        'lam_replica':  lam_replica,
        'lam_routing':  lam_routing,
    }
    if additional_constraints:
        components['storage'] = quicksum(additional_constraints)

    return full_qubo, components


def make_total_cost_qubo(n_replicas, Q, U, I, c, f, v, m,
                         additional_constraints=None):
    """
    Build a QUBO for the divergent design tuning problem on a total cost basis.
    """
    if additional_constraints is None:
        additional_constraints = []

    objective = BinaryQuadraticModel('BINARY')
    for r in range(n_replicas):
        for q in Q:
            objective.add_linear(f't-q{q}-r{r}', -f[q] * c[q] / m)
            for i in I:
                objective.add_quadratic(
                    f'x-i{i}-r{r}', f't-q{q}-r{r}',
                    -f[q] * v[i][q] / m
                )
        for u in U:
            objective.offset += -f[u] * c[u]
            for i in I:
                objective.add_linear(f'x-i{i}-r{r}', -f[u] * v[i][u])

    lam_routing = (omega(Q, U, I, c, f, n_replicas) ** 3) * n_replicas + 1

    routing_bqms = []
    for q in Q:
        constraint_model = BinaryQuadraticModel('BINARY')
        for r in range(n_replicas):
            constraint_model.add_linear(f't-q{q}-r{r}', 1)
        constraint_model.offset = -m
        hubo = square_bqm_to_binary_polynomial(constraint_model)
        qubo = make_quadratic(hubo, 1.0, 'BINARY')
        qubo.scale(lam_routing)
        routing_bqms.append(qubo)

    routing_combined = quicksum(routing_bqms) if routing_bqms else BinaryQuadraticModel('BINARY')
    full_qubo = quicksum([objective, *routing_bqms, *additional_constraints])

    components = {
        'objective': objective,
        'routing':     routing_combined,
        'lam_routing': lam_routing,
    }
    if additional_constraints:
        components['storage'] = quicksum(additional_constraints)

    return full_qubo, components


def make_storage_constraint(r, candidates, costs, storage_budget, calibration_bqm, queries, updates, n_replicas, baseline):
    """
    Build a penalised storage budget constraint for replica r:
        storage_budget - sum_i w_i x_i^r - s = 0

    Lambda is derived from calibration_bqm, which should be the assembled
    replica_load BQM (after lam_replica scaling) so that lam_storage sits
    in the same tier as lam_routing — both above the replica_load energy range.

    Parameters
    ----------
    r               : replica index
    candidates      : list of index candidates (used only for count)
    costs           : normalised storage costs per candidate
    storage_budget  : normalised storage budget (must be >= 1)
    calibration_bqm : BQM from which to derive lambda (use replica_load_combined)
    """
    lam_storage = (omega(queries, updates, candidates, baseline, [1 for _ in range(len(queries) + len(updates))], n_replicas) ** 3) + 1

    constraint_model = BinaryQuadraticModel(vartype='BINARY')
    constraint_model.offset = storage_budget
    for i, cost in enumerate(costs):
        constraint_model.add_linear(f'x-i{i}-r{r}', -cost)
    for var, bias in create_slack_variables(
            f's-wmax-r{r}', max(1, int(storage_budget))).iter_linear():
        constraint_model.add_linear(var, -bias)

    max_coeff = max(
        (abs(b) for _, b in constraint_model.iter_linear()), default=1.0
    )
    hubo = square_bqm_to_binary_polynomial(constraint_model)
    qubo = make_quadratic(hubo, 2.0 * max_coeff, 'BINARY')
    qubo.scale(lam_storage)
    return qubo, lam_storage


def anneal(qubo: BinaryQuadraticModel, algorithm='anneal', mode='simulate', num_reads=100):
    assert algorithm in ('anneal', 'qaoa'), 'algorithm must be "anneal" or "qaoa"'
    assert mode in ('simulate', 'quantum'), 'mode must be "simulate" or "quantum"'
    if algorithm == 'anneal':
        if mode == 'simulate':
            sampler = SimulatedAnnealingSampler()
        elif mode == 'quantum':
            sampler = PathIntegralAnnealingSampler()
        return sampler.sample(qubo, num_reads=num_reads)
    elif algorithm == 'qaoa':
        optimiser = QAOAOptimiser(qubo.num_variables, 1, num_reads, mode)
        qiskit_qubo = QuadraticProgram()
        linear_dict = {}
        quadratic_dict = {}
        names = set()
        for vars, bias in qubo.to_qubo()[0].items():
            names.add(vars[0])
            names.add(vars[1])
            if vars[0] == vars[1]:
                linear_dict[vars[0]] = bias
            else:
                quadratic_dict[vars] = bias
        for name in names:
            qiskit_qubo.binary_var(name)
        qiskit_qubo.minimize(constant=qubo.offset, linear=linear_dict, quadratic=quadratic_dict)
        return optimiser.optimise(qiskit_qubo)
