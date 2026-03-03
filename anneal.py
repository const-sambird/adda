import math
from dimod import ExactSolver, BinaryQuadraticModel, make_quadratic, quicksum
from dwave.samplers import PathIntegralAnnealingSampler
from util import square_bqm_to_binary_polynomial

BINARY_VARTYPE = 'BINARY'


def create_slack_variables(name, S_max):
    """
    Create a BQM representing a binary-encoded slack variable with value in [0, S_max].
    The variable decomposes as sum_{k=0}^{floor(log2(S_max))} 2^k * s_k.
    """
    model = BinaryQuadraticModel('BINARY')
    for j in range(0, math.floor(math.log2(S_max)) + 1):
        model.add_linear(f'{name}-{j}', 2**j)
    return model


def estimate_objective_scale(Z_max):
    """
    Return a rough upper bound on the objective value, used to calibrate
    penalty strengths. For the max-cost formulation, the objective is bounded
    above by Z_max (the maximum possible replica workload cost).
    """
    return float(Z_max)


def make_replica_load_lambda(Z_max):
    """
    Lambda for the per-replica load constraint:
        z^(0) - sum_q qcost(q, I_r) - sum_u ucost(u, I_r) - s^(r) = 0

    Must be large enough that violating this constraint costs more than the
    entire objective range. We use Z_max as the objective scale, so a violation
    of size 1 (one cost unit) should cost more than Z_max.

    lambda_replica * 1^2 > Z_max  =>  lambda_replica > Z_max
    We use a factor of 2 as a safety margin.
    """
    return 2.0 * Z_max


def make_routing_lambda(Z_max):
    """
    Lambda for the hard routing constraint: sum_r t_q^r = m.

    This must be much larger than lambda_replica, so that routing feasibility
    is enforced before load balancing. A factor of Z_max^2 ensures this
    constraint dominates even large objective values. This is the one constraint
    where we intentionally allow it to overwhelm the objective.
    """
    return Z_max ** 2


def make_storage_lambda(Z_max):
    """
    Lambda for the storage budget constraint.

    Should be in the same order as lambda_replica — strong enough to enforce
    the constraint, but not so large that it swamps the objective and prevents
    near-optimal solutions from being distinguishable.
    """
    return 2.0 * Z_max


def make_max_cost_qubo(Z_max, n_replicas, Q, U, I, c, f, v, m, additional_constraints=[]):
    """
    Build a QUBO for the divergent design tuning problem on a maximum cost basis.

    The objective is:
        min z^(0)
    subject to:
        z^(0) >= sum_q qcost(q, I_r) + sum_u ucost(u, I_r)  for all r
        sum_r t_q^r = m                                        for all q

    The max operator is encoded via:
        (z^(0) - sum_q qcost(q, I_r) - sum_u ucost(u, I_r) - s^(r))^2 = 0

    where s^(r) is a non-negative slack variable absorbing the gap between
    z^(0) and the actual replica r load.

    Parameters
    ----------
    Z_max : int
        Upper bound on the maximum replica workload cost (after normalisation).
        Must satisfy Z_max >= max_r (sum_q qcost(q, I_r) + sum_u ucost(u, I_r)).
        If Z_max is too small, the slack variables cannot represent the required
        range and some feasible solutions will appear infeasible.
    n_replicas : int
    Q : list of query template indices
    U : list of update statement indices
    I : list of index candidate indices
    c : list of baseline costs, indexed by statement
    f : list of frequency weights, indexed by statement
    v : list of lists of index benefits, v[i][q] = benefit of index i on query q
    m : routing multiplicity factor
    additional_constraints : list of pre-scaled BQMs to add to the objective
    """
    lam_replica = make_replica_load_lambda(Z_max)
    lam_routing = make_routing_lambda(Z_max)

    # Objective: minimise z^(0), the encoded upper bound on replica load.
    # z^(0) = sum_{k} 2^k z_k
    objective = create_slack_variables('z', Z_max)
    models = []

    for r in range(n_replicas):
        # Build the expression:
        #   z^(0) - sum_q [f(q)/m * (c_q + sum_i v_i^q x_i^r) * t_q^r]
        #         - sum_u [f(u) * (c_u + sum_i v_i^u x_i^r)]
        #         - s^(r)
        # and penalise its square.
        constraint_model = BinaryQuadraticModel('BINARY')

        # z^(0) terms: copy linear biases from the objective slack variables
        for var, bias in objective.iter_linear():
            constraint_model.add_linear(var, bias)

        # Query cost terms: -f(q)/m * c_q * t_q^r  (linear in t)
        #                   -f(q)/m * v_i^q * x_i^r * t_q^r  (quadratic)
        for q in Q:
            constraint_model.add_linear(f't-q{q}-r{r}', -f[q] * c[q] / m)
            for i in I:
                constraint_model.add_quadratic(
                    f'x-i{i}-r{r}', f't-q{q}-r{r}',
                    -f[q] * v[i][q] / m
                )

        # Update cost terms: -f(u) * c_u  (constant, goes into offset)
        #                    -f(u) * v_i^u * x_i^r  (linear in x)
        for u in U:
            constraint_model.offset += -f[u] * c[u]
            for i in I:
                constraint_model.add_linear(f'x-i{i}-r{r}', -f[u] * v[i][u])

        # Slack variable s^(r): absorbs z^(0) - load_r >= 0
        # Must have range [0, Z_max] to cover all possible gaps.
        for var, bias in create_slack_variables(f's-r{r}', Z_max).iter_linear():
            constraint_model.add_linear(var, -bias)

        # Square the expression and reduce to QUBO.
        # The strength passed to make_quadratic is the penalty for the
        # auxiliary variables introduced in the HUBO->QUBO reduction.
        # It must be large enough to enforce the auxiliary variable equalities,
        # but using lam_replica here would double-count; use a moderate value
        # proportional to the largest coefficient in the constraint expression.
        max_coeff = max(abs(b) for _, b in constraint_model.iter_linear())
        hubo_reduction_strength = 2.0 * max_coeff

        hubo = square_bqm_to_binary_polynomial(constraint_model)
        qubo = make_quadratic(hubo, hubo_reduction_strength, 'BINARY')

        # Apply the replica load penalty lambda.
        qubo.scale(lam_replica)
        models.append(qubo)

    # Hard routing constraint: sum_r t_q^r = m for each query q.
    # Encoded as lambda_routing * (sum_r t_q^r - m)^2.
    for q in Q:
        constraint_model = BinaryQuadraticModel('BINARY')
        for r in range(n_replicas):
            constraint_model.add_linear(f't-q{q}-r{r}', 1)
        constraint_model.offset = -m
        hubo = square_bqm_to_binary_polynomial(constraint_model)
        # Routing constraint is already quadratic after squaring (linear terms
        # squared give at most quadratic), so make_quadratic strength is unused.
        qubo = make_quadratic(hubo, 1.0, 'BINARY')
        qubo.scale(lam_routing)
        models.append(qubo)

    return quicksum([objective, *models, *additional_constraints])


def make_total_cost_qubo(n_replicas, Q, U, I, c, f, v, m, additional_constraints=[]):
    """
    Build a QUBO for the divergent design tuning problem on a total cost basis.

    The objective is already linear/quadratic; only the routing constraint
    and any additional constraints need penalty encoding.
    """
    # For the total cost basis, Z_max is not directly available, so we
    # estimate the objective scale from the sum of all baseline costs.
    Z_max_estimate = sum(c[q] * f[q] for q in Q) + sum(c[u] * f[u] for u in U)
    lam_routing = make_routing_lambda(Z_max_estimate)

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

    models = []

    # Hard routing constraint: sum_r t_q^r = m
    for q in Q:
        constraint_model = BinaryQuadraticModel('BINARY')
        for r in range(n_replicas):
            constraint_model.add_linear(f't-q{q}-r{r}', 1)
        constraint_model.offset = -m
        hubo = square_bqm_to_binary_polynomial(constraint_model)
        qubo = make_quadratic(hubo, 1.0, 'BINARY')
        qubo.scale(lam_routing)
        models.append(qubo)

    return quicksum([objective, *models, *additional_constraints])


def make_storage_constraint(r, candidates, costs, storage_budget, Z_max):
    """
    Build a penalised storage budget constraint for replica r:
        sum_i w_i x_i^r <= storage_budget
    encoded as:
        lambda_storage * (storage_budget - sum_i w_i x_i^r - s)^2 = 0

    Parameters
    ----------
    r : int, replica index
    candidates : list of index candidate indices
    costs : list of storage costs per candidate (normalised)
    storage_budget : int, normalised storage budget
    Z_max : int, used to calibrate lambda
    """
    lam_storage = make_storage_lambda(Z_max)

    constraint_model = BinaryQuadraticModel(vartype='BINARY')
    constraint_model.offset = storage_budget
    for i, cost in enumerate(costs):
        constraint_model.add_linear(f'x-i{i}-r{r}', -cost)
    for var, bias in create_slack_variables(f's-wmax-r{r}', storage_budget).iter_linear():
        constraint_model.add_linear(var, -bias)

    max_coeff = max(abs(b) for _, b in constraint_model.iter_linear())
    hubo_reduction_strength = 2.0 * max_coeff

    hubo = square_bqm_to_binary_polynomial(constraint_model)
    qubo = make_quadratic(hubo, hubo_reduction_strength, 'BINARY')
    qubo.scale(lam_storage)
    return qubo


def anneal(qubo, mode='exact', num_reads=100):
    assert mode in ('exact', 'anneal'), 'mode must be "exact" or "anneal"'
    if mode == 'exact':
        sampler = ExactSolver()
    elif mode == 'anneal':
        sampler = PathIntegralAnnealingSampler()
    return sampler.sample(qubo, num_reads=num_reads)