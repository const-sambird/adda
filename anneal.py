import math
from dimod import ExactSolver, BinaryQuadraticModel, make_quadratic, quicksum
from dwave.samplers import PathIntegralAnnealingSampler
from util import square_bqm_to_binary_polynomial

BINARY_VARTYPE = 'BINARY'

def create_slack_varibles(name, S_max):
    model = BinaryQuadraticModel('BINARY')
    for j in range(0, math.floor(math.log2(S_max))):
        model.add_linear(f'{name}-{j}', 2**j)
    return model

def apply_penalty_lagrangian(model: BinaryQuadraticModel, lam: float = None):
    if lam is None:
        lam = 10 * abs(max(model.to_qubo()[0].values(), key=abs)) # same as dimod default
    
    model.scale(lam)

def make_qubo(Z_max, n_replicas, Q, U, I, c, f, v, m, additional_constraints = []):
    objective = create_slack_varibles('z', Z_max)
    models = []

    for r in range(n_replicas):
        constraint_model = BinaryQuadraticModel('BINARY')

        constraint_model.add_variables_from(objective.iter_linear())
        for q in Q:
            constraint_model.add_linear(f't-q{q}-r{r}', -1 * f[q] * c[q] / m)
            for i in I:
                constraint_model.add_quadratic(f'x-i{i}-r{r}', f't-q{q}-r{r}', f[q] * -v[i][q] / m)
        
        for u in U:
            constraint_model.offset += -1 * c[u] * f[u]
            for i in I:
                constraint_model.add_linear()

        for var, bias in create_slack_varibles(f's-r{r}', Z_max).iter_linear():
            constraint_model.add_linear(var, -bias)

        hubo = square_bqm_to_binary_polynomial(constraint_model)
        qubo = make_quadratic(hubo, 10000, 'BINARY')
        apply_penalty_lagrangian(qubo)
        models.append(qubo)
    
    # \sum_r t_q^r = m
    for q in Q:
        constraint_model = BinaryQuadraticModel('BINARY')
        for r in range(n_replicas):
            constraint_model.add_linear(f't-q{q}-r{r}', 1)
        constraint_model.offset = -m
        hubo = square_bqm_to_binary_polynomial(constraint_model)
        qubo = make_quadratic(hubo, 1, 'BINARY')
        apply_penalty_lagrangian(qubo, 1e50)
        models.append(qubo)
    
    for c in additional_constraints:
        apply_penalty_lagrangian(c)

    return quicksum([objective, *models, *additional_constraints])

def anneal(qubo, mode = 'exact', num_reads=100):
    assert mode == 'exact' or mode == 'anneal', 'use exact solver or simulated quantum annealing?'
    if mode == 'exact':
        sampler = ExactSolver()
    elif mode == 'anneal':
        sampler = PathIntegralAnnealingSampler()
    
    return sampler.sample(qubo, num_reads=num_reads)
