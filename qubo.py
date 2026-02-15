import math
from qiskit_optimization import QuadraticProgram

class Algorithm:
    def __init__(self, candidates, n_templates, n_replicas):
        self.candidates = candidates
        self.n_candidates = len(candidates)
        self.n_templates = n_templates
        self.n_replicas = n_replicas
        self.linear_terms = {}
        self.quadratic_terms = {}

        self.qubo = QuadraticProgram()
    
    def create_qubo(self, benefits, costs, W_max, A, C):
        j_max = math.ceil(math.log2(W_max))
        for r in range(self.n_replicas):
            for i in range(self.n_candidates):
                self.qubo.binary_var(f'x_{i}^{r}')
                self.linear_terms[f'x_{i}^{r}'] = A * ((costs[i] ** 2) - (2 * W_max * costs[i]))
            for m in range(self.n_templates):
                self.qubo.binary_var(f't_{m}^{r}')
            for j in range(j_max):
                self.qubo.binary_var(f'b_{j}^{r}')
                self.linear_terms[f'b_{j}^{r}'] = A * (pow(2, 2*j) - (2 * W_max * pow(2, j)))
            
            # quadratic terms
            for i in range(self.n_candidates):
                for k in range(i + 1, self.n_candidates):
                    self.quadratic_terms[(f'x_{i}^{r}', f'x_{k}^{r}')] = 2 * A * costs[i] * costs[k]
            
            for j in range(j_max):
                for l in range(j + 1, j_max):
                    self.quadratic_terms[(f'b_{j}^{r}', f'b_{l}^{r}')] = 2 * A * pow(2, j + l)
            
            for i in range(self.n_candidates):
                for j in range(j_max):
                    self.quadratic_terms[(f'x_{i}^{r}', f'b_{j}^{r}')] = 2 * A * costs[i] * pow(2, j)
            
            for i in range(self.n_candidates):
                for m in range(self.n_templates):
                    self.quadratic_terms[(f'x_{i}^{r}', f't_{m}^{r}')] = -1 * benefits[i][m]
            
        # routing table mutual exclusion constraint
        for m in range(self.n_templates):
            for r_i in range(self.n_replicas):
                for r_j in range(r_i + 1, self.n_replicas):
                    self.quadratic_terms[(f't_{m}^{r_i}', f't_{m}^{r_j}')] = C
    
        self.qubo.minimize(constant=0, linear=self.linear_terms, quadratic=self.quadratic_terms)

        return self.qubo
        
