import numpy as np
from scipy.optimize import minimize
from dimod import SampleSet

from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.optimizers import COBYLA
 
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from qiskit_aer import AerSimulator

class QAOAOptimiser:
    def __init__(self, n_qubits, reps = 1, shots = 1024, mode = 'simulate'):
        assert mode == 'simulate' or mode == 'quantum', 'select a supported solver'
        self.reps = reps
        self.shots = shots
        self.mode = mode
        self.n_qubits = n_qubits

        if mode == 'quantum':
            self.service = QiskitRuntimeService()
            self.backend = self.service.least_busy(
                operational=True, simulator=False, min_num_qubits=max(127, n_qubits)
            )
        else:
            self.backend = AerSimulator()
        
        print('initialised qiskit connection with backend', self.backend)

        self.pass_manager = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
    
    def get_qaoa_ansatz(self, operator):
        qc = QAOAAnsatz(cost_operator=operator, reps=self.reps, initial_state=None)
        qc.measure_active()
        return qc
    
    def optimise(self, qubo: QuadraticProgram):
        sampler = Sampler(mode=self.backend, options={'default_shots': self.shots})
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=20), reps=self.reps, pass_manager=self.pass_manager)
        optimiser = MinimumEigenOptimizer(qaoa)
        result = optimiser.solve(qubo)

        sample = result.x
        sample = SampleSet.from_samples(sample, 'BINARY', 0)
        sample.relabel_variables({i: v.name for i, v in enumerate(result.variables)})

        return sample
