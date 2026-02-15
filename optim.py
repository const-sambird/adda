import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
 
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit_aer import AerSimulator

class QAOAOptimiser:
    def __init__(self, benefits, weights, budget, reps = 1, shots = 1024, mode = 'simulate'):
        assert mode == 'simulate' or mode == 'quantum', 'select a supported solver'
        self.benefits = benefits
        self.weights = weights
        self.budget = budget
        self.reps = reps
        self.shots = shots
        self.mode = mode
        self.n_indexes = len(benefits)

        if mode == 'quantum':
            self.service = QiskitRuntimeService()
            self.backend = self.service.least_busy(
                operational=True, simulator=False, min_num_qubits=127
            )
        else:
            self.backend = AerSimulator(method='matrix_product_state')
        
        print('initialised qiskit connection with backend', self.backend)

        self.pass_manager = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
    
    def get_qaoa_ansatz(self, operator):
        qc = QAOAAnsatz(cost_operator=operator, reps=self.reps, initial_state=None)
        qc.measure_active()
        return qc
    
    def optimise(self, qubo):
        sampler = Sampler(mode=self.backend, options={'default_shots': self.shots})
        n_executions = 0

        operator, offset = qubo
        ansatz = self.get_qaoa_ansatz(operator)
        ansatz = self.pass_manager.run(ansatz)
        print(ansatz.size(), 'operations')

        initial_gamma = np.pi
        initial_beta = np.pi / 2
        init_params = []

        for _ in range(self.reps):
            init_params.append(initial_beta)
        for _ in range(self.reps):
            init_params.append(initial_gamma)
        
        def qaoa_objective(params, ansatz, hamiltonian, sampler):
            '''
            Sampler-based approximation of an Estimator call for the
            classical optimiser to get expectation values for the
            trainable QAOA parameters
            '''
            nonlocal n_executions
            n_executions += 1
            # Bind parameters into the QAOA ansatz
            circ = ansatz.assign_parameters(params)
            
            # Run circuit with Sampler primitive (hardware or simulator)
            job = sampler.run([circ])
            result = job.result()[0]
            counts = result.data.meas.get_counts()
            
            # Convert to probabilities
            total_shots = sum(counts.values())
            probs = {bit: count / total_shots for bit, count in counts.items()}

            # Compute expected energy
            energy = 0.0
            for bitstring, p in probs.items():
                # Compute classical energy E(z)
                z = np.array([int(b) for b in bitstring[::-1]])  # Qiskit uses little endian
                E_z = 0.0
                for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
                    # Only diagonal Z terms matter
                    label = pauli.to_label()[::-1]
                    term_val = 1.0
                    for bit, pauli_char in zip(z, label):
                        if pauli_char == "Z":
                            term_val *= (1 - 2 * bit)  # Z eigenvalues: +1 for |0>, -1 for |1>
                        elif pauli_char == "I":
                            term_val *= 1
                        else:
                            # X or Y terms vanish under Z measurement
                            term_val = 0.0
                            break
                    E_z += coeff.real * term_val
                energy += p * E_z

            return energy.real

        param_est = minimize(
            qaoa_objective,
            init_params,
            args=(ansatz, operator, sampler),
            method="COBYLA",
            tol=1e-2,
            #options={'maxiter': 4}
        )
        opt_params = param_est.x
        bound_circ = ansatz.assign_parameters(opt_params)

        pub = (bound_circ,)
        job = sampler.run([pub])
        n_executions += 1

        counts_int = job.result()[0].data.meas.get_int_counts()
        shots = sum(counts_int.values())
        final_distribution_int = {key: val / shots for key, val in counts_int.items()}

        solution = []
        
        for bits, freq in final_distribution_int.items():
            b = list(np.binary_repr(bits, width=ansatz.num_qubits))
            b.reverse()
            b = [int(bit) for bit in b]
            solution.append((b, -freq))
        
        solution = np.array(solution, dtype=[('sample', 'i1', (ansatz.num_qubits,)), ('energy', '<f8')])
        solution = solution.view(np.recarray)
        print(solution)

        # 5. compute x_cost, x_feas using samples
        x_cost = solution[np.recarray.argmin(solution.energy)]
        
        print('quantum circuit executions:', n_executions)
        return solution, x_cost
