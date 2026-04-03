# ADDA

ADDA is an **A**nnealing-based **D**ivergent **D**esign **A**dvisor, used to solve the divergent design tuning problem.

## Installation

ADDA was built using Python 3.12.9, but it will probably work with any Python version that supports Qiskit, dimod, and psycopg.

Create a venv and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To create the queries (if not done so already in a pregenerated set) the prerequisites for the TPC-H and TPC-DS `qgen`/`dsqgen` programs need to be installed:

```bash
$ sudo apt-get install gcc make flex bison byacc git gcc-9
```

Then, download the runkits from the [TPC website](https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) and create a Makefile by renaming and editing `makefile.suite` to your system specifications. `qgen` will not compile on macOS without changing references from `malloc.h` to `stdlib.h`. (The experimental results for qDINA were run on Ubuntu 24.04, and a Linux environment is recommended for reproducibility).

> [!TIP]
> The [benchmarking utility](https://github.com/const-sambird/qdina-bench) has further utilities for creating a workload of queries using the TPC-H qgen utility. This is particularly relevant as the paper's results use the same workload for recommending indexes as evaluating results. It is strongly recommended that this is used for reproducing our results, though of course any workload should work.

## Configuration

ADDA assumes that there is access to a Postgres database running hypopg.

qDINA requires a `replicas.csv` file to list the database replicas to create (simulated) indexes on. The format that is expected for a single connection is

```
id,hostname,port,dbname,user,password,
```

| Field     | Explanation
|-----------|------------------------------------
| id        | A number to identify the database replica (1, 2, ...)
| hostname  | The IP address of the PostgreSQL database
| port      | Which port number to connect to (the default is 5432 but it must be specified)
| dbname    | The name of the database
| user      | The user to connect with. This user must have sufficient privileges on the database to create and drop hypothetical indexes and run EXPLAIN commands
| password  | The password for the user

One line per replica. Note that only one real database replica is necessary, as the only time the database is used is to invoke the cost estimator (to obtain the baseline costs, index candidates, and marginal index benefits). Accordingly, to generate a 4-replica divergent design, the same replica can simply be listed four times; eg,

```
1,localhost,5432,tpchdb,,
1,localhost,5432,tpchdb,,
1,localhost,5432,tpchdb,,
1,localhost,5432,tpchdb,,
```

## Running

By default, ADDA assumes that the inputs should be computed dynamically from the provided query workload and database connection. Alternatively, pre-computed coefficients may be used. These are defined in [`problem.py`](./problem.py).

```
usage: run.py [-h] [-q] [-a] [-A PENALTY_TERM_A] [-C PENALTY_TERM_C] [-w STORAGE_BUDGET] [--cost-normalisation-factor COST_NORMALISATION_FACTOR]
              [--benefit-normalisation-factor BENEFIT_NORMALISATION_FACTOR] [-n NUM_READS] [-d] [-p {QAOA_TOY_TOTAL,QAOA_TOY_MAX}]
              [--workload-path WORKLOAD_PATH]
              {total,max}

positional arguments:
  {total,max}           cost basis for objective function

options:
  -h, --help            show this help message and exit
  -q, --quantum         use a real quantum computer (for QAOA) or simulated quantum annealing (for annealing)
  -a, --qaoa            use the quantum approximate optimisation algorithm instead of annealing
  -w STORAGE_BUDGET, --storage-budget STORAGE_BUDGET
                        storage budget
  --cost-normalisation-factor COST_NORMALISATION_FACTOR
  --benefit-normalisation-factor BENEFIT_NORMALISATION_FACTOR
  -n NUM_READS, --num-reads NUM_READS
                        number of annealer reads
  -d, --dry-run         don't actually run the annealer
  -p {QAOA_TOY_TOTAL,QAOA_TOY_MAX}, --problem {QAOA_TOY_TOTAL,QAOA_TOY_MAX}
  --workload-path WORKLOAD_PATH
```

For the experiments reported in the paper, the following invocations were used:

```
# TPC-C replica study
python run.py -w 200000000 -q -n 10 --benefit-normalisation-factor 300 --cost-normalisation-factor 819200 max
# TPC-H replica study
python run.py -w 5000000000 -q -n 10 max
# Ablation study, annealing
python run.py -p QAOA_TOY_MAX -q -n 10 max
# Ablation study, simulated QAOA
python run.py -p QAOA_TOY_MAX -a -n 1024 max
# Ablation study, real QAOA
python run.py -p QAOA_TOY_MAX -a -n 1024 max
```

The query workload should be placed in a `workload/` folder. Each query should be saved in a file `[TEMPLATE_NO]_[QUERY_NO].sql`, where `TEMPLATE_NO` is the template number and `QUERY_NO` is the query number within each template. eg, `1_0.sql`.

## Other algorithms

We compared other algorithms against ADDA in our report.

* DINA and qDINA: [code found here](https://github.com/const-sambird/dina)
* DiversityClusterDB: [code found here](https://github.com/const-sambird/extend-dist)
* Benchmarking utility: [code found here](https://github.com/const-sambird/qdina-bench)
