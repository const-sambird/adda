import psycopg
from index_candidate import IndexCandidate
from replica import Replica
from parser import WorkloadParser

def create_index_on(index: IndexCandidate, name, cur):
    cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % index.create_str(name))

def reset_indexes(cur):
    cur.execute('SELECT hypopg_reset();')

def get_benefit(query, indexes, replica):
    conn = psycopg.connect(replica.connection_string)
    cur = conn.cursor()

    for i, index in enumerate(indexes):
        create_index_on(index, f'idx_{i}', cur)
    
    cost = 0

    for statement in query.split(';'):
        if 'create view' in statement or 'drop view' in statement:
            cur.execute(statement)
        elif 'select' in statement:
            cur.execute('EXPLAIN (FORMAT JSON) %s' % statement)
            if after_timing := cur.fetchone()[0][0]['Plan']['Total Cost']:
                cost += int(after_timing)
    
    reset_indexes(cur)
    conn.close()
    return cost

def compute_delta_overlap(workload, indexes, replica):
    delta = 0
    s = 0

    for i, query in enumerate(workload):
        print(f'- {i + 1} / {len(workload)}')
        baseline = get_benefit(query, [], replica)
        s += baseline
        print(baseline)
        all_cost = get_benefit(query, indexes, replica)
        for to_remove in range(len(indexes)):
            removed = indexes[:to_remove] + indexes[to_remove + 1:]
            all_but_one = get_benefit(query, removed, replica)
            marginal_benefit = get_benefit(query, [indexes[to_remove]], replica)

            this_delta = abs(all_cost - (all_but_one + marginal_benefit - baseline))
            if this_delta > delta:
                delta = this_delta
    
    print('sum:', s)
    return delta

replica = Replica(1, 'localhost', '5432', 'tpchdb', 'sam')
parser = WorkloadParser(replica)
parser.read_queries('./workload')
parser.get_all_columns()
parser.extract_candidates()
print(compute_delta_overlap(parser.get_workload(), parser.get_candidates(), replica))