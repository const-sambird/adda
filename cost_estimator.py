import psycopg

class CostEstimator:
    def __init__(self, replicas, candidates, workload, templates, n_templates):
        self.replica = replicas[0]
        self.candidates = candidates
        self.workload = workload
        self.templates = templates
        self.n_candidates = len(candidates)
        self.n_templates = n_templates
        self.baseline = []
    
    def get_benefits(self):
        baseline = [0 for _ in range(self.n_templates)]
        benefits = []
        for _ in range(self.n_candidates):
            benefits.append([0 for _ in range(self.n_templates)])
        
        with psycopg.connect(self.replica.connection_string) as conn:
            with conn.cursor() as cur:
                # compute baseline
                print('+ computing baseline query costs...')
                for idx, query in enumerate(self.workload):
                    for statement in query.split(';'):
                        if 'create view' in statement or 'drop view' in statement:
                            cur.execute(statement)
                        elif 'select' in statement:
                            cur.execute('EXPLAIN (FORMAT JSON) %s' % statement)
                            if after_timing := cur.fetchone()[0][0]['Plan']['Total Cost']:
                                baseline[self.templates[idx]] += int(after_timing)
                print('+ computing index candidate benefits for each query type')
                for i_candidate, candidate in enumerate(self.candidates):
                    print('-', i_candidate + 1, '/', self.n_candidates)
                    cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % candidate.create_str())

                    query_costs = [0 for _ in range(self.n_templates)]

                    for i_query, query in enumerate(self.workload):
                        for statement in query.split(';'):
                            if 'create view' in statement or 'drop view' in statement:
                                cur.execute(statement)
                            elif 'select' in statement:
                                cur.execute('EXPLAIN (FORMAT JSON) %s' % statement)
                                if after_timing := cur.fetchone()[0][0]['Plan']['Total Cost']:
                                    query_costs[self.templates[i_query]] += int(after_timing)
                    
                    for template in range(self.n_templates):
                        benefits[i_candidate][template] = baseline[template] - query_costs[template]
                    
                    cur.execute('SELECT hypopg_reset();')
        
        self.baseline = baseline

        return benefits

    def get_storage_costs(self):
        costs = [0 for _ in range(self.n_candidates)]

        with psycopg.connect(self.replica.connection_string) as conn:
            with conn.cursor() as cur:
                print('+ computing storage costs for each index candidate')
                for i, candidate in enumerate(self.candidates):
                    cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % candidate.create_str())
                    virtual_oid = cur.fetchone()[0]
                    cur.execute('SELECT hypopg_relation_size(%s) FROM hypopg_list_indexes;' % virtual_oid)
                    computed_size = cur.fetchone()[0]
                    cur.execute('SELECT hypopg_drop_index(%s);' % virtual_oid)
                    costs[i] = computed_size
        
        return costs

    def get_baseline(self):
        return self.baseline
