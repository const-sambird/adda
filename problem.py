from dataclasses import dataclass

@dataclass
class Problem:
    '''
    Represents a divergent design tuning problem whose values have already
    been precomputed; ie, there is no need to invoke the cost estimator and
    parse a workload of queries.
    '''

    name: str
    baseline: list[int]
    benefits: list[list[int]]
    weights: list[int]
    budget: int

    def num_candidates(self) -> int:
        return len(self.benefits)


PROBLEMS: dict[str, Problem] = {
    'QAOA_TOY_TOTAL': Problem(
        'QAOA_TOY_TOTAL',
        [14, 12, 10, 8, 7, 7],
        [
            [4, 0, 4, 2, 5, 2],
            [0, 10, 0, 0, 0, 3],
            [8, 2, 4, 2, 1, 0],
            [0, 0, 2, 3, 0, 0]
        ],
        [],
        0
    ),
    'QAOA_TOY_MAX': Problem(
        'QAOA_TOY_MAX',
        [5, 2],
        [
            [3, 0],
            [2, 1],
        ],
        [2, 1],
        2
    )
}
