from dataclasses import dataclass

@dataclass
class Problem:
    name: str
    benefits: list[int]
    weights: list[int]
    budget: int

    def num_candidates(self) -> int:
        return len(self.benefits)
    
QIA_PROBLEMS = {
    'I5': Problem('I5', [14, 12, 10, 8, 3], [7, 6, 4, 2, 1], 19),
    'I6': Problem('I6', [14, 12, 10, 8, 3, 13], [7, 6, 4, 2, 1, 7], 19),
    'I7': Problem('I7', [14, 12, 10, 8, 3, 13, 12], [7, 6, 4, 2, 1, 7, 6], 19),
    'CDB_I7_ADJ': Problem('CDB_I7_ADJ', [4, 5, 27, 27, 27, 1, 1], [126, 114, 3, 72, 95, 1, 4], 75),
    'CDB_I7': Problem('CDB_I7', [165811, 178871, 1213770, 1213770, 1213770, 44370, 44370], [266, 232, 8, 132, 199, 2, 9], 140)
}

TRUMMER_PROBLEM = Problem('Trummer', [], [], 0)

PROBLEMS = {
    **QIA_PROBLEMS,
    'Trummer': TRUMMER_PROBLEM
}
