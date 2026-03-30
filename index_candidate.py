from dataclasses import dataclass

@dataclass
class IndexCandidate:
    column: str
    table: str

    def create_str(self, name = 'idx'):
        return f'CREATE INDEX {name} ON {self.table} ({self.column})'
    
    def __repr__(self):
        return f'Index on {self.column} ({self.table})'

class DummyIndexCandidate:
    def __init__(self, index: int, column: str = None, table: str = 'none'):
        self.index = index
        self.table = table

        if not column:
            self.column = f'idx_{index}'
        else:
            self.column = column

    def create_str(self, name = 'idx'):
        return f'SELECT 0'
    
    def __repr__(self):
        return f'Index placeholder #{self.index}'
