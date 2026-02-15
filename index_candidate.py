from dataclasses import dataclass

@dataclass
class IndexCandidate:
    column: str
    table: str

    def create_str(self, name = 'idx'):
        return f'CREATE INDEX {name} ON {self.table} ({self.column})'
    
    def __repr__(self):
        return f'Index on {self.column} ({self.table})'
