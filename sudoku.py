from typing import *
from copy import *
from enum import IntEnum
from itertools import combinations, groupby
from dataclasses import dataclass, field
from collections import deque

# Contract that SudokuInt is an int between 1 and 9 inclusive, no enforcement to check
SudokuInt = int
def is_sudoku_int(x: int) -> TypeGuard[SudokuInt]:
	return 1 <= x <= 9
# A runtime check, hope not to use it
def assert_sudoku_int(x: SudokuInt) -> None:
	assert is_sudoku_int(x), f"Value {x} is not a valid SudokuInt [1-9]."

SudokuIndex = int
def is_sudoku_index(x: int) -> TypeGuard[SudokuIndex]:
	return 0 <= x < 9
def assert_sudoku_index(x: SudokuIndex) -> None:
	assert is_sudoku_index(x), f"Value {x} is not a valid SudokuIndex [0-9)."

@dataclass(eq=False)
class PartialSudoku:
	@dataclass
	class Cell:
		"""
		Represents a cell in a Sudoku puzzle.
		_value: A list of booleans indicating which numbers (1-9) are possible in this cell.
		is_valid: True if at least one number is possible in this cell, False if no numbers are possible (indicating an invalid state).
		is_fixed_number: True if only one number is possible in this cell, indicating that it is a fixed number in the puzzle.
		"""
		value: List[bool] = field(default_factory=lambda: [True] * 9)

		def is_valid(self) -> bool:
			return any(self.value)
		def is_fixed_number(self) -> bool:
			return self.value.count(True) == 1
		@property
		def number(self) -> SudokuInt:
			if self.is_fixed_number():
				return self.value.index(True) + 1
			else:
				raise ValueError("Cell does not contain a fixed number.")
		@number.setter
		def number(self, num: SudokuInt):
			assert_sudoku_int(num)
			self.value = [False] * 9
			self.value[num - 1] = True

		@property
		def notes(self) -> Set[SudokuInt]:
			return {i + 1 for i in range(9) if self.value[i]}
		@notes.setter
		def notes(self, nums: Set[SudokuInt]):
			for num in nums:
				assert_sudoku_int(num)
			self.value = [i + 1 in nums for i in range(9)]
		def toggle_note(self, num: SudokuInt) -> None:
			assert_sudoku_int(num)
			self.value[num - 1] = not self.value[num - 1]

		def serialize(self, blank: str = ".") -> str:
			return "".join(str(i + 1) if self.value[i] else blank for i in range(9))
		@classmethod
		def deserialize(cls, text: str, blank: str = ".") -> Self:
			assert len(text) == 9, "Input text must be 9 characters long."
			value = [(c != blank) for c in text]
			return cls(value)

	board: List[List[Cell]] = field(default_factory=lambda: [[PartialSudoku.Cell() for _ in range(9)] for _ in range(9)])

	def __post_init__(self) -> None:
		assert len(self.board) == 9 and all(len(row) == 9 for row in self.board)

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, PartialSudoku):
			return False
		return all(self.board[r][c].value == other.board[r][c].value for r in range(9) for c in range(9))

	def serialize(self) -> str:
		return "".join("".join(cell.serialize() for cell in row) for row in self.board)
	@classmethod
	def deserialize(cls, text: str) -> Self:
		assert len(text) == 729, f"Input text must be 729 characters long. Got {len(text)}."
		return cls([
			[cls.Cell.deserialize(text[(i * 81) + (j * 9):(i * 81) + (j * 9) + 9]) for j in range(9)]
			for i in range(9)
		])

	@staticmethod
	def iter_units() -> Generator[List[Tuple[SudokuIndex, SudokuIndex]]]:
		# row units
		for r in range(9):
			yield [(r, c) for c in range(9)]
		# col units
		for c in range(9):
			yield [(r, c) for r in range(9)]
		# box units
		for br in range(3):
			for bc in range(3):
				yield [(3 * br + r, 3 * bc + c) for r in range(3) for c in range(3)]

	def fill_number(self, row: SudokuIndex, col: SudokuIndex, num: SudokuInt) -> None:
		assert_sudoku_index(row)
		assert_sudoku_index(col)
		assert_sudoku_int(num)
		self.board[row][col].number = num

	def validate(self) -> bool:
		for i in range(9):
			row_nums = set()
			col_nums = set()
			box_nums = set()
			for j in range(9):
				if not self.board[i][j].is_valid():
					return False
				# Check row
				if self.board[i][j].is_fixed_number():
					num = self.board[i][j].number
					if num in row_nums:
						return False
					row_nums.add(num)
				# Check column
				if self.board[j][i].is_fixed_number():
					num = self.board[j][i].number
					if num in col_nums:
						return False
					col_nums.add(num)
				# Check box
				r, c = 3 * (i // 3) + (j // 3), 3 * (i % 3) + (j % 3)
				if self.board[r][c].is_fixed_number():
					num = self.board[r][c].number
					if num in box_nums:
						return False
					box_nums.add(num)
		return True

	def validate_notes(self) -> bool:
		for r in range(9):
			for c in range(9):
				cell = self.board[r][c]
				if cell.is_fixed_number():
					num = cell.number
					for rr in range(9):
						if rr != r:
							if self.board[rr][c].value[num - 1]:
								return False
					for cc in range(9):
						if cc != c:
							if self.board[r][cc].value[num - 1]:
								return False
					for rr in range(3 * (r // 3), 3 * (r // 3) + 3):
						for cc in range(3 * (c // 3), 3 * (c // 3) + 3):
							if rr != r or cc != c:
								if self.board[rr][cc].value[num - 1]:
									return False
		return True

	def validate_full(self) -> bool:
		return self.validate() and self.validate_notes()

	def rebuild_notes(self) -> None:
		self.board = Sudoku.from_fixed_numbers(self).board

	def draw(self) -> None:
		def cell_3x3(r: int, c: int) -> List[str]:
			cell = self.board[r][c]
			if cell.is_fixed_number():
				return ["   ", f"*{cell.number}*", "   "]
			has = set(cell.notes)
			return [
				"".join(str(d) if d in has else "." for d in (1, 2, 3)),
				"".join(str(d) if d in has else "." for d in (4, 5, 6)),
				"".join(str(d) if d in has else "." for d in (7, 8, 9)),
			]
		gap = " " * 3
		margin = " " * 2
		seg_content_w = 3 * 3 + 2 * len(gap)
		seg_w = seg_content_w + 2 * len(margin)
		top = "┌" + "─" * seg_w + "┬" + "─" * seg_w + "┬" + "─" * seg_w + "┐"
		mid = "├" + "─" * seg_w + "┼" + "─" * seg_w + "┼" + "─" * seg_w + "┤"
		bot = "└" + "─" * seg_w + "┴" + "─" * seg_w + "┴" + "─" * seg_w + "┘"
		def seg_line(parts3: List[str]) -> str:
			content = gap.join(parts3)
			return f"{margin}{content}{margin}"
		def spacer_line() -> str:
			return f"│{' ' * seg_w}│{' ' * seg_w}│{' ' * seg_w}│"
		print(top)
		for r in range(9):
			cells = [cell_3x3(r, c) for c in range(9)]
			for sr in range(3):
				seg0 = seg_line([cells[c][sr] for c in range(0, 3)])
				seg1 = seg_line([cells[c][sr] for c in range(3, 6)])
				seg2 = seg_line([cells[c][sr] for c in range(6, 9)])
				print(f"│{seg0}│{seg1}│{seg2}│")
			match r:
				case 2 | 5:
					print(mid)
				case 8:
					print(bot)
				case _:
					print(spacer_line())

	def interactive_mode(self) -> SudokuInteractive:
		return SudokuInteractive(self)

class Sudoku(PartialSudoku):
	@overload
	def __init__(self, partial: PartialSudoku, /, *, copy_board = True, validated: bool = False) -> None: ...
	@overload
	def __init__(self, *args, **kwargs) -> None: ...
	def __init__(self, *args, copy_board = True, validated: bool = False, **kwargs) -> None:
		match args:
			case (PartialSudoku() as partial,):
				if copy_board:
					self.board = deepcopy(partial.board)
				else:
					self.board = partial.board
			case _:
				super().__init__(*args, **kwargs)
		if validated or (not args and not kwargs):
			return
		if not PartialSudoku.validate(self):
			raise ValueError("Invalid Sudoku board: violates Sudoku rules.")
		# envalidate notes
		for r in range(9):
			for c in range(9):
				cell = self.board[r][c]
				if cell.is_fixed_number():
					num = cell.number
					# Remove num from notes of related cells
					for rr in range(9):
						if rr != r:
							self.board[rr][c].value[num - 1] = False
					for cc in range(9):
						if cc != c:
							self.board[r][cc].value[num - 1] = False
					for rr in range(3 * (r // 3), 3 * (r // 3) + 3):
						for cc in range(3 * (c // 3), 3 * (c // 3) + 3):
							if rr != r or cc != c:
								self.board[rr][cc].value[num - 1] = False

	def delete_note(self, row: SudokuIndex, col: SudokuIndex, num: SudokuInt) -> None:
		cell = self.board[row][col]
		cell_past_is_fixed_number = cell.is_fixed_number()
		cell.value[num - 1] = False
		if not cell_past_is_fixed_number and cell.is_fixed_number():
			self.fill_number(row, col, cell.number)

	@override
	def fill_number(self, row: SudokuIndex, col: SudokuIndex, num: SudokuInt) -> None:
		assert_sudoku_index(row)
		assert_sudoku_index(col)
		assert_sudoku_int(num)
		self.board[row][col].number = num
		for r in range(9):
			if r != row:
				self.delete_note(r, col, num)
		for c in range(9):
			if c != col:
				self.delete_note(row, c, num)
		for r in range(3 * (row // 3), 3 * (row // 3) + 3):
			for c in range(3 * (col // 3), 3 * (col // 3) + 3):
				if r != row or c != col:
					self.delete_note(r, c, num)

	@override
	def validate(self) -> bool:
		in_cell_valid = all([cell.is_valid() for row in self.board for cell in row])
		cross_cell_valid = all(
			{d for r, c in unit for d in self.board[r][c].notes} == set(range(1, 10))
			for unit in self.iter_units()
		)
		return in_cell_valid and cross_cell_valid

	@classmethod
	def from_fixed_numbers(cls, partial: PartialSudoku) -> Self:
		retval = Sudoku()
		for r in range(9):
			for c in range(9):
				cell = partial.board[r][c]
				if cell.is_fixed_number():
					retval.fill_number(r, c, cell.number)
		return cls(retval, copy_board=False, validated=True)

SudokuSolverState = IntEnum("SudokuSolverState", ["SOLVED", "INVALID", "MULTI_ANSWER"], start=0)

class SudokuSolver(Protocol):
	@staticmethod
	def solve(sudoku: Sudoku) -> Tuple[SudokuSolverState, Optional[Sudoku]]: ...

class SudokuBackTrackingSolver:
	@staticmethod
	def pick_next_cell(sudoku: Sudoku) -> Optional[Tuple[SudokuIndex, SudokuIndex]]:
		# select cell with least notes(MRV)
		cells = ((len(sudoku.board[r][c].notes), r, c) \
			for r in range(9) for c in range(9) \
			if not sudoku.board[r][c].is_fixed_number())
		result = min(cells, key=lambda x: x[0], default=None)
		return result[1:] if result else None

	@staticmethod
	def solve(sudoku: Sudoku) -> Tuple[SudokuSolverState, Optional[Sudoku]]:
		pos = SudokuBackTrackingSolver.pick_next_cell(sudoku)
		if pos is None:
			return (SudokuSolverState.SOLVED, sudoku)
		r, c = pos
		cell = sudoku.board[r][c]

		solution: Optional[Sudoku] = None
		for num in sorted(cell.notes):
			sudoku_copy = deepcopy(sudoku)
			sudoku_copy.fill_number(r, c, num)
			if not sudoku_copy.validate():
				return (SudokuSolverState.INVALID, None)
			state, sol = SudokuBackTrackingSolver.solve(sudoku_copy)
			match state:
				case SudokuSolverState.INVALID:
					continue
				case SudokuSolverState.MULTI_ANSWER:
					return (SudokuSolverState.MULTI_ANSWER, None)
				case SudokuSolverState.SOLVED:
					if solution is not None:
						return (SudokuSolverState.MULTI_ANSWER, None)
					solution = sol
				case _:
					assert_never(state)

		if solution is None:
			return (SudokuSolverState.INVALID, None)
		return (SudokuSolverState.SOLVED, solution)

class SudokuDancingLinksSolver:
	class Node:
		def __init__(self) -> None:
			self.left: SudokuDancingLinksSolver.Node = self
			self.right: SudokuDancingLinksSolver.Node = self
			self.up: SudokuDancingLinksSolver.Node = self
			self.down: SudokuDancingLinksSolver.Node = self
			self.col: SudokuDancingLinksSolver.Column = None # type: ignore[assignment]
			self.row_id: Optional[Tuple[SudokuIndex, SudokuIndex, SudokuInt]] = None # (r,c,d)

	class Column(Node):
		def __init__(self) -> None:
			super().__init__()
			self.size = 0

	@staticmethod
	def link_lr(a: SudokuDancingLinksSolver.Node, b: SudokuDancingLinksSolver.Node) -> None:
		b.right = a.right
		b.left = a
		a.right.left = b
		a.right = b

	@staticmethod
	def link_ud(col: SudokuDancingLinksSolver.Column, n: SudokuDancingLinksSolver.Node) -> None:
		# insert at the bottom of the col
		n.down = col
		n.up = col.up
		col.up.down = n
		col.up = n
		n.col = col
		col.size += 1

	@staticmethod
	def cover(c: SudokuDancingLinksSolver.Column) -> None:
		# remove column c
		c.right.left = c.left
		c.left.right = c.right
		# remove rows that have nodes in column c
		i = c.down
		while i is not c:
			j = i.right
			while j is not i:
				j.down.up = j.up
				j.up.down = j.down
				j.col.size -= 1
				j = j.right
			i = i.down

	@staticmethod
	def uncover(c: SudokuDancingLinksSolver.Column) -> None:
		i = c.up
		while i is not c:
			j = i.left
			while j is not i:
				j.col.size += 1
				j.down.up = j
				j.up.down = j
				j = j.left
			i = i.up
		c.right.left = c
		c.left.right = c

	@staticmethod
	def solve(sudoku: Sudoku) -> Tuple[SudokuSolverState, Optional[Sudoku]]:
		root = SudokuDancingLinksSolver.Column()
		cols: Dict[str, SudokuDancingLinksSolver.Column] = {}

		def add_col(name: str) -> SudokuDancingLinksSolver.Column:
			col = SudokuDancingLinksSolver.Column()
			SudokuDancingLinksSolver.link_lr(root.left, col) # insert to the tail of the column
			cols[name] = col
			return col
		# cell restraint
		for r in range(9):
			for c in range(9):
				add_col(f"cell {r} {c}")
		# row restraint
		for r in range(9):
			for d in range(1, 10):
				add_col(f"row {r} {d}")
		# col restraint
		for c in range(9):
			for d in range(1, 10):
				add_col(f"col {c} {d}")
		# box restraint
		for b in range(9):
			for d in range(1, 10):
				add_col(f"box {b} {d}")

		def box_index(r: SudokuIndex, c: SudokuIndex) -> SudokuIndex:
			return (r // 3) * 3 + (c // 3)

		# construct rows
		row_lookup: Dict[Tuple[SudokuIndex, SudokuIndex, SudokuInt], SudokuDancingLinksSolver.Node] = {}
		for r in range(9):
			for c in range(9):
				allowed = sudoku.board[r][c].notes
				for d in sorted(allowed):
					c1 = cols[f"cell {r} {c}"]
					c2 = cols[f"row {r} {d}"]
					c3 = cols[f"col {c} {d}"]
					c4 = cols[f"box {box_index(r, c)} {d}"]
					col_list = (c1, c2, c3, c4)

					nodes = [SudokuDancingLinksSolver.Node() for _ in range(4)]
					for n in nodes:
						n.row_id = (r, c, d)
					# add to row
					for i in range(4):
						nodes[i].right = nodes[(i + 1) % 4]
						nodes[i].left = nodes[(i - 1) % 4]
					# add to columns
					for col, n in zip(col_list, nodes):
						SudokuDancingLinksSolver.link_ud(col, n)
					row_lookup[(r, c, d)] = nodes[0]

		solution_nodes: List[SudokuDancingLinksSolver.Node] = []

		# Step 1: Initiate precise coverage problem
		def select_row(rn: SudokuDancingLinksSolver.Node) -> None:
			solution_nodes.append(rn)
			j = rn
			while True:
				SudokuDancingLinksSolver.cover(j.col)
				j = j.right
				if j is rn:
					break
		for r in range(9):
			for c in range(9):
				cell = sudoku.board[r][c]
				if cell.is_fixed_number():
					d = cell.number
					rn = row_lookup.get((r, c, d))
					if rn is None:
						return (SudokuSolverState.INVALID, None)
					select_row(rn)

		# Step 2: Search for solution using Algorithm X
		def choose_column() -> Optional[SudokuDancingLinksSolver.Column]:
			# heuristic: choose the column with the smallest size to reduce branching factor
			best: Optional[SudokuDancingLinksSolver.Column] = None
			j = root.right
			while isinstance(j, SudokuDancingLinksSolver.Column) and j is not root:
				col = j
				if best is None or col.size < best.size:
					best = col
					if best.size == 0:
						break
				j = j.right
			return best

		def search() -> Tuple[SudokuSolverState, Optional[Sudoku]]:
			if root.right is root:
				# Step3: Reconstruct solution from selected rows
				result = Sudoku()
				for n in solution_nodes:
					if n.row_id is None:
						continue
					r, c, d = n.row_id
					result.board[r][c].number = d
				return (SudokuSolverState.SOLVED, result)

			c = choose_column()
			if c is None or c.size == 0:
				return (SudokuSolverState.INVALID, None)
			SudokuDancingLinksSolver.cover(c)
			try:
				solution: Optional[Sudoku] = None

				rn = c.down
				while rn is not c:
					solution_nodes.append(rn)
					j = rn.right
					while j is not rn:
						SudokuDancingLinksSolver.cover(j.col)
						j = j.right
					state, sol = search()
					solution_nodes.pop()
					j = rn.left
					while j is not rn:
						SudokuDancingLinksSolver.uncover(j.col)
						j = j.left
					match state:
						case SudokuSolverState.INVALID:
							pass
						case SudokuSolverState.MULTI_ANSWER:
							return (SudokuSolverState.MULTI_ANSWER, None)
						case SudokuSolverState.SOLVED:
							if solution is not None:
								return (SudokuSolverState.MULTI_ANSWER, None)
							solution = sol
					rn = rn.down

				if solution is None:
					return (SudokuSolverState.INVALID, None)
				return (SudokuSolverState.SOLVED, solution)
			finally:
				SudokuDancingLinksSolver.uncover(c)

		return search()

class SudokuHumanFriendlySolver:
	Step = Callable[[Sudoku], Tuple[bool, Optional[Sudoku]]]

	@staticmethod
	def choice(*steps: Step) -> Step:
		def run(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
			for step in steps:
				progress, solution = step(sudoku)
				if solution is None:
					return (False, None)
				if progress:
					return (True, solution)
			return (False, sudoku)
		return run

	@staticmethod
	def many(step: Step) -> Callable[[Sudoku], Optional[Sudoku]]:
		def run(sudoku: Sudoku) -> Optional[Sudoku]:
			while True:
				progress, solution = step(sudoku)
				if solution is None:
					return None
				if not progress:
					return sudoku
				sudoku = solution
		return run

	@staticmethod
	def identity(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		return (False, sudoku)

	@staticmethod
	def hidden_subset(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		"""
		Hidden Subset is a technique that identifies a subset of k digits that only appear in k cells within a unit (row, column, or box). If such a pattern is found, those k cells must contain those k digits, and any other candidates can be removed from those cells.
		"""
		# k can be at most 9 // 2
		for k in range(1, 5):
			for unit in Sudoku.iter_units():
				fixed = set(sudoku.board[r][c].number for r, c in unit if sudoku.board[r][c].is_fixed_number())

				# digit -> candidate cells in this unit
				digit_to_cells: Dict[SudokuInt, Set[Tuple[SudokuIndex, SudokuIndex]]] = {}
				for d in range(1, 10):
					if d in fixed:
						continue
					pos = set((r, c) for r, c in unit if not sudoku.board[r][c].is_fixed_number() and d in sudoku.board[r][c].notes)
					if pos:
						digit_to_cells[d] = pos

				digits = sorted(digit_to_cells.keys())
				if len(digits) < k:
					continue
				for ds in combinations(digits, k):
					union_cells: Set[Tuple[SudokuIndex, SudokuIndex]] = set()
					for d in ds:
						union_cells |= digit_to_cells[d]
					if len(union_cells) != k:
						continue # not hidden subset

					su = deepcopy(sudoku)
					allowed: Set[SudokuInt] = set(ds)
					changed = False
					for (r, c) in union_cells:
						cell = su.board[r][c]
						if cell.is_fixed_number():
							continue
						to_remove = cell.notes - allowed
						for num in sorted(to_remove):
							su.delete_note(r, c, num)
							changed = True
					if changed:
						if not su.validate():
							return (False, None)
						return (True, su)

		return (False, sudoku)

	@staticmethod
	def naked_subset(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		"""
		Naked Subset is a technique that identifies a subset of k cells within a unit (row, column, or box) that contain only k candidates in total. If such a pattern is found, those k candidates must be placed in those k cells, and any other candidates can be removed from those cells.
		"""
		# k can be at most 9 // 2
		# k == 1 (which is naked single) is already handled by Cell
		for k in range(2, 5):
			for unit in Sudoku.iter_units():
				cells = [(r, c) for r, c in unit if not sudoku.board[r][c].is_fixed_number() and 0 < len(sudoku.board[r][c].notes) <= k]
				if len(cells) < k:
					continue
				for cs in combinations(cells, k):
					union_digits: Set[SudokuInt] = set()
					for r, c in cs:
						union_digits |= sudoku.board[r][c].notes
					if len(union_digits) != k:
						continue # not naked subset

					su = deepcopy(sudoku)
					to_remove: Set[SudokuInt] = union_digits
					changed = False
					for (r, c) in unit:
						if (r, c) in cs:
							continue
						cell = su.board[r][c]
						if cell.is_fixed_number():
							continue
						to_remove_cell = cell.notes & to_remove
						for num in sorted(to_remove_cell):
							su.delete_note(r, c, num)
							changed = True
					if changed:
						if not su.validate():
							return (False, None)
						return (True, su)

		return (False, sudoku)

	@staticmethod
	def locked_candidate(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		"""
		Locked Candidate is a technique that identifies a candidate digit that is confined to a single row or column within a box. If such a pattern is found, it can be removed from the corresponding row or column outside of that box.
		"""
		for br in range(3):
			for bc in range(3):
				box = [(3 * br + r, 3 * bc + c) for r in range(3) for c in range(3)]
				box_rs = {r for r, _ in box}
				box_cs = {c for _, c in box}

				for d in range(1, 10):
					pos = [(r, c) for (r, c) in box if (not sudoku.board[r][c].is_fixed_number()) and (d in sudoku.board[r][c].notes)]
					if len(pos) < 2:
						continue # not locked candidate, one number in at least 2 candidate cells needed to form a pattern

					rows = {r for r, _ in pos}
					cols = {c for _, c in pos}

					# pointing -> row
					if len(rows) == 1:
						target_r = next(iter(rows))
						su = deepcopy(sudoku)
						changed = False
						for c in range(9):
							if c in box_cs:
								continue # not notes in box
							cell = su.board[target_r][c]
							if cell.is_fixed_number():
								continue
							if d in cell.notes:
								su.delete_note(target_r, c, d)
								changed = True
						if changed:
							if not su.validate():
								return (False, None)
							return (True, su)

					# pointing -> col
					if len(cols) == 1:
						target_c = next(iter(cols))
						su = deepcopy(sudoku)
						changed = False
						for r in range(9):
							if r in box_rs:
								continue
							cell = su.board[r][target_c]
							if cell.is_fixed_number():
								continue
							if cell.value[d - 1]:
								su.delete_note(r, target_c, d)
								changed = True
						if changed:
							if not su.validate():
								return (False, None)
							return (True, su)

		return (False, sudoku)

	@staticmethod
	def unfinned_fish(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		"""
		Unfinned Fish is a generalization of the X-Wing, Swordfish and Jellyfish techniques. It identifies a candidate digit that forms a pattern across multiple rows and columns, where the candidate is confined to a specific set of rows and columns. If such a pattern is found, the candidate can be removed from the corresponding rows or columns outside of that pattern.
		"""
		def is_candidate(r: int, c: int, d: int) -> bool:
			cell = sudoku.board[r][c]
			return (not cell.is_fixed_number()) and d in cell.notes

		for k in range(2, 5):
			# Row-based unfinned fish
			for d in range(1, 10):
				row_to_cols: Dict[int, List[int]] = {}
				for r in range(9):
					cols = [c for c in range(9) if is_candidate(r, c, d)]
					if 2 <= len(cols) <= k:
						row_to_cols[r] = cols

				rows = sorted(row_to_cols.keys())
				if len(rows) < k:
					continue

				for rs in combinations(rows, k):
					union_cols = set()
					for r in rs:
						union_cols |= set(row_to_cols[r])
					if len(union_cols) != k:
						continue

					su = deepcopy(sudoku)
					changed = False
					for r in range(9):
						if r in rs:
							continue
						for c in union_cols:
							cell = su.board[r][c]
							if cell.is_fixed_number():
								continue
							if cell.value[d - 1]:
								su.delete_note(r, c, d)
								changed = True

					if changed:
						if not su.validate():
							return (False, None)
						return (True, su)

			# Column-based unfinned fish
			for d in range(1, 10):
				col_to_rows: Dict[int, List[int]] = {}
				for c in range(9):
					rows = [r for r in range(9) if is_candidate(r, c, d)]
					if 2 <= len(rows) <= k:
						col_to_rows[c] = rows

				cols = sorted(col_to_rows.keys())
				if len(cols) < k:
					continue

				for cs in combinations(cols, k):
					union_rows = set()
					for c in cs:
						union_rows |= set(col_to_rows[c])
					if len(union_rows) != k:
						continue

					su = deepcopy(sudoku)
					changed = False
					for c in range(9):
						if c in cs:
							continue
						for r in union_rows:
							cell = su.board[r][c]
							if cell.is_fixed_number():
								continue
							if cell.value[d - 1]:
								su.delete_note(r, c, d)
								changed = True

					if changed:
						if not su.validate():
							return (False, None)
						return (True, su)

		return (False, sudoku)

	@staticmethod
	def basic_x_chain(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		"""
		Basic X-Chain is a technique that identifies a 2nd-order AIC(alternative inference chain) of a single digit. It's a generalization of the Skyscraper, Two-String-Like and Crane techniques. If such a pattern is found, the candidate can be removed from the corresponding cells that see both ends of the chain.
		"""
		return SudokuHumanFriendlySolver.x_chain(sudoku, k=2)

	@staticmethod
	def empty_rectangle(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		raise NotImplementedError("Empty Rectangle technique is not implemented yet.")

	@staticmethod
	def bug_plus_1(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		"""
		BUG+1 is a uniqueness technique that identifies a pattern where there is only one cell with more than two candidates and all other cells have exactly two candidates. If such a pattern is found, the candidate in the cell that is not part of any pair should be placed in that cell to prevent forming a BUG(binary universal grave).
		"""
		cells_not_fixed = [(r, c) for r in range(9) for c in range(9) if not sudoku.board[r][c].is_fixed_number()]
		group_key = lambda rc: len(sudoku.board[rc[0]][rc[1]].notes)
		nums_to_cells: Dict[int, List[Tuple[SudokuIndex, SudokuIndex]]] = {key: list(group) for key, group in groupby(sorted(cells_not_fixed, key=group_key), key=group_key)}
		if not (nums_to_cells.get(2) and nums_to_cells.get(3) and len(nums_to_cells[3]) == 1):
			return (False, sudoku)
		r, c = nums_to_cells[3][0]
		candidates = sudoku.board[r][c].notes

		counts = Counter(
			d
			for (rr, cc) in nums_to_cells[2]
			for d in sudoku.board[rr][cc].notes
			if d in candidates
		)
		unique_candidate = max(counts, key=lambda d: counts[d])

		sudoku_copy = deepcopy(sudoku)
		sudoku_copy.fill_number(r, c, unique_candidate)
		return (True, sudoku_copy)

	@staticmethod
	def finned_fish(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		raise NotImplementedError("Finned Fish technique is not implemented yet.")

	@staticmethod
	def x_chain(sudoku: Sudoku, k: int = 0) -> Tuple[bool, Optional[Sudoku]]:
		"""
		X-Chain is a technique that identifies an AIC of a single digit. If such a pattern is found, the candidate can be removed from the corresponding cells that see both ends of the chain.
		"""
		def is_candidate(r: int, c: int, d: int) -> bool:
			cell = sudoku.board[r][c]
			return (not cell.is_fixed_number()) and d in cell.notes

		def peers(r: int, c: int) -> Set[Tuple[int, int]]:
			ps: Set[Tuple[int, int]] = set()
			# row
			for cc in range(9):
				if cc != c:
					ps.add((r, cc))
			# col
			for rr in range(9):
				if rr != r:
					ps.add((rr, c))
			# box
			br = (r // 3) * 3
			bc = (c // 3) * 3
			for rr in range(br, br + 3):
				for cc in range(bc, bc + 3):
					if rr != r or cc != c:
						ps.add((rr, cc))
			return ps

		def box_index(r: int, c: int) -> int:
			return (r // 3) * 3 + (c // 3)

		for d in range(1, 10):
			# collect all candidates of digit d
			cands: List[Tuple[int, int]] = [(r, c) for r in range(9) for c in range(9) if is_candidate(r, c, d)]
			if len(cands) < 4:
				continue

			# weak neighbors: any other candidate of d that sees this one
			row_map: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(9)}
			col_map: Dict[int, List[Tuple[int, int]]] = {c: [] for c in range(9)}
			box_map: Dict[int, List[Tuple[int, int]]] = {b: [] for b in range(9)}
			for (r, c) in cands:
				row_map[r].append((r, c))
				col_map[c].append((r, c))
				box_map[box_index(r, c)].append((r, c))

			weak_neighbors: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
			for (r, c) in cands:
				nb = set(row_map[r]) | set(col_map[c]) | set(box_map[box_index(r, c)])
				nb.discard((r, c))
				weak_neighbors[(r, c)] = nb

			# strong neighbors: conjugate pairs (exactly 2 candidates in a unit)
			strong_neighbors: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {(r, c): set() for (r, c) in cands}
			for unit in Sudoku.iter_units():
				pos = [(r, c) for (r, c) in unit if is_candidate(r, c, d)]
				if len(pos) == 2:
					a, b = pos
					strong_neighbors[a].add(b)
					strong_neighbors[b].add(a)

			# BFS alternating edges, starting with STRONG; endpoints are reached after an odd number of edges (last edge STRONG)
			# state: (node, need_strong_next)
			for start in cands:
				q = deque([(start, True, 0)])  # need_strong_next=True at start => first edge must be strong
				visited: Set[Tuple[Tuple[int, int], bool]] = {(start, True)}

				while q:
					cur, need_strong, dist = q.popleft()
					nbrs = strong_neighbors[cur] if need_strong else weak_neighbors[cur]

					for nb in nbrs:
						next_need_strong = not need_strong
						nd = dist + 1
						state = (nb, next_need_strong)
						if state in visited:
							continue
						visited.add(state)
						q.append((nb, next_need_strong, nd))

						# last edge was strong <=> next expected is weak
						if next_need_strong is False and (k == 0 and nd >= 3 or k > 0 and nd == 2 * k - 1):
							end = nb

							targets = peers(*start) & peers(*end)
							if not targets:
								continue

							# check whether there is anything to eliminate before deepcopy
							elim_cells = [(rr, cc) for (rr, cc) in targets if is_candidate(rr, cc, d)]
							if not elim_cells:
								continue

							su = deepcopy(sudoku)
							changed = False
							for (rr, cc) in sorted(elim_cells):
								# remove digit d from cells that see both ends
								su.delete_note(rr, cc, d)
								changed = True

							if changed:
								if not su.validate():
									return (False, None)
								return (True, su)

		return (False, sudoku)

	@staticmethod
	def unique_rectangle(sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		raise NotImplementedError("Unique Rectangle technique is not implemented yet.")

	@staticmethod
	def simple_technique() -> Step:
		return SudokuHumanFriendlySolver.choice(
			SudokuHumanFriendlySolver.identity,
			SudokuHumanFriendlySolver.hidden_subset,
			SudokuHumanFriendlySolver.naked_subset,
			SudokuHumanFriendlySolver.locked_candidate,
		)

	@staticmethod
	def medium_technique() -> Step:
		return SudokuHumanFriendlySolver.choice(
			SudokuHumanFriendlySolver.simple_technique(),
			SudokuHumanFriendlySolver.unfinned_fish,
			SudokuHumanFriendlySolver.basic_x_chain,
			SudokuHumanFriendlySolver.empty_rectangle,
		)

	@staticmethod
	def hard_technique() -> Step:
		return SudokuHumanFriendlySolver.choice(
			SudokuHumanFriendlySolver.medium_technique(),
			SudokuHumanFriendlySolver.finned_fish,
			SudokuHumanFriendlySolver.x_chain,
			SudokuHumanFriendlySolver.unique_rectangle,
			SudokuHumanFriendlySolver.bug_plus_1,
		)

	@staticmethod
	def uniqueness_technique() -> Step:
		return SudokuHumanFriendlySolver.choice(
			SudokuHumanFriendlySolver.bug_plus_1,
			SudokuHumanFriendlySolver.unique_rectangle,
		)

	@staticmethod
	def next_technique() -> Step:
		return SudokuHumanFriendlySolver.choice(
			SudokuHumanFriendlySolver.hidden_subset,
			SudokuHumanFriendlySolver.naked_subset,
			SudokuHumanFriendlySolver.locked_candidate,
		)

	@staticmethod
	def solve_step_by_step(sudoku: Sudoku) -> Generator[Tuple[bool, Optional[Sudoku]], Optional[Step], Optional[Sudoku]]:
		progress = True # yield true first, so that we can write stuffs like: for progress, solution in solve_step_by_step(...): if not progress: break
		solution = sudoku
		while True:
			next_step_func = yield (progress, solution)
			if next_step_func is None:
				next_step_func = SudokuHumanFriendlySolver.next_technique()
			progress, solution = next_step_func(solution)
			if solution is None:
				return None
			if all(cell.is_fixed_number() for row in solution.board for cell in row):
				return solution

	@staticmethod
	def solve(sudoku: Sudoku) -> Tuple[SudokuSolverState, Optional[Sudoku]]:
		for _ in SudokuHumanFriendlySolver.solve_step_by_step(sudoku): pass
		try:
			next(SudokuHumanFriendlySolver.solve_step_by_step(sudoku))
		except StopIteration as e:
			return (SudokuSolverState.SOLVED, e.value)
		return (SudokuSolverState.INVALID, None)

class SudokuGenerator:
	_seed = None

	def generate(self, difficulty: Literal["simple", "medium", "hard", "impossible"] = "simple") -> Sudoku:
		import random
		random.seed(self._seed)

		def generate_unique_puzzle() -> Sudoku:
			puzzle = Sudoku()
			positions = [(r, c) for r in range(9) for c in range(9)]
			random.shuffle(positions)
			for r, c in positions[:36]:
				if puzzle.board[r][c].is_fixed_number():
					continue
				candidates = puzzle.board[r][c].notes
				puzzle_old = deepcopy(puzzle)
				puzzle.fill_number(r, c, random.choice(list(candidates)))
				if not puzzle.validate():
					puzzle = puzzle_old
					continue
				state, _ = SudokuDancingLinksSolver.solve(puzzle)
				if state == SudokuSolverState.SOLVED:
					return puzzle
			return self.generate(difficulty) # Too many givens, regenerate

		puzzle = generate_unique_puzzle()

		def further_remove_cells(puzzle: Sudoku) -> Sudoku:
			positions = [(r, c) for r in range(9) for c in range(9) if puzzle.board[r][c].is_fixed_number()]
			random.shuffle(positions)
			for r, c in positions:
				puzzle_new = deepcopy(puzzle)
				puzzle_new.board[r][c].value = [True] * 9 # Temporarily remove the number
				puzzle_new.rebuild_notes() # Rebuild notes to reflect the removal
				if puzzle == puzzle_new:
					continue # More removals may help, but I don't know how to handle it
				state, _ = SudokuDancingLinksSolver.solve(puzzle_new)
				if state == SudokuSolverState.SOLVED:
					puzzle = puzzle_new
			return puzzle

		puzzle = further_remove_cells(puzzle)

		match difficulty:
			case "simple":
				solver = SudokuHumanFriendlySolver.identity
			case "medium":
				solver = SudokuHumanFriendlySolver.simple_technique()
			case "hard":
				solver = SudokuHumanFriendlySolver.medium_technique()
			case "impossible":
				solver = SudokuHumanFriendlySolver.hard_technique()
			case _:
				assert_never(difficulty)
		while True:
			progress, solution = solver(puzzle)
			if not progress:
				break
			assert solution
			puzzle = solution

		if all(cell.is_fixed_number() for row in puzzle.board for cell in row):
			return self.generate(difficulty) # Regenerate if the puzzle is already solved
		return puzzle

class SudokuInteractive:
	def __init__(self, sudoku: PartialSudoku):
		self._sudoku = sudoku
		self._history: List[PartialSudoku] = []
		self._history_commands: List[str] = []

	def run(self) -> None:
		def help() -> str:
			return """Commands:
- w(rite){number} r(ow){row} c(ol){col}: write number at row, col (1-indexed)
- n(ote){number} r(ow){row} c(ol){col}: note number at row, col (1-indexed)
- s(how){number}: highlight all cells that can be number
- s(how) r(ow){row} c(ol){col}: highlight cell at row, col (1-indexed)
- u(ndo): undo last action
- h(elp): show this message
- q(uit): exit the program
"""
		while True:
			print("\033[H", end="") # Move cursor to top-left corner
			self._sudoku.draw()
			print("\033[50;1H", end="") # Move cursor out of the board
			print("Type 'q' or 'quit' to exit. 'h' or 'help' for commands.")
			cmd = input("> ").strip().lower()
			match cmd.split():
				case ["q" | "quit"]:
					print("Exiting...")
					break
				case ["h" | "help"]:
					print(help())
				case ["w" | "write", num, "r" | "row", row, "c" | "col", col]:
					try:
						self._history.append(deepcopy(self._sudoku))
						self._history_commands.append(f"write {num} row {row} col {col}")
						self._sudoku.board[int(row) - 1][int(col) - 1].number = int(num)
					except (ValueError, IndexError):
						print("Invalid command. Please try again.")
				case ["n" | "note", num, "r" | "row", row, "c" | "col", col]:
					try:
						self._history.append(deepcopy(self._sudoku))
						self._history_commands.append(f"note {num} row {row} col {col}")
						self._sudoku.board[int(row) - 1][int(col) - 1].toggle_note(int(num))
					except (ValueError, IndexError):
						print("Invalid command. Please try again.")
				case ["u" | "undo"]:
					if self._history:
						self._sudoku = self._history.pop()
					else:
						print("No history to undo.")
				case ["s" | "show", num]:
					raise NotImplementedError("Show command not implemented yet.")
				case ["s" | "show", "r" | "row", row, "c" | "col", col]:
					raise NotImplementedError("Show command not implemented yet.")
				case ["\033[A"]:
					raise NotImplementedError("Up arrow command not implemented yet.")
				case _:
					print("Unknown command.")
					print(help())

class Test:
	board = "12.45.7.....4.67.9.2.4.67..1..4.6..9.......8...3.......2..567.9...4567.9.2.45....1..4..7.....4.6789...4.678.....5....1..4.6....2.........3..67.9...4.67.9..34...8..2.45.......4.6.89..3.........4.6..9...4.6.........7..1...........456..9.2.45..8...34..7......5............91..4.6.8.1..4.6...1..4.6.8...3...7...2.......1.3.............8...34..7.....4..7...2...............91..4.......3.5.7..1...5.7.......6........6....2.......1..........3............7......5.......4............8.........9........91............5.......4.6....2..........4.6..........8...3............7...234..7....34.67...2.4.67..1......8.....5....1......8..2...6..9...4.6..9.2.4......2.4........4.6.8..2.4.6.8.......7....3..............9.2..56...1..456...12.45...."
	solved = "......7.......6....2.......1...............8...3..............9...4.........5....1................9...4.........5.........6....2.........3............7.........8.....5...........8...3..............9...4...........7..1.............6....2..........4.........5............9.....6...1...............8.......7...2.........3.............8...3............7...2...............9...4.........5....1.............6........6....2.......1..........3............7......5.......4............8.........9........91............5.......4......2............6..........8...3............7....3............7.......6..........8.....5....1.........2...............9...4......2..........4............8.......7....3..............9.....6.......5....1........"
	multi = "12.............7......5....12..............8.........9.....6......4.......3......12............6......4.....12.........3............7......5...........8.........9........9..3.............8.....5.......4..........6.........7...2.......1..........3......1................9.......8.......7......5.......4..........6....2............6.......5..........7..........9.2..........4.....1..........3.............8....4............8..2............6...1..........3..............9....5..........7.........8..2............6......4.............91..........3............7......5........5............9..3............7.......6..........8..2.......1...........4...........7.....4.....1..........3..........5.....2..............8.........9.....6..."
	very_difficult = ".23.....9.23.5.7.9....5...9123...7.9.....6...1.3...789...4.....123.5..8.1.3.5..8..234.6....23..67.........8.1234..7..1.34.........5....12...6...........91.3..6...1.........23.56..9...456..9.234....9..34...8...34...89......7...23.5..8...3.56.8.......7....3.5..89....5...91.3.5....1.3.5..8......6...1.......9...4......2.........34...8...3....8.1..........34.............9.2...........5.........6.........78....4.6.89....56.89.2.......1..45..........7..1..4...8...3......1......8.1......89.2...6.8912...6.89.....6..91.34567.91.345....1.34..7.912...6..9123.5.7..1.34567.9.....6..91....6..9..3......1..4567.9.2.......1..4..7.9.......8.1...5.7..1..4567.9....5.......4...........7.........8.1.3......1.3.....912...6..9123......1.3..6..9"

	def test_serialize_and_equal(self):
		su = PartialSudoku.deserialize(self.board)
		assert su.validate_full()
		assert su.board[0][4].number == 8
		assert su.board[1][3].number == 5
		assert su.board[1][5].number == 2
		assert su.board[2][2].number == 3
		assert su.board[2][6].number == 1
		assert su.board[3][1].number == 5
		assert su.board[3][7].number == 2
		assert su.board[4][0].number == 8
		assert su.board[4][4].number == 9
		assert su.board[4][8].number == 6
		assert su.board[5][0].number == 6
		assert su.board[5][3].number == 3
		assert su.board[5][4].number == 7
		assert su.board[5][5].number == 5
		assert su.board[5][8].number == 9
		assert su.board[6][1].number == 1
		assert su.board[6][2].number == 5
		assert su.board[6][4].number == 2
		assert su.board[6][6].number == 8
		assert su.board[6][7].number == 3
		assert su.board[7][4].number == 5
		assert su.board[8][3].number == 7
		assert su.board[8][4].number == 3
		assert su.board[8][5].number == 9
		assert su.serialize() == self.board
		su_f = Sudoku.deserialize(self.board)
		assert su_f.serialize() == self.board
		assert su == su_f

	def test_fill_and_validate(self):
		su = PartialSudoku.deserialize(self.board)
		assert su.validate_full()
		su_f = Sudoku(su, validated=True)
		su.fill_number(0, 0, 1)
		assert su.board[0][0].number == 1
		assert su.validate()
		assert not su.validate_full()
		su.fill_number(1, 1, 2)
		assert not su.validate()
		su_f.fill_number(0, 0, 1)
		assert su_f.validate_full()
		su_f.fill_number(1, 1, 2)
		assert not su_f.validate()

	def test_backtracking_solve(self):
		su = Sudoku.deserialize(self.board)
		state, solution = SudokuBackTrackingSolver().solve(su)
		assert state == SudokuSolverState.SOLVED
		assert solution == PartialSudoku.deserialize(self.solved)
		multi = Sudoku.deserialize(self.multi)
		state, solution = SudokuBackTrackingSolver().solve(multi)
		assert state == SudokuSolverState.MULTI_ANSWER

	def test_dancinglinks_solve(self):
		su = Sudoku.deserialize(self.board)
		state, solution = SudokuDancingLinksSolver().solve(su)
		assert state == SudokuSolverState.SOLVED
		assert solution == PartialSudoku.deserialize(self.solved)
		multi = Sudoku.deserialize(self.multi)
		state, solution = SudokuDancingLinksSolver().solve(multi)
		assert state == SudokuSolverState.MULTI_ANSWER

	def test_puzzle_generation(self):
		puzzle = SudokuGenerator().generate()
		state, _ = SudokuDancingLinksSolver().solve(puzzle)
		assert state == SudokuSolverState.SOLVED

	def test_solve_hidden_subset(self):
		def n(*args: int) -> str:
			return "".join(str(i) if i in args else "." for i in range(1, 10))
		su = Sudoku.deserialize("".join(map(lambda r: "".join(map(lambda l: n(*l) if isinstance(l, list) else n(l), r)), [
			[[3,4,5,9], [1,3,5,9], [1,3,4,5], [2,3,9], [3,7,9], [3,4,7], [2,5,6,7,9], 8, [2,6,7,9]],
			[7, 8, 6, [2,9], 1, 5, [2,9], 4, 3],
			[[3,4,5,9], [3,5,9], 2, [3,8,9], 6, [3,4,7,8], [5,7,9], 1, [7,9]],
			[[3,4,5,8], [1,3,5], [1,3,4,5,8], 7, 2, 9, [3,4], 6, [5,8]],
			[[2,3,4,6,8,9], [2,3,6,7,9], [3,4,8], [1,3,5], [3,5], [1,3,6], [3,4], [2,7,9], [2,8,9]],
			[[2,3,5,6,9], [2,3,5,6,7,9], [3,5], 4, 8, [3,6], 1, [2,7,9], [2,5,9]],
			[[2,3,8], 4, [3,8], 6, [3,7,9], [1,3,7,8], [2,8,9], 5, [1,2,9]],
			[[5,6,8], [5,6], 9, [1,5,8], 4, 2, [6,7,8], 3, [1,6,7]],
			[1, [2,3,5,6], 7, [3,5,8,9], [3,5,9], [3,8], [2,6,8,9], [2,9], 4],
		])))
		progress, so = SudokuHumanFriendlySolver().hidden_subset(su)
		assert progress and so != su

	def test_solve_naked_subset(self):
		def n(*args: int) -> str:
			return "".join(str(i) if i in args else "." for i in range(1, 10))
		su = Sudoku.deserialize("".join(map(lambda r: "".join(map(lambda l: n(*l) if isinstance(l, list) else n(l), r)), [
			[[3,7], [5,6], [2,6,7], [3,4], 8, 1, 9, [5,7], [2,4]],
			[[2,3,5,7,9], 1, [2,7,9], [3,4], 6, [2,7], [2,3,4], [5,7], 8],
			[[2,3,7], 8, 4, 9, 5, [2,7], [2,3], 6, 1],
			[[8,9], [4,6], [8,9], 5, 2, 3, [4,7], 1, [6,7]],
			[[2,6], 3, 5, 7, 1, 4, [2,6], 8, 9],
			[[1,2,4], 7, [1,2], 8, 9, 6, 5, 3, [2,4]],
			[[1,8], 9, [6,7], 2, 3, 5, [1,8], 4, [6,7]],
			[[6,7], 2, 3, 1, 4, 8, [6,7], 9, 5],
			[[4,5], [4,5], [1,8], 6, 7, 9, [1,8], 2, 3],
		])))
		progress, so = SudokuHumanFriendlySolver().naked_subset(su)
		assert progress and so != su

	def test_solve_locked_candidate(self):
		def n(*args: int) -> str:
			return "".join(str(i) if i in args else "." for i in range(1, 10))
		su = Sudoku.deserialize("".join(map(lambda r: "".join(map(lambda l: n(*l) if isinstance(l, list) else n(l), r)), [
			[[5,6,7,9], 2, [5,6,7,9], 1, [7,9], 4, 3, 8, [5,6,7,9]],
			[[3,5,7,9], [3,4,5,9], 8, 2, 6, [3,5], 1, [4,5], [5,7,9]],
			[1, [3,4,5,9], [5,6,7,9], 8, [7,9], [3,5], [4,9], 2, [5,6,7,9]],
			[[3,9], 7, 4, [3,9], 8, 6, 5, 1, 2],
			[8, [3,5], 1, [3,5], 2, 7, 6, 9, 4],
			[[5,9], 6, 2, [5,9], 4, 1, 7, 3, 8],
			[[5,6,7,9], [5,9], [5,6,7,9], [4,6], 1, 2, 8, [4,5], 3],
			[4, 8, 3, 7, 5, 9, 2, 6, 1],
			[2, 1, [5,6], [4,6], 3, 8, [4,9], 7, [5,9]],
		])))
		progress, so = SudokuHumanFriendlySolver().locked_candidate(su)
		assert progress and so != su

	def test_solve_unfinned_fish(self):
		def n(*args: int) -> str:
			return "".join(str(i) if i in args else "." for i in range(1, 10))
		su = Sudoku.deserialize("".join(map(lambda r: "".join(map(lambda l: n(*l) if isinstance(l, list) else n(l), r)), [
			[[2,4], 7, 1, 3, 6, [2,4], 9, 5, 8],
			[[2,3,4,8], 6, [2,3,4,8], 9, [2,5], [4,5], [2,3], 1, 7],
			[[2,3,5], 9, [2,3,5], 1, 8, 7, 6, [2,3], 4],
			[[2,5], 3, 6, 7, 4, 1, 8, 9, [2,5]],
			[7, [4,8], [4,8], [2,5], [2,5], 9, 1, 6, 3],
			[9, 1, [2,5], 8, 3, 6, 4, 7, [2,5]],
			[1, 5, [3,8], 4, 9, [2,3,8], 7, [2,3,8], 6],
			[[3,4,8], [2,4,8], 7, 6, 1, [2,3,5,8], [2,3,5], [2,3,8], 9],
			[6, [2,8], 9, [2,5], 7, [2,3,5,8], [2,3,5], 4, 1],
		])))
		progress, so = SudokuHumanFriendlySolver().unfinned_fish(su)
		assert progress and so != su

	def test_solve_bug_plus_1(self):
		def n(*args: int) -> str:
			return "".join(str(i) if i in args else "." for i in range(1, 10))
		su = Sudoku.deserialize("".join(map(lambda r: "".join(map(lambda l: n(*l) if isinstance(l, list) else n(l), r)), [
			[[2,6], 7, 8, [1,6,9], [2,9], [1,9], 5, 3, 4],
			[[4,6], 5, 1, [4,6], 3, 8, 9, 7, 2],
			[[2,4], 9, 3, 7, [2,5], [4,5], 1, 6, 8],
			[5, 1, 4, [2,9], 6, [2,9], 7, 8, 3],
			[3, 8, 2, [1,5], 7, [1,5], 4, 9, 6],
			[9, 6, 7, 8, 4, 3, 2, 1, 5],
			[8, 2, 6, [5,9], [5,9], 7, 3, 4, 1],
			[7, 4, 5, 3, 1, 6, 8, 2, 9],
			[1, 3, 9, [2,4], 8, [2,4], 6, 5, 7],
		])))
		progress, so = SudokuHumanFriendlySolver().bug_plus_1(su)
		assert progress and so != su

	def test_solve_basic_x_chain(self):
		def n(*args: int) -> str:
			return "".join(str(i) if i in args else "." for i in range(1, 10))
		su = Sudoku.deserialize("".join(map(lambda r: "".join(map(lambda l: n(*l) if isinstance(l, list) else n(l), r)), [
			[[6,8],[2,6,8], 4, 1, 3, [2,8], 9, 7, 5],
			[9, [7,8], 1, [5,7], [5,7,8], 4, 6, 2, 3],
			[[3,5,7], [2,3,7], [5,7], [2,9], 6, [2,7,9], 1, 8, 4],
			[[4,6,7], 1, 2, [6,9], [4,7,9], 3, [4,7], 5, 8],
			[[4,5,6,7,8], [4,6,7,8], 9, [2,6], [4,5,7,8], [2,7,8], [4,7], 3, 1],
			[[3,4,8], [3,4,8], [5,7], [5,7], [4,8], 1, 2, 9, 6],
			[[4,7], [4,7], 3, 8, 2, 6, 5, 1, 9],
			[2, 9, 8, 4, 1, 5, 3, 6, 7],
			[1, 5, 6, 3, [7,9], [7,9], 8, 4, 2],
		])))
		progress, so = SudokuHumanFriendlySolver().basic_x_chain(su)
		assert progress and so != su

	def test_solve_x_chain(self):
		def n(*args: int) -> str:
			return "".join(str(i) if i in args else "." for i in range(1, 10))
		su = Sudoku.deserialize("".join(map(lambda r: "".join(map(lambda l: n(*l) if isinstance(l, list) else n(l), r)), [
			[[3,8], 7, 2, 5, [3,8], 6, 1, 9, 4],
			[6, 4, 5, 1, 2, 9, 8, 3, 7],
			[9, 1, [3,8], [4,7], [3,4,8], [3,7], 2, 5, 6],
			[[7,8], 2, 6, [7,8], 9, 1, 5, 4, 3],
			[[3,5], 9, [3,4], 6, [3,4,5], 2, 7, 8, 1],
			[1, [3,5,8], [3,4,7,8], [4,7,8], [3,4,5], [3,7], 9, 6, 2],
			[4, 6, 1, 9, 7, 5, 3, 2, 8],
			[[3,5,7], [3,5], [3,7,9], 2, 6, 8, 4, 1, [5,9]],
			[2, [5,8], [8,9], 3, 1, 4, 6, 7, [5,9]],
		])))
		progress, so = SudokuHumanFriendlySolver().x_chain(su)
		assert progress and so != su

if __name__ == "__main__":
	t = Test()
	for name, func in Test.__dict__.items():
		if isinstance(func, Callable) and not name.startswith("_"):
			print(f"Running {name}...")
			func(t)
