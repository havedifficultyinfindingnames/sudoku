from typing import *
from copy import *
from enum import IntEnum
from itertools import combinations
from dataclasses import dataclass, field

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
		if not self.validate():
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
		return all([cell.is_valid() for row in self.board for cell in row])

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
	def solve(self, sudoku: Sudoku) -> Tuple[SudokuSolverState, Optional[Sudoku]]: ...

class SudokuBackTrackingSolver:
	@staticmethod
	def pick_next_cell(sudoku: Sudoku) -> Optional[Tuple[SudokuIndex, SudokuIndex]]:
		# select cell with least notes(MRV)
		cells = ((len(sudoku.board[r][c].notes), r, c) \
			for r in range(9) for c in range(9) \
			if not sudoku.board[r][c].is_fixed_number())
		result = min(cells, key=lambda x: x[0], default=None)
		return result[1:] if result else None

	def solve(self, sudoku: Sudoku) -> Tuple[SudokuSolverState, Optional[Sudoku]]:
		pos = self.pick_next_cell(sudoku)
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
			state, sol = self.solve(sudoku_copy)
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
			self.col: SudokuDancingLinksSolver.Column = None  # type: ignore[assignment]
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

	def solve(self, sudoku: Sudoku) -> Tuple[SudokuSolverState, Optional[Sudoku]]:
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
						self.link_ud(col, n)
					row_lookup[(r, c, d)] = nodes[0]

		solution_nodes: List[SudokuDancingLinksSolver.Node] = []

		# Step 1: Initiate precise coverage problem
		def select_row(rn: SudokuDancingLinksSolver.Node) -> None:
			solution_nodes.append(rn)
			j = rn
			while True:
				self.cover(j.col)
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
			self.cover(c)
			try:
				solution: Optional[Sudoku] = None

				rn = c.down
				while rn is not c:
					solution_nodes.append(rn)
					j = rn.right
					while j is not rn:
						self.cover(j.col)
						j = j.right
					state, sol = search()
					solution_nodes.pop()
					j = rn.left
					while j is not rn:
						self.uncover(j.col)
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
				self.uncover(c)

		return search()

class SudokuHumanFriendlySolver:
	def _iter_units(self) -> Generator[List[Tuple[SudokuIndex, SudokuIndex]]]:
		# row units
		for r in range(9):
			yield [(r, c) for c in range(9)]
		# col units
		for c in range(9):
			yield [(r, c) for r in range(9)]
		# box units
		for br in range(3):
			for bc in range(3):
				yield [(br * 3 + dr, bc * 3 + dc) for dr in range(3) for dc in range(3)]

	def hidden_subset(self, sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		"""
		Hidden Subset is a technique that identifies a subset of k digits that only appear in k cells within a unit (row, column, or box). If such a subset is found, those k cells must contain those k digits, and any other candidates can be removed from those cells.
		"""
		# k can be at most 9 // 2
		for k in range(1, 4):
			for unit in self._iter_units():
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
						# not hidden subset
						continue

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

	def a_very_basic_solve_process(self, sudoku: Sudoku) -> Tuple[bool, Optional[Sudoku]]:
		raise NotImplementedError("Human-friendly solver not implemented yet.")

	def choose_next_technic(self) -> Callable[[SudokuHumanFriendlySolver, Sudoku], Tuple[bool, Optional[Sudoku]]]:
		raise NotImplementedError("Human-friendly solver not implemented yet.")
		return SudokuHumanFriendlySolver.a_very_basic_solve_process

	def solve_step_by_step(self, sudoku: Sudoku) -> Generator[
		Tuple[bool, Optional[Sudoku]],
		Optional[Callable[[SudokuHumanFriendlySolver, Sudoku], Tuple[bool, Optional[Sudoku]]]],
		Optional[Sudoku]]:
		progress = False
		step = sudoku
		while True:
			next_step_func = yield (progress, step)
			if next_step_func is None:
				next_step_func = self.choose_next_technic()
			progress, step = next_step_func(self, step)
			if step is None:
				return None
			if all(cell.is_fixed_number() for row in step.board for cell in row):
				return step
	def solve(self, sudoku: Sudoku) -> Optional[Sudoku]:
		for _ in self.solve_step_by_step(sudoku): pass
		try:
			next(self.solve_step_by_step(sudoku))
		except StopIteration as e:
			return e.value

class SudokuGenerator:
	@staticmethod
	def generate() -> Sudoku:
		import random
		dlx = SudokuDancingLinksSolver()

		def generate_unique_puzzle() -> Sudoku:
			puzzle = Sudoku()
			positions = [(r, c) for r in range(9) for c in range(9)]
			random.shuffle(positions)
			for r, c in positions[:30]:
				if puzzle.board[r][c].is_fixed_number():
					continue
				candidates = puzzle.board[r][c].notes
				puzzle_old = deepcopy(puzzle)
				puzzle.fill_number(r, c, random.choice(list(candidates)))
				if not puzzle.validate():
					puzzle = puzzle_old
					continue
				state, _ = dlx.solve(puzzle)
				if state == SudokuSolverState.SOLVED:
					return puzzle
			# Too many givens, retry generation
			return SudokuGenerator.generate()

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
				state, _ = dlx.solve(puzzle_new)
				if state == SudokuSolverState.SOLVED:
					puzzle = puzzle_new
			return puzzle

		return further_remove_cells(puzzle)

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
		for _ in range(5):
			puzzle = SudokuGenerator.generate()
			state, _ = SudokuDancingLinksSolver().solve(puzzle)
			assert state == SudokuSolverState.SOLVED

if __name__ == "__main__":
	t = Test()
	for name, func in Test.__dict__.items():
		if isinstance(func, Callable) and not name.startswith("_"):
			print(f"Running {name}...")
			func(t)
