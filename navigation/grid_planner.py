from collections import deque
import heapq


OCCUPIED = -1
UNKNOWN = 0
FREE = 1

DIRECTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]


def is_valid_cell(grid, cell, start=None):
    row, col = cell
    rows, cols = grid.shape

    if not (0 <= row < rows and 0 <= col < cols):
        return False

    if cell == start:
        return True

    return grid[row, col] == FREE


def get_neighbors(grid, cell, start=None):
    row, col = cell
    neighbors = []

    for dr, dc in DIRECTIONS:
        next_cell = (row + dr, col + dc)

        if is_valid_cell(grid, next_cell, start):
            neighbors.append(next_cell)

    return neighbors


def reconstruct_path(parent, current):
    path = [current]

    while current in parent:
        current = parent[current]
        path.append(current)

    path.reverse()
    return path


def bfs_path(grid, start, goal):
    if start == goal:
        return [start]

    queue = deque([start])
    visited = {start}
    parent = {}

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(parent, current)

        for neighbor in get_neighbors(grid, current, start):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return None


def astar_path(grid, start, goal):
    if start == goal:
        return [start]

    def heuristic(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    open_heap = [(heuristic(start), 0, start)]
    parent = {}
    g_score = {start: 0}
    visited = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            return reconstruct_path(parent, current)

        for neighbor in get_neighbors(grid, current, start):
            tentative_cost = current_cost + 1

            if tentative_cost < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_cost
                parent[neighbor] = current

                priority = tentative_cost + heuristic(neighbor)

                heapq.heappush(
                    open_heap,
                    (priority, tentative_cost, neighbor)
                )

    return None


def get_frontier_cells(grid):
    rows, cols = grid.shape
    frontiers = []

    for row in range(rows):
        for col in range(cols):
            if grid[row, col] != FREE:
                continue

            for dr, dc in DIRECTIONS:
                nr, nc = row + dr, col + dc

                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and grid[nr, nc] == UNKNOWN
                ):
                    frontiers.append((row, col))
                    break

    return frontiers


def count_unknown_neighbors(grid, cell):
    row, col = cell
    rows, cols = grid.shape
    count = 0

    for dr, dc in DIRECTIONS:
        nr, nc = row + dr, col + dc

        if (
            0 <= nr < rows
            and 0 <= nc < cols
            and grid[nr, nc] == UNKNOWN
        ):
            count += 1

    return count


def select_frontier(
    grid,
    start,
    algorithm="astar",
    information_weight=0.75,
    blocked_frontiers=None
):
    frontiers = get_frontier_cells(grid)

    if blocked_frontiers is None:
        blocked_frontiers = set()
    else:
        blocked_frontiers = set(blocked_frontiers)

    # Exclude recently rejected frontier goals.
    available_frontiers = [
        frontier
        for frontier in frontiers
        if frontier not in blocked_frontiers
        and frontier != start
    ]

    if not available_frontiers:
        return None

    best_goal = None
    best_path = None
    best_score = float("-inf")

    for frontier in available_frontiers:

        if algorithm.lower() == "bfs":
            path = bfs_path(grid, start, frontier)

        elif algorithm.lower() == "astar":
            path = astar_path(grid, start, frontier)

        else:
            raise ValueError(
                "Algorithm must be 'bfs' or 'astar'."
            )

        if path is None:
            continue

        path_length = len(path) - 1
        information_gain = count_unknown_neighbors(
            grid, frontier
        )

        score = (
            information_weight * information_gain
            - (1 - information_weight) * path_length
        )

        if score > best_score:
            best_score = score
            best_goal = frontier
            best_path = path

    if best_goal is None:
        return None

    return {
        "algorithm": algorithm.upper(),
        "goal": best_goal,
        "path": best_path,
        "frontier_count": len(available_frontiers),
        "path_length": len(best_path) - 1,
        "success": True,
    }


def plan_frontier_path(
    grid,
    start,
    algorithm="astar",
    blocked_frontiers=None
):
    result = select_frontier(
        grid,
        start,
        algorithm=algorithm,
        blocked_frontiers=blocked_frontiers
    )

    if result is None:
        return {
            "algorithm": algorithm.upper(),
            "goal": None,
            "path": None,
            "frontier_count": len(get_frontier_cells(grid)),
            "path_length": 0,
            "success": False,
        }

    return result