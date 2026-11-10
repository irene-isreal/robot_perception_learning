
import numpy as np

from navigation.grid_planner import (
    bfs_path,
    astar_path,
    get_frontier_cells,
    plan_frontier_path,
)


def main():
    # Create an example partially explored indoor map.
    # -1 = obstacle, 0 = unexplored, 1 = free
    grid = np.zeros((12, 16), dtype=int)

    # Mark an explored region as free.
    grid[1:9, 1:12] = 1

    # Add an obstacle inside the explored region.
    grid[4, 4:8] = -1

    start = (2, 2)

    # Choose one reachable frontier for direct comparison.
    frontiers = get_frontier_cells(grid)

    assert frontiers, "No frontiers were detected!"

    goal = frontiers[0]

    bfs_result = bfs_path(grid, start, goal)
    astar_result = astar_path(grid, start, goal)

    print("\n========== BFS TEST ==========")
    print("Start:", start)
    print("Goal:", goal)
    print("Path:", bfs_result)
    print("Path length:", len(bfs_result) - 1 if bfs_result else None)

    print("\n========== A* TEST ==========")
    print("Start:", start)
    print("Goal:", goal)
    print("Path:", astar_result)
    print("Path length:", len(astar_result) - 1 if astar_result else None)

    assert bfs_result is not None, "BFS failed to find a path!"
    assert astar_result is not None, "A* failed to find a path!"

    assert len(bfs_result) == len(astar_result), (
        "BFS and A* should find equal-length shortest paths "
        "on this four-directional grid."
    )

    print("\n========== FRONTIER PLANNING TEST ==========")

    result = plan_frontier_path(
        grid,
        start,
        algorithm="astar"
    )

    print("Algorithm:", result["algorithm"])
    print("Detected frontiers:", result["frontier_count"])
    print("Selected goal:", result["goal"])
    print("Selected path:", result["path"])
    print("Path length:", result["path_length"])
    print("Success:", result["success"])

    assert result["success"], "Frontier planning failed!"

    print("\nAll grid planner tests passed!")


if __name__ == "__main__":
    main()
