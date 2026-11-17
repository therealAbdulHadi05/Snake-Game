import time
from concurrent.futures import ProcessPoolExecutor
from src.environment import SnakeGame, BOARD_SIZE
from src.agents import BaselineAgent, BFSAgent, AStarAgent, GAAgent

AGENT_MAP = {
    "Baseline": BaselineAgent,
    "BFS": BFSAgent,
    "A*": AStarAgent,
    "GA": GAAgent,
}


def _run_single_seed(args):
    agent_key, seed, size, max_steps = args
    agent = AGENT_MAP[agent_key]()
    game = SnakeGame(size=size, seed=seed)
    while game.alive and game.steps < max_steps:
        game.step(agent.get_action(game))
    return (
        game.score,
        game.steps,
        agent.nodes_expanded,
        agent.decisions,
        agent.no_path_count,
    )


def evaluate(agent_key, n_games=100, size=BOARD_SIZE, base_seed=1000, max_steps=50000):
    tasks = [(agent_key, base_seed + i, size, max_steps) for i in range(n_games)]
    scores, steps = [], []
    total_nodes = 0
    total_decisions = 0
    total_no_path = 0

    with ProcessPoolExecutor() as pool:
        for s, t, nodes, dec, no_path in pool.map(_run_single_seed, tasks, chunksize=5):
            scores.append(s)
            steps.append(t)
            total_nodes += nodes
            total_decisions += dec
            total_no_path += no_path

    return {
        "agent": agent_key,
        "avg_score": sum(scores) / n_games,
        "max_score": max(scores),
        "avg_steps": sum(steps) / n_games,
        "nodes_per_decision": total_nodes / max(1, total_decisions),
        "no_path_per_game": total_no_path / n_games,
    }


if __name__ == "__main__":
    print(
        f"{'Agent':<10} {'AvgScore':>9} {'MaxScore':>9} {'AvgSteps':>9} "
        f"{'Nodes/move':>11} {'NoPath/game':>12} {'Time(s)':>8}"
    )
    for key in ("Baseline", "BFS", "A*", "GA"):
        t0 = time.perf_counter()
        r = evaluate(key)
        dt = time.perf_counter() - t0
        print(
            f"{r['agent']:<10} {r['avg_score']:>9.2f} {r['max_score']:>9} "
            f"{r['avg_steps']:>9.1f} {r['nodes_per_decision']:>11.1f} "
            f"{r['no_path_per_game']:>12.2f} {dt:>8.2f}"
        )