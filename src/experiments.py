import csv
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment import SnakeGame, BOARD_SIZE, DELTAS
from src.agents import BaselineAgent, BFSAgent, AStarAgent, CSPAgent, GAAgent


def _classify_death(game):
    if len(game.body) >= game.size * game.size:
        return "Full"
    starve_limit = getattr(game, "starvation_limit", 200)
    idle = getattr(game, "idle", getattr(game, "steps_since_food", 0))
    if idle >= starve_limit:
        return "Starve"
    dx, dy = DELTAS[game.direction]
    nx, ny = game.head[0] + dx, game.head[1] + dy
    if not (0 <= nx < game.size and 0 <= ny < game.size):
        return "Wall"
    return "Self"


def evaluate_agent(agent_factory, seeds=range(100, 200), max_steps=10000):
    scores = []
    steps_list = []
    nodes_per_move = []
    no_paths = []
    death_counts = {"Wall": 0, "Self": 0, "Starve": 0, "Full": 0}
    raw_rows = []

    t0 = time.perf_counter()
    for s in seeds:
        agent = agent_factory()
        game = SnakeGame(size=BOARD_SIZE, seed=s)
        while game.alive and game.steps < max_steps:
            game.step(agent.get_action(game))

        cause = _classify_death(game)
        death_counts[cause] += 1
        npm = agent.nodes_expanded / max(1, agent.decisions)

        scores.append(game.score)
        steps_list.append(game.steps)
        nodes_per_move.append(npm)
        no_paths.append(agent.no_path_count)
        raw_rows.append((agent.name, s, game.score, game.steps, npm, agent.no_path_count, cause))

    elapsed = time.perf_counter() - t0
    n = len(seeds)
    scores_arr = np.array(scores, dtype=np.float64)

    summary = {
        "Agent": agent_factory().name,
        "AvgScore": float(np.mean(scores_arr)),
        "StdScore": float(np.std(scores_arr)),
        "MedianScore": float(np.median(scores_arr)),
        "MaxScore": int(np.max(scores_arr)),
        "AvgSteps": float(np.mean(steps_list)),
        "Nodes/move": float(np.mean(nodes_per_move)),
        "NoPath/game": float(np.mean(no_paths)),
        "Time(s)": elapsed,
        "Wall%": 100.0 * death_counts["Wall"] / n,
        "Self%": 100.0 * death_counts["Self"] / n,
        "Starve%": 100.0 * death_counts["Starve"] / n,
        "Full%": 100.0 * death_counts["Full"] / n,
    }
    return summary, scores_arr, raw_rows


def _generate_charts(summaries, score_distributions, out_dir="outputs", history_path="data/training_history.csv"):
    os.makedirs(out_dir, exist_ok=True)
    names = [s["Agent"] for s in summaries]
    colors = ["#7f8c8d", "#3498db", "#2ecc71", "#1abc9c", "#e67e22", "#9b59b6"][: len(names)]

    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=150)
    bp = ax.boxplot(
        score_distributions,
        tick_labels=names,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title("Snake AI Score Distribution Across 100 Benchmark Seeds", fontsize=12, fontweight="bold")
    ax.set_ylabel("Apples Collected (Score)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_distribution.png"))
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)
    avg_scores = [s["AvgScore"] for s in summaries]
    nodes = [s["Nodes/move"] for s in summaries]
    times = [s["Time(s)"] for s in summaries]

    bars1 = ax1.bar(names, avg_scores, color=colors, alpha=0.85)
    ax1.set_title("Mean Score Comparison", fontweight="bold")
    ax1.set_ylabel("Average Score")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars1, avg_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}", ha="center", fontsize=8.5)

    bars2 = ax2.bar(names, nodes, color=colors, alpha=0.85)
    ax2.set_title("Search Effort (Nodes Expanded / Move)", fontweight="bold")
    ax2.set_ylabel("Avg Nodes Expanded per Move")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val, t_val in zip(bars2, nodes, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}\n({t_val:.1f}s)", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "efficiency_comparison.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    wall = np.array([s["Wall%"] for s in summaries])
    self_c = np.array([s["Self%"] for s in summaries])
    starve = np.array([s["Starve%"] for s in summaries])
    full = np.array([s["Full%"] for s in summaries])

    ax.bar(names, self_c, label="Self Collision", color="#e74c3c", alpha=0.85)
    ax.bar(names, wall, bottom=self_c, label="Wall Collision", color="#34495e", alpha=0.85)
    ax.bar(names, starve, bottom=self_c + wall, label="Starvation (200 idle)", color="#f1c40f", alpha=0.85)
    ax.bar(names, full, bottom=self_c + wall + starve, label="Full Board", color="#2ecc71", alpha=0.85)
    ax.set_title("Game Termination Causes (% of 100 Seeds)", fontweight="bold")
    ax.set_ylabel("Percentage of Games (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "termination_causes.png"))
    plt.close(fig)

    if os.path.exists(history_path):
        gens, bests, means = [], [], []
        with open(history_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gens.append(int(row["generation"]))
                bests.append(float(row["best_fitness"]))
                means.append(float(row["mean_fitness"]))
        if gens:
            fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
            ax.plot(gens, bests, label="Best Fitness", color="#e67e22", linewidth=2)
            ax.plot(gens, means, label="Population Mean Fitness", color="#3498db", linestyle="--", linewidth=1.8)
            ax.set_title("Genetic Algorithm Fitness Progression", fontweight="bold")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness (Score × 100 + Steps × 0.1)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ga_learning_curve.png"))
            plt.close(fig)


def run_benchmark(seeds=range(100, 200), out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    agents = [
        lambda: BaselineAgent(seed=42),
        lambda: BFSAgent(safe_fallback=True),
        lambda: AStarAgent(safe_fallback=True),
        lambda: CSPAgent(horizon=4),
        lambda: GAAgent(track_path=False, lookahead=1),
        lambda: GAAgent(track_path=False, lookahead=2),
    ]

    summaries = []
    distributions = []
    all_raw = []

    header = (
        f"{'Agent':<10} {'AvgScore':>8} {'Std':>6} {'Med':>6} {'Max':>5} "
        f"{'AvgSteps':>9} {'Nodes/mv':>9} {'NoPath':>7} {'Time(s)':>8} | "
        f"{'Self%':>6} {'Wall%':>6} {'Starv%':>6}"
    )
    print(header)
    print("-" * len(header))

    for factory in agents:
        summary, dist, raw = evaluate_agent(factory, seeds=seeds)
        summaries.append(summary)
        distributions.append(dist)
        all_raw.extend(raw)

        print(
            f"{summary['Agent']:<10} {summary['AvgScore']:>8.2f} {summary['StdScore']:>6.2f} "
            f"{summary['MedianScore']:>6.1f} {summary['MaxScore']:>5d} {summary['AvgSteps']:>9.1f} "
            f"{summary['Nodes/move']:>9.1f} {summary['NoPath/game']:>7.2f} {summary['Time(s)']:>8.2f} | "
            f"{summary['Self%']:>5.1f}% {summary['Wall%']:>5.1f}% {summary['Starve%']:>5.1f}%",
            flush=True,
        )

    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "seed", "score", "steps", "nodes_per_move", "no_path_count", "death_cause"])
        writer.writerows(all_raw)

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    _generate_charts(summaries, distributions, out_dir=out_dir)
    print(f"\nSaved CSVs and charts to '{out_dir}/'.")


if __name__ == "__main__":
    run_benchmark()