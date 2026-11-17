import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.environment import SnakeGame, BOARD_SIZE
from src.agents import GAAgent

_WORKER_AGENT = None


def _eval_single_game(args):
    global _WORKER_AGENT
    idx, weights, seed, max_steps = args
    if _WORKER_AGENT is None:
        _WORKER_AGENT = GAAgent(weights=weights, track_path=False)
    else:
        _WORKER_AGENT.weights = weights
    game = SnakeGame(size=BOARD_SIZE, seed=seed)
    step = game.step
    get_action = _WORKER_AGENT.get_action
    while game.alive and game.steps < max_steps:
        step(get_action(game))
    return idx, (game.score * 100.0) + (game.steps * 0.1)


class GeneticTrainer:
    def __init__(
        self,
        pop_size=50,
        generations=60,
        mutation_rate=0.05,
        mutation_sigma=0.3,
        tournament_k=4,
        elitism=2,
        train_seeds=range(1, 21),
        max_steps=5000,
        seed=42,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.train_seeds = tuple(train_seeds)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def _select_parent(self, population, fitnesses):
        idxs = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = idxs[int(np.argmax(fitnesses[idxs]))]
        return population[best_idx]

    def _crossover(self, p1, p2):
        mask = self.rng.random(p1.shape) < 0.5
        return np.where(mask, p1, p2)

    def _mutate(self, genome):
        mask = self.rng.random(genome.shape) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_sigma, size=genome.shape)
        return genome + np.where(mask, noise, 0.0)

    def train(self, save_path="data/best_weights.npy", verbose=True):
        population = self.rng.uniform(
            -5.0, 5.0, size=(self.pop_size, GAAgent.NUM_FEATURES)
        )
        history = []
        fitness_cache = {}
        best_overall_weights = None
        best_overall_fitness = -float("inf")
        n_seeds = len(self.train_seeds)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with ProcessPoolExecutor() as pool:
            for gen in range(self.generations):
                t0 = time.perf_counter()
                fitnesses = np.zeros(self.pop_size, dtype=np.float64)
                uncached_keys = {}
                tasks = []

                for i, ind in enumerate(population):
                    key = tuple(np.round(ind, 10))
                    if key in fitness_cache:
                        fitnesses[i] = fitness_cache[key]
                    else:
                        uncached_keys[i] = key
                        w_tuple = tuple(float(x) for x in ind)
                        for s in self.train_seeds:
                            tasks.append((i, w_tuple, s, self.max_steps))

                if tasks:
                    for idx, fit in pool.map(_eval_single_game, tasks, chunksize=8):
                        fitnesses[idx] += fit
                    for idx, key in uncached_keys.items():
                        fitnesses[idx] /= n_seeds
                        fitness_cache[key] = fitnesses[idx]

                best_idx = int(np.argmax(fitnesses))
                gen_best = float(fitnesses[best_idx])
                gen_mean = float(np.mean(fitnesses))
                history.append((gen + 1, gen_best, gen_mean))

                if gen_best > best_overall_fitness:
                    best_overall_fitness = gen_best
                    best_overall_weights = population[best_idx].copy()
                    if save_path:
                        np.save(save_path, best_overall_weights)

                if verbose:
                    dt = time.perf_counter() - t0
                    print(
                        f"Gen {gen + 1:02d}/{self.generations} | "
                        f"Best: {gen_best:7.2f} | Mean: {gen_mean:7.2f} | "
                        f"Eval: {len(uncached_keys):2d}/{self.pop_size} | Time: {dt:4.1f}s",
                        flush=True,
                    )

                elite_idxs = np.argsort(fitnesses)[-self.elitism :]
                next_pop = [population[i].copy() for i in elite_idxs]

                while len(next_pop) < self.pop_size:
                    p1 = self._select_parent(population, fitnesses)
                    p2 = self._select_parent(population, fitnesses)
                    child = self._mutate(self._crossover(p1, p2))
                    next_pop.append(child)

                population = np.array(next_pop)

        return best_overall_weights, history


if __name__ == "__main__":
    weights, _ = GeneticTrainer().train()
    print("Saved best weights:", np.round(weights, 4))