import os
import heapq
import random
from collections import deque
import numpy as np
from numba import njit
from src.environment import DELTAS, OPPOSITE

_NEIGHBORS_CACHE = {}
_NUMBA_NEIGHBORS_CACHE = {}


def _get_neighbors(n):
    if n not in _NEIGHBORS_CACHE:
        table = []
        for y in range(n):
            for x in range(n):
                nbs = []
                if y > 0:
                    nbs.append((y - 1) * n + x)
                if x + 1 < n:
                    nbs.append(y * n + (x + 1))
                if y + 1 < n:
                    nbs.append((y + 1) * n + x)
                if x > 0:
                    nbs.append(y * n + (x - 1))
                table.append(tuple(nbs))
        _NEIGHBORS_CACHE[n] = tuple(table)
    return _NEIGHBORS_CACHE[n]


def _get_numba_neighbors(n):
    if n not in _NUMBA_NEIGHBORS_CACHE:
        nbs_list = _get_neighbors(n)
        arr = np.full((n * n, 4), -1, dtype=np.int32)
        counts = np.zeros(n * n, dtype=np.int32)
        for idx, nbs in enumerate(nbs_list):
            counts[idx] = len(nbs)
            for j, nb in enumerate(nbs):
                arr[idx, j] = nb
        _NUMBA_NEIGHBORS_CACHE[n] = (arr, counts)
    return _NUMBA_NEIGHBORS_CACHE[n]


@njit(cache=True)
def _ga_choose_action(
    n,
    hx,
    hy,
    dir_idx,
    old_tail_idx,
    second_idx,
    body_len,
    has_food,
    fx,
    fy,
    w0,
    w1,
    w2,
    w3,
    w4,
    w5,
    base_grid,
    work_grid,
    stack,
    nbs_arr,
    nbs_counts,
):
    opp_idx = (dir_idx + 2) & 3
    best_idx = dir_idx
    best_utility = -1e18
    dead_utility = w1 + w4
    need_flood = (w2 != 0.0) or (w3 != 0.0)
    inv_max_dist = 1.0 / (2.0 * (n - 1)) if has_food else 0.0
    nn = n * n

    dxs = (0, 1, 0, -1)
    dys = (-1, 0, 1, 0)

    for a_idx in range(4):
        if a_idx == opp_idx:
            continue

        nx = hx + dxs[a_idx]
        ny = hy + dys[a_idx]

        if nx < 0 or nx >= n or ny < 0 or ny >= n:
            if dead_utility > best_utility:
                best_utility = dead_utility
                best_idx = a_idx
            continue

        next_idx = ny * n + nx
        ate = has_food and (nx == fx) and (ny == fy)

        if base_grid[next_idx] != 0 and not (next_idx == old_tail_idx and not ate):
            if dead_utility > best_utility:
                best_utility = dead_utility
                best_idx = a_idx
            continue

        for k in range(nn):
            work_grid[k] = base_grid[k]

        if ate:
            new_tail_idx = old_tail_idx
            new_body_len = body_len + 1
        else:
            work_grid[old_tail_idx] = 0
            new_tail_idx = second_idx if body_len > 1 else next_idx
            new_body_len = body_len

        work_grid[new_tail_idx] = 0
        work_grid[next_idx] = 1

        nb_cnt = nbs_counts[next_idx]
        hazards = 4 - nb_cnt
        for j in range(nb_cnt):
            if work_grid[nbs_arr[next_idx, j]] != 0:
                hazards += 1

        if need_flood:
            stack[0] = next_idx
            sp = 1
            count = 1
            tail_reachable = 0.0
            while sp > 0:
                sp -= 1
                idx = stack[sp]
                c_cnt = nbs_counts[idx]
                for j in range(c_cnt):
                    nb = nbs_arr[idx, j]
                    if nb == new_tail_idx:
                        tail_reachable = 1.0
                    if work_grid[nb] == 0:
                        work_grid[nb] = 1
                        count += 1
                        stack[sp] = nb
                        sp += 1
            max_cells = nn - new_body_len
            if max_cells < 1:
                max_cells = 1
            free_ratio = count / max_cells
            if free_ratio > 1.0:
                free_ratio = 1.0
        else:
            free_ratio = 0.0
            tail_reachable = 0.0

        if has_food:
            dx_f = nx - fx if nx >= fx else fx - nx
            dy_f = ny - fy if ny >= fy else fy - ny
            food_dist = (dx_f + dy_f) * inv_max_dist
        else:
            food_dist = 0.0

        utility = (
            w0
            + w1 * food_dist
            + w2 * free_ratio
            + w3 * tail_reachable
            + w4 * (hazards * 0.25)
            + (w5 if ate else 0.0)
        )
        if utility > best_utility:
            best_utility = utility
            best_idx = a_idx

    return best_idx


def _reconstruct(parent, goal):
    cells, actions = [], []
    cur = goal
    while parent[cur] is not None:
        prev, action = parent[cur]
        cells.append(cur)
        actions.append(action)
        cur = prev
    cells.append(cur)
    cells.reverse()
    actions.reverse()
    return cells, actions


class BaseAgent:
    name = "Base"

    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        self.nodes_expanded = 0
        self.decisions = 0
        self.no_path_count = 0
        self.last_path = []
        self.last_actions = []

    def get_action(self, game):
        raise NotImplementedError


class BaselineAgent(BaseAgent):
    name = "Baseline"

    def __init__(self, seed=42):
        super().__init__()
        self.rng = random.Random(seed)

    def get_action(self, game):
        self.decisions += 1
        self.last_path = []
        moves = game.safe_moves()
        if not moves:
            return game.direction
        if game.food is None:
            return moves[0]

        fx, fy = game.food
        hx, hy = game.head

        def dist_to_food(action):
            dx, dy = DELTAS[action]
            return abs(hx + dx - fx) + abs(hy + dy - fy)

        self.rng.shuffle(moves)
        best_move = min(moves, key=dist_to_food)
        dx, dy = DELTAS[best_move]
        self.last_path = [game.head, (hx + dx, hy + dy)]
        return best_move


class PathAgent(BaseAgent):
    name = "PathAgent"

    def find_path(self, game):
        raise NotImplementedError

    def get_action(self, game):
        self.decisions += 1
        self.last_path, self.last_actions = self.find_path(game)
        if self.last_actions:
            return self.last_actions[0]
        self.no_path_count += 1
        moves = game.safe_moves()
        return moves[0] if moves else game.direction


class BFSAgent(PathAgent):
    name = "BFS"

    def find_path(self, game):
        start, goal = game.head, game.food
        if goal is None:
            return [], []
        n = game.size
        seg = [-1] * (n * n)
        for i, (bx, by) in enumerate(game.body):
            seg[by * n + bx] = i

        parent = {start: None}
        depth = {start: 0}
        queue = deque([start])
        opp = OPPOSITE[game.direction]
        deltas = tuple(DELTAS.items())

        while queue:
            cur = queue.popleft()
            self.nodes_expanded += 1
            if cur == goal:
                return _reconstruct(parent, goal)
            cx, cy = cur
            t = depth[cur] + 1
            is_start = cur == start
            for a, (dx, dy) in deltas:
                if is_start and a == opp:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < n and 0 <= ny < n:
                    nxt = (nx, ny)
                    if nxt not in parent:
                        idx = seg[ny * n + nx]
                        if idx == -1 or idx < t:
                            parent[nxt] = (cur, a)
                            depth[nxt] = t
                            queue.append(nxt)
        return [], []


class AStarAgent(PathAgent):
    name = "A*"

    def find_path(self, game):
        start, goal = game.head, game.food
        if goal is None:
            return [], []
        n = game.size
        gx, gy = goal
        seg = [-1] * (n * n)
        for i, (bx, by) in enumerate(game.body):
            seg[by * n + bx] = i

        parent = {start: None}
        g = {start: 0}
        closed = set()
        counter = 0
        h0 = abs(start[0] - gx) + abs(start[1] - gy)
        heap = [(h0, counter, start)]
        opp = OPPOSITE[game.direction]
        deltas = tuple(DELTAS.items())

        while heap:
            _, _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            self.nodes_expanded += 1
            if cur == goal:
                return _reconstruct(parent, goal)
            cx, cy = cur
            ng = g[cur] + 1
            is_start = cur == start
            for a, (dx, dy) in deltas:
                if is_start and a == opp:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < n and 0 <= ny < n:
                    idx = seg[ny * n + nx]
                    if idx == -1 or idx < ng:
                        nxt = (nx, ny)
                        if nxt not in g or ng < g[nxt]:
                            g[nxt] = ng
                            parent[nxt] = (cur, a)
                            counter += 1
                            f = ng + abs(nx - gx) + abs(ny - gy)
                            heapq.heappush(heap, (f, counter, nxt))
        return [], []


class GAAgent(BaseAgent):
    name = "GA"
    NUM_FEATURES = 6
    _DIR_TO_IDX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
    _IDX_TO_DIR = ("UP", "RIGHT", "DOWN", "LEFT")

    def __init__(self, weights=None, weights_path="data/best_weights.npy", track_path=True):
        super().__init__()
        self.track_path = track_path
        if weights is not None:
            self.weights = tuple(float(x) for x in weights)
        elif os.path.exists(weights_path):
            self.weights = tuple(float(x) for x in np.load(weights_path))
        else:
            self.weights = (5.0, -2.5, 4.0, 1.5, -1.5, 3.0)
        self._buf_size = 0
        self._base_grid = np.empty(0, dtype=np.uint8)
        self._work_grid = np.empty(0, dtype=np.uint8)
        self._stack = np.empty(0, dtype=np.int32)

    def _ensure_buffers(self, n):
        nn = n * n
        if self._buf_size != nn:
            self._buf_size = nn
            self._base_grid = np.zeros(nn, dtype=np.uint8)
            self._work_grid = np.empty(nn, dtype=np.uint8)
            self._stack = np.empty(nn, dtype=np.int32)

    def get_action(self, game):
        self.decisions += 1
        self.nodes_expanded += 3
        n = game.size
        self._ensure_buffers(n)

        base = self._base_grid
        base.fill(0)
        body = game.body
        for bx, by in body:
            base[by * n + bx] = 1

        hx, hy = game.head
        tx, ty = body[0]
        old_tail_idx = ty * n + tx
        body_len = len(body)
        if body_len > 1:
            sx, sy = body[1]
            second_idx = sy * n + sx
        else:
            second_idx = old_tail_idx

        food = game.food
        if food is not None:
            has_food = True
            fx, fy = food
        else:
            has_food = False
            fx, fy = -1, -1

        nbs_arr, nbs_counts = _get_numba_neighbors(n)
        w0, w1, w2, w3, w4, w5 = self.weights

        best_idx = _ga_choose_action(
            n,
            hx,
            hy,
            self._DIR_TO_IDX[game.direction],
            old_tail_idx,
            second_idx,
            body_len,
            has_food,
            fx,
            fy,
            w0,
            w1,
            w2,
            w3,
            w4,
            w5,
            base,
            self._work_grid,
            self._stack,
            nbs_arr,
            nbs_counts,
        )

        best_action = self._IDX_TO_DIR[best_idx]
        if self.track_path:
            dx, dy = DELTAS[best_action]
            self.last_path = [(hx, hy), (hx + dx, hy + dy)]
        return best_action