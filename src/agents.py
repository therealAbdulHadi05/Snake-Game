import os
import heapq
import random
from collections import deque
import numpy as np
from numba import njit
from src.environment import DELTAS, OPPOSITE

_NEIGHBORS_CACHE = {}
_NUMBA_NEIGHBORS_CACHE = {}

_ACTIONS_TUPLE = (("UP", 0, -1), ("RIGHT", 1, 0), ("DOWN", 0, 1), ("LEFT", -1, 0))
_DIR_TO_IDX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
_IDX_TO_DIR = ("UP", "RIGHT", "DOWN", "LEFT")


def _get_neighbors(n):
    if n not in _NEIGHBORS_CACHE:
        table = []
        for y in range(n):
            for x in range(n):
                nbs = []
                if y > 0:
                    nbs.append(((y - 1) * n + x, "UP"))
                if x + 1 < n:
                    nbs.append((y * n + (x + 1), "RIGHT"))
                if y + 1 < n:
                    nbs.append(((y + 1) * n + x, "DOWN"))
                if x > 0:
                    nbs.append((y * n + (x - 1), "LEFT"))
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
            for j, (nb_idx, _) in enumerate(nbs):
                arr[idx, j] = nb_idx
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


@njit(cache=True)
def _ga_choose_action_2ply(
    n,
    hx,
    hy,
    dir_idx,
    body_xs,
    body_ys,
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
    best_total = -1e18
    dead_utility = w1 + w4
    inv_max_dist = 1.0 / (2.0 * (n - 1)) if has_food else 0.0
    nn = n * n

    dxs = (0, 1, 0, -1)
    dys = (-1, 0, 1, 0)

    old_tail_idx = body_ys[0] * n + body_xs[0]
    second_idx = (body_ys[1] * n + body_xs[1]) if body_len > 1 else old_tail_idx
    third_idx = (body_ys[2] * n + body_xs[2]) if body_len > 2 else second_idx

    for a1 in range(4):
        if a1 == opp_idx:
            continue

        nx1 = hx + dxs[a1]
        ny1 = hy + dys[a1]

        if nx1 < 0 or nx1 >= n or ny1 < 0 or ny1 >= n:
            if dead_utility > best_total:
                best_total = dead_utility
                best_idx = a1
            continue

        idx1 = ny1 * n + nx1
        ate1 = has_food and (nx1 == fx) and (ny1 == fy)

        if base_grid[idx1] != 0 and not (idx1 == old_tail_idx and not ate1):
            if dead_utility > best_total:
                best_total = dead_utility
                best_idx = a1
            continue

        for k in range(nn):
            work_grid[k] = base_grid[k]

        if ate1:
            tail1_idx = old_tail_idx
            len1 = body_len + 1
        else:
            work_grid[old_tail_idx] = 0
            tail1_idx = second_idx if body_len > 1 else idx1
            len1 = body_len

        work_grid[tail1_idx] = 0
        work_grid[idx1] = 1

        nb_cnt = nbs_counts[idx1]
        haz1 = 4 - nb_cnt
        for j in range(nb_cnt):
            if work_grid[nbs_arr[idx1, j]] != 0:
                haz1 += 1

        stack[0] = idx1
        sp = 1
        cnt1 = 1
        tr1 = 0.0
        while sp > 0:
            sp -= 1
            c_idx = stack[sp]
            c_cnt = nbs_counts[c_idx]
            for j in range(c_cnt):
                nb = nbs_arr[c_idx, j]
                if nb == tail1_idx:
                    tr1 = 1.0
                if work_grid[nb] == 0:
                    work_grid[nb] = 1
                    cnt1 += 1
                    stack[sp] = nb
                    sp += 1
        mc1 = max(1, nn - len1)
        fr1 = min(1.0, cnt1 / mc1)

        if has_food:
            dx_f = nx1 - fx if nx1 >= fx else fx - nx1
            dy_f = ny1 - fy if ny1 >= fy else fy - ny1
            fd1 = (dx_f + dy_f) * inv_max_dist
        else:
            fd1 = 0.0

        u1 = w0 + w1 * fd1 + w2 * fr1 + w3 * tr1 + w4 * (haz1 * 0.25) + (w5 if ate1 else 0.0)

        opp2 = (a1 + 2) & 3
        best_u2 = dead_utility

        for a2 in range(4):
            if a2 == opp2:
                continue
            nx2 = nx1 + dxs[a2]
            ny2 = ny1 + dys[a2]
            if nx2 < 0 or nx2 >= n or ny2 < 0 or ny2 >= n:
                continue

            idx2 = ny2 * n + nx2
            ate2 = has_food and (not ate1) and (nx2 == fx) and (ny2 == fy)

            occupied_after_1 = (base_grid[idx2] != 0 and not (idx2 == old_tail_idx and not ate1)) or (idx2 == idx1)
            if occupied_after_1 and not (idx2 == tail1_idx and not ate2):
                continue

            for k in range(nn):
                work_grid[k] = base_grid[k]
            if not ate1:
                work_grid[old_tail_idx] = 0
            work_grid[idx1] = 1

            if ate2:
                tail2_idx = tail1_idx
                len2 = len1 + 1
            else:
                work_grid[tail1_idx] = 0
                if ate1:
                    tail2_idx = second_idx if body_len > 1 else idx1
                else:
                    if body_len > 2:
                        tail2_idx = third_idx
                    elif body_len == 2:
                        tail2_idx = idx1
                    else:
                        tail2_idx = idx2
                len2 = len1

            work_grid[tail2_idx] = 0
            work_grid[idx2] = 1

            nb_cnt2 = nbs_counts[idx2]
            haz2 = 4 - nb_cnt2
            for j in range(nb_cnt2):
                if work_grid[nbs_arr[idx2, j]] != 0:
                    haz2 += 1

            stack[0] = idx2
            sp = 1
            cnt2 = 1
            tr2 = 0.0
            while sp > 0:
                sp -= 1
                c_idx = stack[sp]
                c_cnt = nbs_counts[c_idx]
                for j in range(c_cnt):
                    nb = nbs_arr[c_idx, j]
                    if nb == tail2_idx:
                        tr2 = 1.0
                    if work_grid[nb] == 0:
                        work_grid[nb] = 1
                        cnt2 += 1
                        stack[sp] = nb
                        sp += 1
            mc2 = max(1, nn - len2)
            fr2 = min(1.0, cnt2 / mc2)

            if has_food and not ate1:
                dx_f2 = nx2 - fx if nx2 >= fx else fx - nx2
                dy_f2 = ny2 - fy if ny2 >= fy else fy - ny2
                fd2 = (dx_f2 + dy_f2) * inv_max_dist
            else:
                fd2 = 0.0

            u2 = w0 + w1 * fd2 + w2 * fr2 + w3 * tr2 + w4 * (haz2 * 0.25) + (w5 if ate2 else 0.0)
            if u2 > best_u2:
                best_u2 = u2

        total = u1 + 0.5 * best_u2
        if total > best_total:
            best_total = total
            best_idx = a1

    return best_idx


def _reconstruct_1d(parent, act_arr, start_idx, goal_idx, n):
    cells = []
    actions = []
    cur = goal_idx
    while cur != start_idx and cur != -1:
        cells.append((cur % n, cur // n))
        actions.append(_IDX_TO_DIR[act_arr[cur]])
        cur = parent[cur]
    cells.append((start_idx % n, start_idx // n))
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
        self.last_path = [game.head, (hx + dx, hy + dy), game.food]
        return best_move


class PathAgent(BaseAgent):
    name = "PathAgent"

    def __init__(self, safe_fallback=True):
        super().__init__()
        self.safe_fallback = safe_fallback

    def find_path(self, game, target=None):
        raise NotImplementedError

    def _largest_flood_move(self, game):
        moves = game.safe_moves()
        if not moves:
            return game.direction
        if len(moves) == 1:
            return moves[0]

        n = game.size
        nn = n * n
        base = bytearray(nn)
        for bx, by in game.body:
            base[by * n + bx] = 1
        old_tail_idx = game.tail[1] * n + game.tail[0]
        neighbors = _get_neighbors(n)

        best_move = moves[0]
        best_area = -1
        for m in moves:
            dx, dy = DELTAS[m]
            nx, ny = game.head[0] + dx, game.head[1] + dy
            next_idx = ny * n + nx
            ate = (nx, ny) == game.food

            grid = bytearray(base)
            if not ate:
                grid[old_tail_idx] = 0
            grid[next_idx] = 1

            stack = [next_idx]
            area = 1
            while stack:
                idx = stack.pop()
                for nb, _ in neighbors[idx]:
                    if not grid[nb]:
                        grid[nb] = 1
                        area += 1
                        stack.append(nb)

            if area > best_area:
                best_area = area
                best_move = m
        return best_move

    def get_action(self, game):
        self.decisions += 1
        self.last_path, self.last_actions = self.find_path(game, game.food)
        if self.last_actions:
            return self.last_actions[0]

        self.no_path_count += 1
        if self.safe_fallback:
            if len(game.body) > 2:
                tail_path, tail_actions = self.find_path(game, game.tail)
                if tail_actions:
                    self.last_path = tail_path
                    self.last_actions = tail_actions
                    return tail_actions[0]
            best_move = self._largest_flood_move(game)
            dx, dy = DELTAS[best_move]
            self.last_path = [game.head, (game.head[0] + dx, game.head[1] + dy)]
            return best_move

        moves = game.safe_moves()
        return moves[0] if moves else game.direction


class BFSAgent(PathAgent):
    name = "BFS"

    def find_path(self, game, target=None):
        goal = game.food if target is None else target
        start = game.head
        if goal is None or start == goal:
            return [], []

        n = game.size
        nn = n * n
        start_idx = start[1] * n + start[0]
        goal_idx = goal[1] * n + goal[0]

        seg = [-1] * nn
        for i, (bx, by) in enumerate(game.body):
            seg[by * n + bx] = i

        parent = [-1] * nn
        act_arr = [-1] * nn
        depth = [-1] * nn
        depth[start_idx] = 0
        queue = deque([start_idx])
        opp = OPPOSITE[game.direction]
        neighbors = _get_neighbors(n)

        while queue:
            cur = queue.popleft()
            self.nodes_expanded += 1
            if cur == goal_idx:
                return _reconstruct_1d(parent, act_arr, start_idx, goal_idx, n)
            t = depth[cur] + 1
            is_start = cur == start_idx
            for nxt, a in neighbors[cur]:
                if is_start and a == opp:
                    continue
                if depth[nxt] == -1:
                    s_idx = seg[nxt]
                    if s_idx == -1 or s_idx < t:
                        parent[nxt] = cur
                        act_arr[nxt] = _DIR_TO_IDX[a]
                        depth[nxt] = t
                        queue.append(nxt)
        return [], []


class AStarAgent(PathAgent):
    name = "A*"

    def find_path(self, game, target=None):
        goal = game.food if target is None else target
        start = game.head
        if goal is None or start == goal:
            return [], []

        n = game.size
        nn = n * n
        gx, gy = goal
        start_idx = start[1] * n + start[0]
        goal_idx = gy * n + gx

        seg = [-1] * nn
        for i, (bx, by) in enumerate(game.body):
            seg[by * n + bx] = i

        parent = [-1] * nn
        act_arr = [-1] * nn
        g_score = [1_000_000] * nn
        g_score[start_idx] = 0
        closed = bytearray(nn)

        counter = 0
        h0 = abs(start[0] - gx) + abs(start[1] - gy)
        heap = [(h0, counter, start_idx)]
        opp = OPPOSITE[game.direction]
        neighbors = _get_neighbors(n)

        while heap:
            _, _, cur = heapq.heappop(heap)
            if closed[cur]:
                continue
            closed[cur] = 1
            self.nodes_expanded += 1
            if cur == goal_idx:
                return _reconstruct_1d(parent, act_arr, start_idx, goal_idx, n)
            ng = g_score[cur] + 1
            is_start = cur == start_idx
            for nxt, a in neighbors[cur]:
                if is_start and a == opp:
                    continue
                if closed[nxt]:
                    continue
                s_idx = seg[nxt]
                if s_idx == -1 or s_idx < ng:
                    if ng < g_score[nxt]:
                        g_score[nxt] = ng
                        parent[nxt] = cur
                        act_arr[nxt] = _DIR_TO_IDX[a]
                        counter += 1
                        f = ng + abs((nxt % n) - gx) + abs((nxt // n) - gy)
                        heapq.heappush(heap, (f, counter, nxt))
        return [], []


class CSPAgent(BaseAgent):
    name = "CSP"

    def __init__(self, horizon=4):
        super().__init__()
        self.horizon = horizon

    def _valid_moves(self, head_idx, cur_dir, occ, tail_idx, food_idx, n, neighbors):
        opp = OPPOSITE[cur_dir]
        valid = []
        for nxt, action in neighbors[head_idx]:
            if action == opp:
                continue
            ate = nxt == food_idx
            if occ[nxt] and not (nxt == tail_idx and not ate):
                continue
            valid.append((nxt, action, ate))
        return valid

    def _satisfies_global_safety(self, head_idx, tail_idx, body_len, occ, n, neighbors):
        grid = bytearray(occ)
        grid[tail_idx] = 0
        grid[head_idx] = 1
        stack = [head_idx]
        area = 1
        tail_seen = head_idx == tail_idx
        while stack:
            cur = stack.pop()
            for nb, _ in neighbors[cur]:
                if nb == tail_idx:
                    tail_seen = True
                if not grid[nb]:
                    grid[nb] = 1
                    area += 1
                    stack.append(nb)
        return tail_seen or area >= body_len

    def _backtrack(self, step, max_depth, head_idx, cur_dir, body_q, occ, food_idx, fx, fy, n, neighbors, path_cells, path_actions):
        if step == max_depth:
            if self._satisfies_global_safety(head_idx, body_q[0], len(body_q), occ, n, neighbors):
                return True
            return False

        candidates = self._valid_moves(head_idx, cur_dir, occ, body_q[0], food_idx, n, neighbors)
        if not candidates:
            return False

        if food_idx != -1:
            candidates.sort(key=lambda item: abs((item[0] % n) - fx) + abs((item[0] // n) - fy))

        for nxt, action, ate in candidates:
            self.nodes_expanded += 1
            removed_tail = None
            if not ate:
                removed_tail = body_q.popleft()
                occ[removed_tail] = 0
            body_q.append(nxt)
            occ[nxt] = 1

            next_food = -1 if ate else food_idx
            forward_ok = True
            if step + 1 < max_depth:
                if not self._valid_moves(nxt, action, occ, body_q[0], next_food, n, neighbors):
                    forward_ok = False

            if forward_ok:
                path_cells.append((nxt % n, nxt // n))
                path_actions.append(action)
                if self._backtrack(
                    step + 1,
                    max_depth,
                    nxt,
                    action,
                    body_q,
                    occ,
                    next_food,
                    fx,
                    fy,
                    n,
                    neighbors,
                    path_cells,
                    path_actions,
                ):
                    return True
                path_cells.pop()
                path_actions.pop()

            body_q.pop()
            occ[nxt] = 0
            if removed_tail is not None:
                body_q.appendleft(removed_tail)
                occ[removed_tail] = 1

        return False

    def get_action(self, game):
        self.decisions += 1
        n = game.size
        nn = n * n
        neighbors = _get_neighbors(n)

        base_occ = bytearray(nn)
        base_body = deque()
        for bx, by in game.body:
            idx = by * n + bx
            base_occ[idx] = 1
            base_body.append(idx)

        head_idx = game.head[1] * n + game.head[0]
        if game.food is not None:
            fx, fy = game.food
            food_idx = fy * n + fx
        else:
            fx, fy = -1, -1
            food_idx = -1

        for k in range(self.horizon, 0, -1):
            occ = bytearray(base_occ)
            body_q = deque(base_body)
            path_cells = [game.head]
            path_actions = []
            if self._backtrack(
                0,
                k,
                head_idx,
                game.direction,
                body_q,
                occ,
                food_idx,
                fx,
                fy,
                n,
                neighbors,
                path_cells,
                path_actions,
            ):
                if k < self.horizon:
                    self.no_path_count += 1
                self.last_path = path_cells
                self.last_actions = path_actions
                return path_actions[0]

        self.no_path_count += 1
        fallback = self._valid_moves(head_idx, game.direction, base_occ, base_body[0], food_idx, n, neighbors)
        if not fallback:
            self.last_path = []
            self.last_actions = []
            return game.direction

        best_action = fallback[0][1]
        best_area = -1
        for nxt, action, ate in fallback:
            grid = bytearray(base_occ)
            if not ate:
                grid[base_body[0]] = 0
            grid[nxt] = 1
            stack = [nxt]
            area = 1
            while stack:
                cur = stack.pop()
                for nb, _ in neighbors[cur]:
                    if not grid[nb]:
                        grid[nb] = 1
                        area += 1
                        stack.append(nb)
            if area > best_area:
                best_area = area
                best_action = action

        dx, dy = DELTAS[best_action]
        self.last_path = [game.head, (game.head[0] + dx, game.head[1] + dy)]
        self.last_actions = [best_action]
        return best_action


class GAAgent(BaseAgent):
    name = "GA"
    NUM_FEATURES = 6

    def __init__(self, weights=None, weights_path="data/best_weights.npy", track_path=True, lookahead=1):
        super().__init__()
        self.track_path = track_path
        self.lookahead = lookahead
        if lookahead == 2:
            self.name = "GA (2-Ply)"
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
        self._body_xs = np.empty(0, dtype=np.int32)
        self._body_ys = np.empty(0, dtype=np.int32)

    def _ensure_buffers(self, n):
        nn = n * n
        if self._buf_size != nn:
            self._buf_size = nn
            self._base_grid = np.zeros(nn, dtype=np.uint8)
            self._work_grid = np.empty(nn, dtype=np.uint8)
            self._stack = np.empty(nn, dtype=np.int32)
            self._body_xs = np.empty(nn, dtype=np.int32)
            self._body_ys = np.empty(nn, dtype=np.int32)

    def get_action(self, game):
        self.decisions += 1
        self.nodes_expanded += 3 if self.lookahead == 1 else 9
        n = game.size
        self._ensure_buffers(n)

        base = self._base_grid
        base.fill(0)
        body = game.body
        body_len = len(body)
        for i, (bx, by) in enumerate(body):
            base[by * n + bx] = 1
            self._body_xs[i] = bx
            self._body_ys[i] = by

        if getattr(game, "is_2p", False):
            for bx, by in getattr(game, "_blocked", ()):
                if 0 <= bx < n and 0 <= by < n:
                    base[by * n + bx] = 1

        hx, hy = game.head
        tx, ty = body[0]
        old_tail_idx = ty * n + tx
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

        if getattr(game, "is_2p", False):
            idle = getattr(game, "idle", 0)
            urgency = 1.0 + (idle / 22.0)
            w1 = w1 * urgency
            w5 = w5 + (idle * 0.08)
            if idle > 35:
                w3 = w3 * 0.25

        if self.lookahead >= 2:
            best_idx = _ga_choose_action_2ply(
                n,
                hx,
                hy,
                _DIR_TO_IDX[game.direction],
                self._body_xs,
                self._body_ys,
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
        else:
            best_idx = _ga_choose_action(
                n,
                hx,
                hy,
                _DIR_TO_IDX[game.direction],
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

        best_action = _IDX_TO_DIR[best_idx]
        if self.track_path:
            dx, dy = DELTAS[best_action]
            nx, ny = hx + dx, hy + dy
            if has_food:
                self.last_path = [(hx, hy), (nx, ny), (fx, fy)]
            else:
                self.last_path = [(hx, hy), (nx, ny)]
        return best_action