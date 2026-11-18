import random
from collections import deque
from src.environment import BOARD_SIZE, DELTAS, OPPOSITE
from src.agents import BaseAgent


class SingleViewAdapter:
    def __init__(self, arena, player_id):
        self.size = arena.size
        self.is_2p = True
        if player_id == 1:
            self.body = arena.p1_body
            self.direction = arena.p1_dir
            self.alive = arena.p1_alive
            self.idle = arena.p1_idle
            other_body = arena.p2_body if arena.p2_alive else ()
        else:
            self.body = arena.p2_body
            self.direction = arena.p2_dir
            self.alive = arena.p2_alive
            self.idle = arena.p2_idle
            other_body = arena.p1_body if arena.p1_alive else ()

        self.starvation_limit = arena.starvation_limit
        self.head = self.body[-1]
        self.tail = self.body[0]
        self.foods = arena.foods
        if arena.foods:
            hx, hy = self.head
            self.food = min(arena.foods, key=lambda f: abs(f[0] - hx) + abs(f[1] - hy))
        else:
            self.food = None

        self._blocked = set(other_body)
        if len(self.body) > 1:
            for seg in list(self.body)[1:]:
                self._blocked.add(seg)

    def safe_moves(self):
        opp = OPPOSITE[self.direction]
        hx, hy = self.head
        moves = []
        for action, (dx, dy) in DELTAS.items():
            if action == opp:
                continue
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in self._blocked:
                moves.append(action)
        return moves


class AdversarialSnakeGame:
    def __init__(self, size=BOARD_SIZE, num_foods=6, seed=100, starvation_limit=200):
        self.size = size
        self.num_foods = num_foods
        self.starvation_limit = starvation_limit
        self.rng = random.Random(seed)
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        mid = self.size // 2
        self.p1_body = deque([(2, mid), (3, mid), (4, mid)])
        self.p1_dir = "RIGHT"
        self.p1_alive = True
        self.p1_score = 0
        self.p1_idle = 0

        self.p2_body = deque([(self.size - 3, mid), (self.size - 4, mid), (self.size - 5, mid)])
        self.p2_dir = "LEFT"
        self.p2_alive = True
        self.p2_score = 0
        self.p2_idle = 0

        self.steps = 0
        self.winner = None
        self.end_reason = ""
        self.foods = []
        self._refill_foods()

    def _refill_foods(self):
        occupied = set(self.p1_body) | set(self.p2_body) | set(self.foods)
        empty = [
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if (x, y) not in occupied
        ]
        self.rng.shuffle(empty)
        needed = max(0, self.num_foods - len(self.foods))
        self.foods.extend(empty[:needed])

    def _resolve_dir(self, current_dir, action):
        if action in DELTAS and action != OPPOSITE[current_dir]:
            return action
        return current_dir

    def step(self, a1, a2):
        if self.winner is not None:
            return

        self.steps += 1
        self.p1_dir = self._resolve_dir(self.p1_dir, a1)
        self.p2_dir = self._resolve_dir(self.p2_dir, a2)

        dx1, dy1 = DELTAS[self.p1_dir]
        dx2, dy2 = DELTAS[self.p2_dir]
        nh1 = (self.p1_body[-1][0] + dx1, self.p1_body[-1][1] + dy1)
        nh2 = (self.p2_body[-1][0] + dx2, self.p2_body[-1][1] + dy2)

        ate1 = nh1 in self.foods
        ate2 = nh2 in self.foods

        p1_next_body = list(self.p1_body if ate1 else list(self.p1_body)[1:])
        p2_next_body = list(self.p2_body if ate2 else list(self.p2_body)[1:])

        p1_dead = (
            not (0 <= nh1[0] < self.size and 0 <= nh1[1] < self.size)
            or nh1 in p1_next_body
            or nh1 in p2_next_body
        )
        p2_dead = (
            not (0 <= nh2[0] < self.size and 0 <= nh2[1] < self.size)
            or nh2 in p2_next_body
            or nh2 in p1_next_body
        )

        if nh1 == nh2:
            if len(self.p1_body) > len(self.p2_body):
                p2_dead = True
            elif len(self.p2_body) > len(self.p1_body):
                p1_dead = True
            else:
                p1_dead = True
                p2_dead = True

        if not p1_dead:
            if not ate1:
                self.p1_body.popleft()
                self.p1_idle += 1
            else:
                self.p1_score += 1
                self.p1_idle = 0
                if nh1 in self.foods:
                    self.foods.remove(nh1)
            self.p1_body.append(nh1)
            if self.p1_idle >= self.starvation_limit:
                p1_dead = True

        if not p2_dead:
            if not ate2:
                self.p2_body.popleft()
                self.p2_idle += 1
            else:
                self.p2_score += 1
                self.p2_idle = 0
                if nh2 in self.foods:
                    self.foods.remove(nh2)
            self.p2_body.append(nh2)
            if self.p2_idle >= self.starvation_limit:
                p2_dead = True

        self.p1_alive = not p1_dead
        self.p2_alive = not p2_dead

        if p1_dead and p2_dead:
            if self.p1_score > self.p2_score:
                self.winner = "P1"
                self.end_reason = "DOUBLE ELIMINATION (P1 SCORE LEAD)"
            elif self.p2_score > self.p1_score:
                self.winner = "P2"
                self.end_reason = "DOUBLE ELIMINATION (P2 SCORE LEAD)"
            else:
                self.winner = "DRAW"
                self.end_reason = "SIMULTANEOUS ELIMINATION"
        elif p1_dead:
            self.winner = "P2"
            self.end_reason = "P1 STARVED" if self.p1_idle >= self.starvation_limit else "P1 COLLISION"
        elif p2_dead:
            self.winner = "P1"
            self.end_reason = "P2 STARVED" if self.p2_idle >= self.starvation_limit else "P2 COLLISION"
        else:
            self._refill_foods()

    def resolve_timeout(self):
        if self.winner is not None:
            return
        if self.p1_score > self.p2_score:
            self.winner = "P1"
            self.end_reason = "TIME UP (P1 HIGHER SCORE)"
        elif self.p2_score > self.p1_score:
            self.winner = "P2"
            self.end_reason = "TIME UP (P2 HIGHER SCORE)"
        elif self.p1_idle < self.p2_idle:
            self.winner = "P1"
            self.end_reason = "TIME UP (TIEBREAKER: FEWER IDLE)"
        elif self.p2_idle < self.p1_idle:
            self.winner = "P2"
            self.end_reason = "TIME UP (TIEBREAKER: FEWER IDLE)"
        else:
            self.winner = "DRAW"
            self.end_reason = "TIME UP (EXACT DRAW)"


class MinimaxAgent(BaseAgent):
    name = "Minimax"

    def __init__(self, player_id=2, depth=2):
        super().__init__()
        self.player_id = player_id
        self.depth = depth

    def _valid_actions(self, head, direction, blocked, n):
        opp = OPPOSITE[direction]
        hx, hy = head
        out = []
        for action, (dx, dy) in DELTAS.items():
            if action == opp:
                continue
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in blocked:
                out.append((action, (nx, ny)))
        return out

    def _flood_area(self, start, blocked, n, limit=120):
        visited = {start}
        stack = [start]
        count = 1
        while stack and count < limit:
            cx, cy = stack.pop()
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                nxt = (nx, ny)
                if 0 <= nx < n and 0 <= ny < n and nxt not in blocked and nxt not in visited:
                    visited.add(nxt)
                    count += 1
                    stack.append(nxt)
        return count

    def _evaluate(self, my_body, my_score, my_idle, op_body, op_score, foods, n):
        my_head = my_body[-1]
        op_head = op_body[-1]
        blocked = set(my_body) | set(op_body)
        my_area = self._flood_area(my_head, blocked, n)
        op_area = self._flood_area(op_head, blocked, n)

        if foods:
            my_food_dist = min(abs(my_head[0] - fx) + abs(my_head[1] - fy) for fx, fy in foods)
            op_food_dist = min(abs(op_head[0] - fx) + abs(op_head[1] - fy) for fx, fy in foods)
        else:
            my_food_dist = 0
            op_food_dist = 0

        opp_dist = abs(my_head[0] - op_head[0]) + abs(my_head[1] - op_head[1])
        urgency = 1.0 + (my_idle / 35.0)
        area_weight = 1.4 if my_area >= len(my_body) + 8 else 3.5

        return (
            (my_score - op_score) * 220.0
            + (my_area - op_area) * area_weight
            - my_food_dist * 8.5 * urgency
            + op_food_dist * 1.5
            - my_idle * 0.6
            - (18.0 if opp_dist <= 2 and len(my_body) <= len(op_body) else 0.0)
        )

    def _minimax(self, depth, is_max, my_body, my_dir, my_score, my_idle, op_body, op_dir, op_score, foods, n, alpha, beta):
        self.nodes_expanded += 1
        if depth == 0:
            return self._evaluate(my_body, my_score, my_idle, op_body, op_score, foods, n)

        if is_max:
            blocked = set(my_body[1:]) | set(op_body)
            moves = self._valid_actions(my_body[-1], my_dir, blocked, n)
            if not moves:
                return -10000.0 - depth * 100.0
            if foods:
                moves.sort(key=lambda m: min(abs(m[1][0] - fx) + abs(m[1][1] - fy) for fx, fy in foods))
            best = -1e18
            for action, nh in moves:
                ate = nh in foods
                next_body = (my_body if ate else my_body[1:]) + (nh,)
                next_foods = tuple(f for f in foods if f != nh) if ate else foods
                val = self._minimax(
                    depth - 1,
                    False,
                    next_body,
                    action,
                    my_score + (1 if ate else 0),
                    0 if ate else my_idle + 1,
                    op_body,
                    op_dir,
                    op_score,
                    next_foods,
                    n,
                    alpha,
                    beta,
                )
                if val > best:
                    best = val
                if best > alpha:
                    alpha = best
                if beta <= alpha:
                    break
            return best
        else:
            blocked = set(op_body[1:]) | set(my_body)
            moves = self._valid_actions(op_body[-1], op_dir, blocked, n)
            if not moves:
                return 10000.0 + depth * 100.0
            if foods:
                moves.sort(key=lambda m: min(abs(m[1][0] - fx) + abs(m[1][1] - fy) for fx, fy in foods))
            best = 1e18
            for action, nh in moves:
                ate = nh in foods
                next_body = (op_body if ate else op_body[1:]) + (nh,)
                next_foods = tuple(f for f in foods if f != nh) if ate else foods
                val = self._minimax(
                    depth,
                    True,
                    my_body,
                    my_dir,
                    my_score,
                    my_idle,
                    next_body,
                    action,
                    op_score + (1 if ate else 0),
                    next_foods,
                    n,
                    alpha,
                    beta,
                )
                if val < best:
                    best = val
                if best < beta:
                    beta = best
                if beta <= alpha:
                    break
            return best

    def get_action(self, arena):
        self.decisions += 1
        n = arena.size
        if self.player_id == 1:
            my_body = tuple(arena.p1_body)
            my_dir = arena.p1_dir
            my_score = arena.p1_score
            my_idle = arena.p1_idle
            op_body = tuple(arena.p2_body) if arena.p2_alive else ()
            op_dir = arena.p2_dir
            op_score = arena.p2_score
        else:
            my_body = tuple(arena.p2_body)
            my_dir = arena.p2_dir
            my_score = arena.p2_score
            my_idle = arena.p2_idle
            op_body = tuple(arena.p1_body) if arena.p1_alive else ()
            op_dir = arena.p1_dir
            op_score = arena.p1_score

        foods = tuple(arena.foods)
        blocked = set(my_body[1:]) | set(op_body)
        moves = self._valid_actions(my_body[-1], my_dir, blocked, n)
        if not moves:
            self.last_path = []
            return my_dir

        if foods:
            moves.sort(key=lambda m: min(abs(m[1][0] - fx) + abs(m[1][1] - fy) for fx, fy in foods))

        best_action = moves[0][0]
        best_val = -1e18
        alpha = -1e18
        beta = 1e18

        for action, nh in moves:
            ate = nh in foods
            next_body = (my_body if ate else my_body[1:]) + (nh,)
            next_foods = tuple(f for f in foods if f != nh) if ate else foods
            val = self._minimax(
                self.depth - 1,
                False,
                next_body,
                action,
                my_score + (1 if ate else 0),
                0 if ate else my_idle + 1,
                op_body,
                op_dir,
                op_score,
                next_foods,
                n,
                alpha,
                beta,
            )
            if val > best_val:
                best_val = val
                best_action = action
            if best_val > alpha:
                alpha = best_val

        dx, dy = DELTAS[best_action]
        hx, hy = my_body[-1]
        nx, ny = hx + dx, hy + dy
        if foods:
            target = min(foods, key=lambda f: abs(f[0] - nx) + abs(f[1] - ny))
            self.last_path = [(hx, hy), (nx, ny), target]
        else:
            self.last_path = [(hx, hy), (nx, ny)]
        return best_action