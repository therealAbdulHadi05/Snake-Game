import random

UP, RIGHT, DOWN, LEFT = "UP", "RIGHT", "DOWN", "LEFT"
DELTAS = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

BOARD_SIZE = 20
STARVATION_LIMIT = 200


class SnakeGame:
    def __init__(self, size=BOARD_SIZE, seed=None, starvation_limit=STARVATION_LIMIT):
        self.size = size
        self.starvation_limit = starvation_limit
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        mid = self.size // 2
        self.body = [(mid - 2, mid), (mid - 1, mid), (mid, mid)]
        self.occupied = set(self.body)
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.idle = 0
        self.alive = True
        self.won = False
        self._place_food()
        return self.get_state()

    @property
    def head(self):
        return self.body[-1]

    @property
    def tail(self):
        return self.body[0]

    def _place_food(self):
        free = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.occupied
        ]
        if not free:
            self.food = None
            self.alive = False
            self.won = True
            return
        self.food = self.rng.choice(free)

    def in_bounds(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    def can_enter(self, p, eating):
        if not self.in_bounds(p):
            return False
        if p not in self.occupied:
            return True
        return p == self.tail and not eating

    def is_safe(self, p):
        return self.can_enter(p, eating=(p == self.food))

    def step(self, action):
        if not self.alive:
            return False, False
        if action == OPPOSITE[self.direction]:
            action = self.direction
        self.direction = action
        dx, dy = DELTAS[action]
        new_head = (self.head[0] + dx, self.head[1] + dy)
        self.steps += 1
        self.idle += 1

        ate = new_head == self.food
        if not self.can_enter(new_head, ate):
            self.alive = False
            return False, False

        if not ate:
            self.occupied.discard(self.body.pop(0))
        self.body.append(new_head)
        self.occupied.add(new_head)

        if ate:
            self.score += 1
            self.idle = 0
            self._place_food()
        elif self.idle >= self.starvation_limit:
            self.alive = False
        return self.alive, ate

    def safe_moves(self):
        out = []
        for a, (dx, dy) in DELTAS.items():
            if a == OPPOSITE[self.direction]:
                continue
            if self.is_safe((self.head[0] + dx, self.head[1] + dy)):
                out.append(a)
        return out

    def get_state(self):
        return {
            "body": list(self.body),
            "food": self.food,
            "direction": self.direction,
            "size": self.size,
        }

    def clone(self):
        g = SnakeGame.__new__(SnakeGame)
        g.size = self.size
        g.starvation_limit = self.starvation_limit
        g.rng = None
        g.body = list(self.body)
        g.occupied = set(self.occupied)
        g.direction = self.direction
        g.score = self.score
        g.steps = self.steps
        g.idle = self.idle
        g.alive = self.alive
        g.won = self.won
        g.food = self.food
        return g