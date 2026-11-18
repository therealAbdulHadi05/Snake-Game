import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pygame

from src.environment import SnakeGame, BOARD_SIZE, DELTAS, OPPOSITE
from src.agents import BaselineAgent, BFSAgent, AStarAgent, CSPAgent, GAAgent
from src.adversarial import AdversarialSnakeGame, MinimaxAgent, SingleViewAdapter
from src.experiments import _generate_charts


WINDOW_W = 1120
WINDOW_H = 800
CELL_SIZE = 34
GRID_PX = BOARD_SIZE * CELL_SIZE

BEZEL_RECT = pygame.Rect(18, 14, 732, 772)
LCD_RECT = pygame.Rect(38, 28, 692, 48)
ARENA_RECT = pygame.Rect(44, 90, GRID_PX, GRID_PX)
SIDE_RECT = pygame.Rect(766, 14, 336, 772)

COL_BG = (15, 22, 30)
COL_BEZEL = (196, 206, 212)
COL_BEZEL_DARK = (130, 144, 154)
COL_LCD_BG = (11, 27, 34)
COL_LCD_TEXT = (102, 242, 255)

COL_CYAN_TOP = (36, 198, 220)
COL_CYAN_BOT = (18, 152, 178)
COL_GRID_LINE = (58, 214, 232)

COL_P1_HEAD = (255, 82, 28)
COL_P1_BODY = (244, 114, 22)
COL_P2_HEAD = (124, 58, 237)
COL_P2_BODY = (147, 51, 234)
COL_SNAKE_BORDER = (34, 24, 20)
COL_TAIL_RING = (250, 204, 21)

COL_PELLET = (255, 214, 51)
COL_PELLET_CORE = (255, 250, 210)
COL_PELLET_GLOW = (255, 224, 102)

COL_CARD = (24, 33, 45)
COL_CARD_BORDER = (46, 62, 82)
COL_TEXT = (236, 242, 248)
COL_MUTED = (148, 163, 184)
COL_ACCENT = (56, 189, 248)
COL_GOLD = (250, 204, 21)
COL_GOOD = (52, 211, 153)
COL_WARN = (251, 146, 60)
COL_BAD = (248, 113, 113)
COL_BTN = (34, 48, 66)
COL_BTN_HOVER = (52, 74, 100)


def _create_1p_agent(name, seed=42):
    if name == "Baseline":
        return BaselineAgent(seed=seed)
    if name == "BFS":
        return BFSAgent(safe_fallback=True)
    if name == "A*":
        return AStarAgent(safe_fallback=True)
    if name == "CSP":
        return CSPAgent(horizon=4)
    if name == "GA":
        return GAAgent(track_path=True, lookahead=1)
    if name == "GA (2-Ply)":
        return GAAgent(track_path=True, lookahead=2)
    return GAAgent(track_path=True, lookahead=1)


def _create_comp_agent(name, player_id, seed=42):
    if name == "Minimax":
        return MinimaxAgent(player_id=player_id, depth=2)
    return _create_1p_agent(name, seed=seed)


def _eval_batch_seed(args):
    name, seed = args
    agent = _create_1p_agent(name, seed=seed)
    game = SnakeGame(size=BOARD_SIZE, seed=seed)
    record = [(list(game.body), game.direction, game.food, game.score, game.steps, getattr(game, "idle", 0), [])]

    while game.alive and game.steps < 8000:
        action = agent.get_action(game)
        path_snapshot = list(agent.last_path)
        game.step(action)
        record.append(
            (
                list(game.body),
                game.direction,
                game.food,
                game.score,
                game.steps,
                getattr(game, "idle", 0),
                path_snapshot,
            )
        )

    if len(game.body) >= game.size * game.size:
        cause = "Full"
    elif getattr(game, "idle", 0) >= getattr(game, "starvation_limit", 200):
        cause = "Starve"
    else:
        dx, dy = DELTAS[game.direction]
        nx, ny = game.head[0] + dx, game.head[1] + dy
        cause = "Wall" if not (0 <= nx < game.size and 0 <= ny < game.size) else "Self"

    npm = agent.nodes_expanded / max(1, agent.decisions)
    return seed, game.score, game.steps, npm, agent.no_path_count, cause, record


class SnakeGUI:
    MAIN_OPTIONS = [
        "1. SINGLE PLAYER",
        "2. AGENTS SIMULATION",
        "3. PLAYER VS COMP",
        "4. COMP VS COMP",
        "5. CONTROLS",
        "6. EXIT",
    ]

    BATCH_OPTIONS = [10, 25, 100]
    SPEED_OPTIONS = [30, 60, 120]
    COMP_AGENTS = ["Minimax", "GA", "GA (2-Ply)", "CSP", "A*", "Baseline"]
    TIMER_OPTIONS = [60, 90, 120]
    SIM_AGENT_NAMES = ["Baseline", "BFS", "A*", "CSP", "GA", "GA (2-Ply)"]

    SPEED_TIERS = {
        "PLAY_1P": [("SLOW", 8), ("NORMAL", 12), ("FAST", 20), ("TURBO", 35)],
        "SIM_REPLAY": [("SLOW", 15), ("NORMAL", 30), ("FAST", 60), ("TURBO", 120)],
        "PLAY_PVC": [("SLOW", 8), ("NORMAL", 12), ("FAST", 20), ("TURBO", 30)],
        "PLAY_CVC": [("SLOW", 12), ("NORMAL", 24), ("FAST", 45), ("TURBO", 90)],
    }

    SUMMARY_VIEWS = [
        ("TABLE", None),
        ("BOXPLOT", "score_distribution.png"),
        ("EFFICIENCY", "efficiency_comparison.png"),
        ("DEATHS", "termination_causes.png"),
        ("GA CURVE", "ga_learning_curve.png"),
    ]

    def __init__(self):
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("snake.game.gui.1.0")
            except Exception:
                pass

        pygame.init()
        pygame.display.set_caption("Snake Game")

        self.fullscreen = False
        self.display = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        self._load_window_icon()

        self.screen = pygame.Surface((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font_lcd = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_menu = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_ui = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.font_xs = pygame.font.SysFont("consolas", 12, bold=True)

        self.arena_bg = self._build_arena_surface()
        self.state = "MENU"
        self.menu_idx = 0

        self.sim_batch_idx = 0
        self.sim_speed_idx = 1

        self.pvc_agent_idx = 0
        self.pvc_timer_idx = 1

        self.cvc_p1_idx = 0
        self.cvc_p2_idx = 1
        self.cvc_timer_idx = 1

        self.paused = False
        self.show_overlay = True
        self.speed_tier_idx = 1
        self.fps = 12

        self.game_1p = None
        self.pending_dir_1p = None
        self.session_best_1p = 0

        self.sim_results = []
        self.sim_best_idx = 0
        self.replay_frame_idx = 0
        self.replay_frames = []
        self.summary_view_idx = 0
        self.chart_surfaces = {}

        self.arena_2p = None
        self.p1_agent = None
        self.p2_agent = None
        self.pending_dir_p1 = None
        self.match_duration = 90.0
        self.match_remaining = 90.0
        self.last_tick_time = time.perf_counter()
        self.move_accum = 0.0

        self.click_targets = []

    def _load_window_icon(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(project_root, "assets", "icon.png"),
            os.path.join(project_root, "assets", "icon.png.png"),
            os.path.join(project_root, "assets", "icon.ico"),
            os.path.join(os.getcwd(), "assets", "icon.png"),
        ]
        icon_path = next((p for p in candidates if os.path.exists(p)), None)
        if not icon_path:
            print(f"[Icon] Not found. Expected at: {candidates[0]}")
            return

        try:
            raw_surf = pygame.image.load(icon_path).convert_alpha()
            icon_32 = pygame.transform.smoothscale(raw_surf, (32, 32))
            pygame.display.set_icon(icon_32)

            if sys.platform == "win32":
                import ctypes
                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    bgra = pygame.image.tostring(icon_32, "BGRA")
                    hbm_color = ctypes.windll.gdi32.CreateBitmap(32, 32, 1, 32, bgra)
                    hbm_mask = ctypes.windll.gdi32.CreateBitmap(32, 32, 1, 1, None)

                    class ICONINFO(ctypes.Structure):
                        _fields_ = [
                            ("fIcon", ctypes.c_bool),
                            ("xHotspot", ctypes.c_uint32),
                            ("yHotspot", ctypes.c_uint32),
                            ("hbmMask", ctypes.c_void_p),
                            ("hbmColor", ctypes.c_void_p),
                        ]

                    ii = ICONINFO(True, 0, 0, hbm_mask, hbm_color)
                    hicon = ctypes.windll.user32.CreateIconIndirect(ctypes.byref(ii))
                    if hicon:
                        WM_SETICON = 0x0080
                        ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 0, hicon)
                        ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 1, hicon)
                    ctypes.windll.gdi32.DeleteObject(hbm_color)
                    ctypes.windll.gdi32.DeleteObject(hbm_mask)
            print(f"[Icon] Loaded: {icon_path}")
        except Exception as exc:
            print(f"[Icon] Failed to load '{icon_path}': {exc}")

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.display = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        self._load_window_icon()

    def _get_scale_rect(self):
        dw, dh = self.display.get_size()
        scale = min(dw / WINDOW_W, dh / WINDOW_H)
        sw = max(1, int(WINDOW_W * scale))
        sh = max(1, int(WINDOW_H * scale))
        ox = (dw - sw) // 2
        oy = (dh - sh) // 2
        return pygame.Rect(ox, oy, sw, sh), scale

    def _to_canvas_pos(self, win_pos):
        dst, scale = self._get_scale_rect()
        if scale <= 0:
            return (-1, -1)
        cx = int((win_pos[0] - dst.left) / scale)
        cy = int((win_pos[1] - dst.top) / scale)
        return (cx, cy)

    def _present(self):
        self.display.fill((8, 12, 18))
        dst, _ = self._get_scale_rect()
        if dst.size == (WINDOW_W, WINDOW_H):
            self.display.blit(self.screen, dst.topleft)
        else:
            scaled = pygame.transform.smoothscale(self.screen, dst.size)
            self.display.blit(scaled, dst.topleft)
        pygame.display.flip()

    def _build_arena_surface(self):
        surf = pygame.Surface((GRID_PX, GRID_PX))
        for y in range(GRID_PX):
            t = y / max(1, GRID_PX - 1)
            r = int(COL_CYAN_TOP[0] * (1 - t) + COL_CYAN_BOT[0] * t)
            g = int(COL_CYAN_TOP[1] * (1 - t) + COL_CYAN_BOT[1] * t)
            b = int(COL_CYAN_TOP[2] * (1 - t) + COL_CYAN_BOT[2] * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (GRID_PX, y))

        for i in range(BOARD_SIZE + 1):
            pos = i * CELL_SIZE
            pygame.draw.line(surf, COL_GRID_LINE, (pos, 0), (pos, GRID_PX), 1)
            pygame.draw.line(surf, COL_GRID_LINE, (0, pos), (GRID_PX, pos), 1)
        return surf

    def _register_button(self, rect, callback, hover_idx=None):
        self.click_targets.append((rect, callback, hover_idx))

    def _draw_button(self, rect, label, callback, accent=COL_ACCENT, filled=False, font=None):
        mouse_pos = self._to_canvas_pos(pygame.mouse.get_pos())
        hovered = rect.collidepoint(mouse_pos)
        if filled:
            bg = accent if not hovered else tuple(min(255, c + 25) for c in accent)
            fg = (12, 20, 28)
        else:
            bg = COL_BTN_HOVER if hovered else COL_BTN
            fg = accent if hovered else COL_TEXT
        pygame.draw.rect(self.screen, bg, rect, border_radius=8)
        pygame.draw.rect(self.screen, accent if hovered else COL_CARD_BORDER, rect, width=1, border_radius=8)
        use_font = font if font is not None else self.font_ui
        txt = use_font.render(label, True, fg)
        self.screen.blit(txt, txt.get_rect(center=rect.center))
        self._register_button(rect, callback)

    def _draw_bezel_and_lcd(self, lcd_text):
        self.screen.fill(COL_BG)
        pygame.draw.rect(self.screen, COL_BEZEL_DARK, BEZEL_RECT.move(0, 4), border_radius=18)
        pygame.draw.rect(self.screen, COL_BEZEL, BEZEL_RECT, border_radius=18)
        pygame.draw.rect(self.screen, COL_BEZEL_DARK, BEZEL_RECT, width=3, border_radius=18)

        pygame.draw.rect(self.screen, COL_LCD_BG, LCD_RECT, border_radius=8)
        pygame.draw.rect(self.screen, COL_BEZEL_DARK, LCD_RECT, width=2, border_radius=8)

        txt_surf = self.font_lcd.render(lcd_text, True, COL_LCD_TEXT)
        txt_rect = txt_surf.get_rect(center=LCD_RECT.center)
        self.screen.blit(txt_surf, txt_rect)

        frame_outer = ARENA_RECT.inflate(10, 10)
        pygame.draw.rect(self.screen, COL_BEZEL_DARK, frame_outer, width=3, border_radius=6)
        self.screen.blit(self.arena_bg, ARENA_RECT.topleft)

    def _draw_display_toggle(self, y, h=40):
        bx = SIDE_RECT.left + 16
        bw = SIDE_RECT.width - 32
        lbl = "WINDOWED MODE" if self.fullscreen else "FULLSCREEN MODE"
        self._draw_button(pygame.Rect(bx, y, bw, h), lbl, self._toggle_fullscreen, accent=COL_ACCENT)

    def _cycle_speed(self):
        tiers = self.SPEED_TIERS.get(self.state)
        if not tiers:
            return
        self.speed_tier_idx = (self.speed_tier_idx + 1) % len(tiers)
        self.fps = tiers[self.speed_tier_idx][1]

    def _draw_speed_toggle(self, y, h=40):
        tiers = self.SPEED_TIERS.get(self.state, [("NORMAL", self.fps)])
        tier_name, tier_fps = tiers[self.speed_tier_idx % len(tiers)]
        bx = SIDE_RECT.left + 16
        bw = SIDE_RECT.width - 32
        lbl = f"SPEED: {tier_name} ({tier_fps} FPS)"
        self._draw_button(pygame.Rect(bx, y, bw, h), lbl, self._cycle_speed, accent=COL_ACCENT)

    def _draw_pellet(self, gx, gy, pulse_phase=0.0):
        cx = ARENA_RECT.left + gx * CELL_SIZE + CELL_SIZE // 2
        cy = ARENA_RECT.top + gy * CELL_SIZE + CELL_SIZE // 2
        glow_r = int(13 + 2.5 * math.sin(pulse_phase))
        glow_surf = pygame.Surface((glow_r * 2 + 4, glow_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*COL_PELLET_GLOW, 85), (glow_r + 2, glow_r + 2), glow_r)
        self.screen.blit(glow_surf, (cx - glow_r - 2, cy - glow_r - 2))

        r = 8
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        pygame.draw.polygon(self.screen, COL_PELLET, pts)
        pygame.draw.polygon(self.screen, (180, 130, 10), pts, 1)
        pygame.draw.circle(self.screen, COL_PELLET_CORE, (cx - 1, cy - 1), 2)

    def _draw_snake(self, body, direction, head_col, body_col, show_tail_ring=True):
        if not body:
            return
        pts = list(body)
        pad = 3

        for i in range(len(pts)):
            x, y = pts[i]
            rx = ARENA_RECT.left + x * CELL_SIZE + pad
            ry = ARENA_RECT.top + y * CELL_SIZE + pad
            rw = CELL_SIZE - pad * 2
            rh = CELL_SIZE - pad * 2
            col = head_col if i == len(pts) - 1 else body_col
            pygame.draw.rect(self.screen, col, (rx, ry, rw, rh), border_radius=5)

            if i + 1 < len(pts):
                nx, ny = pts[i + 1]
                if nx == x + 1:
                    pygame.draw.rect(self.screen, col, (rx + rw - 2, ry, pad * 2 + 2, rh))
                elif nx == x - 1:
                    pygame.draw.rect(self.screen, col, (rx - pad * 2, ry, pad * 2 + 2, rh))
                elif ny == y + 1:
                    pygame.draw.rect(self.screen, col, (rx, ry + rh - 2, rw, pad * 2 + 2))
                elif ny == y - 1:
                    pygame.draw.rect(self.screen, col, (rx, ry - pad * 2, rw, pad * 2 + 2))

        for i, (x, y) in enumerate(pts):
            rx = ARENA_RECT.left + x * CELL_SIZE + pad
            ry = ARENA_RECT.top + y * CELL_SIZE + pad
            rw = CELL_SIZE - pad * 2
            rh = CELL_SIZE - pad * 2
            pygame.draw.rect(self.screen, COL_SNAKE_BORDER, (rx, ry, rw, rh), width=1, border_radius=5)

        if show_tail_ring and len(pts) > 1:
            tx, ty = pts[0]
            trect = pygame.Rect(
                ARENA_RECT.left + tx * CELL_SIZE + 5,
                ARENA_RECT.top + ty * CELL_SIZE + 5,
                CELL_SIZE - 10,
                CELL_SIZE - 10,
            )
            pygame.draw.rect(self.screen, COL_TAIL_RING, trect, width=2, border_radius=4)

        hx, hy = pts[-1]
        cx = ARENA_RECT.left + hx * CELL_SIZE + CELL_SIZE // 2
        cy = ARENA_RECT.top + hy * CELL_SIZE + CELL_SIZE // 2
        offsets = {
            "UP": [(-5, -5), (5, -5)],
            "DOWN": [(-5, 5), (5, 5)],
            "LEFT": [(-5, -5), (-5, 5)],
            "RIGHT": [(5, -5), (5, 5)],
        }
        for ox, oy in offsets.get(direction, [(0, 0)]):
            pygame.draw.circle(self.screen, (255, 255, 255), (cx + ox, cy + oy), 3)

    def _draw_path_overlay(self, path, color=(255, 255, 255)):
        if not self.show_overlay or not path or len(path) < 2:
            return
        pixel_pts = [
            (
                ARENA_RECT.left + x * CELL_SIZE + CELL_SIZE // 2,
                ARENA_RECT.top + y * CELL_SIZE + CELL_SIZE // 2,
            )
            for x, y in path
        ]
        pygame.draw.lines(self.screen, color, False, pixel_pts, 3)
        for px, py in pixel_pts[1:-1]:
            pygame.draw.circle(self.screen, color, (px, py), 4)
        tx, ty = pixel_pts[-1]
        pygame.draw.circle(self.screen, color, (tx, ty), 10, 2)

    def _draw_side_card(self, y, h, title):
        rect = pygame.Rect(SIDE_RECT.left, y, SIDE_RECT.width, h)
        pygame.draw.rect(self.screen, COL_CARD, rect, border_radius=12)
        pygame.draw.rect(self.screen, COL_CARD_BORDER, rect, width=1, border_radius=12)
        lbl = self.font_ui.render(title, True, COL_ACCENT)
        self.screen.blit(lbl, (rect.left + 16, rect.top + 12))
        pygame.draw.line(
            self.screen,
            COL_CARD_BORDER,
            (rect.left + 16, rect.top + 38),
            (rect.right - 16, rect.top + 38),
            1,
        )
        return rect

    def _draw_center_banner(self, title, lines, accent=COL_GOLD, height_override=None):
        h = height_override if height_override else (160 + len(lines) * 30)
        box = pygame.Rect(0, 0, 480, h)
        box.center = ARENA_RECT.center
        surf = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, (12, 18, 26, 238), surf.get_rect(), border_radius=14)
        pygame.draw.rect(surf, accent, surf.get_rect(), width=2, border_radius=14)
        self.screen.blit(surf, box.topleft)

        t_surf = self.font_title.render(title, True, accent)
        self.screen.blit(t_surf, t_surf.get_rect(centerx=box.centerx, top=box.top + 22))

        for i, line in enumerate(lines):
            l_surf = self.font_ui.render(line, True, COL_TEXT)
            self.screen.blit(l_surf, l_surf.get_rect(centerx=box.centerx, top=box.top + 72 + i * 30))
        return box

    def _draw_selector_row(self, center_x, y, label, value_text, on_prev, on_next, accent=COL_ACCENT):
        lbl_surf = self.font_sm.render(label, True, COL_MUTED)
        self.screen.blit(lbl_surf, lbl_surf.get_rect(centerx=center_x, top=y))

        row_y = y + 22
        left_rect = pygame.Rect(center_x - 180, row_y, 40, 34)
        val_rect = pygame.Rect(center_x - 132, row_y, 264, 34)
        right_rect = pygame.Rect(center_x + 140, row_y, 40, 34)

        self._draw_button(left_rect, "<", on_prev, accent=accent)
        pygame.draw.rect(self.screen, (18, 28, 40), val_rect, border_radius=6)
        pygame.draw.rect(self.screen, COL_CARD_BORDER, val_rect, width=1, border_radius=6)
        val_surf = self.font_ui.render(value_text, True, COL_TEXT)
        self.screen.blit(val_surf, val_surf.get_rect(center=val_rect.center))
        self._draw_button(right_rect, ">", on_next, accent=accent)

    def _start_single_player(self):
        self.game_1p = SnakeGame(size=BOARD_SIZE, seed=int(time.time()) % 10000)
        self.pending_dir_1p = None
        self.paused = False
        self.speed_tier_idx = 1
        self.fps = self.SPEED_TIERS["PLAY_1P"][self.speed_tier_idx][1]
        self.move_accum = 0.0
        self.state = "PLAY_1P"

    def _load_chart_surfaces(self, out_dir="outputs"):
        self.chart_surfaces = {}
        for _, filename in self.SUMMARY_VIEWS:
            if filename:
                full_path = os.path.join(out_dir, filename)
                if os.path.exists(full_path):
                    raw_img = pygame.image.load(full_path).convert_alpha()
                    self.chart_surfaces[filename] = raw_img

    def _run_batch_simulation(self):
        batch_size = self.BATCH_OPTIONS[self.sim_batch_idx]
        seeds = list(range(100, 100 + batch_size))
        self.sim_results = []
        summaries_for_export = []
        distributions_for_export = []
        all_raw_rows = []
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)

        with ProcessPoolExecutor() as pool:
            for idx, name in enumerate(self.SIM_AGENT_NAMES):
                self._draw_bezel_and_lcd("BATCH SIMULATION IN PROGRESS...")
                self._draw_center_banner(
                    "SIMULATING AGENTS",
                    [
                        f"Evaluating: {name} ({idx + 1}/{len(self.SIM_AGENT_NAMES)})",
                        f"Seeds: 100..{99 + batch_size} ({batch_size} runs)",
                        "Running multi-core batch...",
                    ],
                    accent=COL_ACCENT,
                )
                self._present()
                pygame.event.pump()

                t0 = time.perf_counter()
                tasks = [(name, s) for s in seeds]
                results = list(pool.map(_eval_batch_seed, tasks, chunksize=4))
                elapsed = time.perf_counter() - t0

                scores = []
                steps_list = []
                nodes_list = []
                no_paths_list = []
                deaths = {"Wall": 0, "Self": 0, "Starve": 0, "Full": 0}
                best_score = -1
                best_seed = 100
                best_frames = []

                for seed, score, steps, npm, no_path, cause, record in results:
                    deaths[cause] += 1
                    scores.append(score)
                    steps_list.append(steps)
                    nodes_list.append(npm)
                    no_paths_list.append(no_path)
                    all_raw_rows.append((name, seed, score, steps, npm, no_path, cause))
                    if score > best_score:
                        best_score = score
                        best_seed = seed
                        best_frames = record

                scores_arr = np.array(scores, dtype=np.float64)
                avg_score = float(np.mean(scores_arr))
                std_score = float(np.std(scores_arr))
                med_score = float(np.median(scores_arr))
                max_score = int(np.max(scores_arr))
                avg_steps = float(np.mean(steps_list))
                nodes_mv = float(np.mean(nodes_list))
                no_path_gm = float(np.mean(no_paths_list))
                wall_pct = 100.0 * deaths["Wall"] / batch_size
                self_pct = 100.0 * deaths["Self"] / batch_size
                starve_pct = 100.0 * deaths["Starve"] / batch_size
                full_pct = 100.0 * deaths["Full"] / batch_size

                summaries_for_export.append(
                    {
                        "Agent": name,
                        "AvgScore": avg_score,
                        "StdScore": std_score,
                        "MedianScore": med_score,
                        "MaxScore": max_score,
                        "AvgSteps": avg_steps,
                        "Nodes/move": nodes_mv,
                        "NoPath/game": no_path_gm,
                        "Time(s)": elapsed,
                        "Wall%": wall_pct,
                        "Self%": self_pct,
                        "Starve%": starve_pct,
                        "Full%": full_pct,
                    }
                )
                distributions_for_export.append(scores_arr)

                self.sim_results.append(
                    {
                        "name": name,
                        "avg_score": avg_score,
                        "std_score": std_score,
                        "med_score": med_score,
                        "max_score": max_score,
                        "avg_steps": avg_steps,
                        "nodes_mv": nodes_mv,
                        "no_path": no_path_gm,
                        "elapsed": elapsed,
                        "self_pct": self_pct,
                        "wall_pct": wall_pct,
                        "starve_pct": starve_pct,
                        "full_pct": full_pct,
                        "best_seed": best_seed,
                        "best_frames": best_frames,
                    }
                )

        with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent", "seed", "score", "steps", "nodes_per_move", "no_path_count", "death_cause"])
            writer.writerows(all_raw_rows)

        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries_for_export[0].keys()))
            writer.writeheader()
            writer.writerows(summaries_for_export)

        _generate_charts(summaries_for_export, distributions_for_export, out_dir=out_dir)
        self._load_chart_surfaces(out_dir=out_dir)
        self.summary_view_idx = 0

        self.sim_best_idx = int(np.argmax([r["max_score"] for r in self.sim_results]))
        self._load_replay(self.sim_best_idx)
        self.state = "SIM_REPLAY"

    def _load_replay(self, idx):
        self.sim_best_idx = idx % len(self.sim_results)
        self.replay_frames = self.sim_results[self.sim_best_idx]["best_frames"]
        self.replay_frame_idx = 0
        self.paused = False
        chosen_fps = self.SPEED_OPTIONS[self.sim_speed_idx]
        tiers = self.SPEED_TIERS["SIM_REPLAY"]
        matched = [i for i, (_, f) in enumerate(tiers) if f == chosen_fps]
        self.speed_tier_idx = matched[0] if matched else 1
        self.fps = tiers[self.speed_tier_idx][1]
        self.move_accum = 0.0

    def _start_pvc(self):
        ai_name = self.COMP_AGENTS[self.pvc_agent_idx]
        self.match_duration = float(self.TIMER_OPTIONS[self.pvc_timer_idx])
        self.match_remaining = self.match_duration
        self.arena_2p = AdversarialSnakeGame(size=BOARD_SIZE, num_foods=6, seed=int(time.time()) % 10000)
        self.p1_agent = None
        self.p2_agent = _create_comp_agent(ai_name, player_id=2, seed=42)
        self.pending_dir_p1 = None
        self.paused = False
        self.speed_tier_idx = 1
        self.fps = self.SPEED_TIERS["PLAY_PVC"][self.speed_tier_idx][1]
        self.move_accum = 0.0
        self.state = "PLAY_PVC"

    def _start_cvc(self):
        p1_name = self.COMP_AGENTS[self.cvc_p1_idx]
        p2_name = self.COMP_AGENTS[self.cvc_p2_idx]
        self.match_duration = float(self.TIMER_OPTIONS[self.cvc_timer_idx])
        self.match_remaining = self.match_duration
        self.arena_2p = AdversarialSnakeGame(size=BOARD_SIZE, num_foods=6, seed=int(time.time()) % 10000)
        self.p1_agent = _create_comp_agent(p1_name, player_id=1, seed=101)
        self.p2_agent = _create_comp_agent(p2_name, player_id=2, seed=202)
        self.paused = False
        self.speed_tier_idx = 1
        self.fps = self.SPEED_TIERS["PLAY_CVC"][self.speed_tier_idx][1]
        self.move_accum = 0.0
        self.state = "PLAY_CVC"

    def _get_2p_action(self, agent, player_id):
        if isinstance(agent, MinimaxAgent):
            return agent.get_action(self.arena_2p)
        view = SingleViewAdapter(self.arena_2p, player_id)
        return agent.get_action(view)

    def _handle_events(self):
        mouse_pos = self._to_canvas_pos(pygame.mouse.get_pos())

        if self.state == "MENU":
            for rect, _, hover_idx in self.click_targets:
                if hover_idx is not None and rect.collidepoint(mouse_pos):
                    self.menu_idx = hover_idx

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.display = pygame.display.set_mode((max(640, event.w), max(480, event.h)), pygame.RESIZABLE)
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                c_pos = self._to_canvas_pos(event.pos)
                for rect, callback, _ in list(self.click_targets):
                    if rect.collidepoint(c_pos):
                        callback()
                        break
                continue

            if event.type != pygame.KEYDOWN:
                continue

            key = event.key
            if key == pygame.K_F11:
                self._toggle_fullscreen()
                continue

            if self.state == "MENU":
                if key in (pygame.K_UP, pygame.K_w):
                    self.menu_idx = (self.menu_idx - 1) % len(self.MAIN_OPTIONS)
                elif key in (pygame.K_DOWN, pygame.K_s):
                    self.menu_idx = (self.menu_idx + 1) % len(self.MAIN_OPTIONS)
                elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    self.menu_idx = key - pygame.K_1
                    self._activate_menu()
                elif key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._activate_menu()

            elif self.state == "CONTROLS":
                if key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                    self.state = "MENU"

            elif self.state == "SETUP_SIM":
                if key == pygame.K_ESCAPE:
                    self.state = "MENU"
                elif key in (pygame.K_LEFT, pygame.K_a):
                    self.sim_batch_idx = (self.sim_batch_idx - 1) % len(self.BATCH_OPTIONS)
                elif key in (pygame.K_RIGHT, pygame.K_d):
                    self.sim_batch_idx = (self.sim_batch_idx + 1) % len(self.BATCH_OPTIONS)
                elif key in (pygame.K_UP, pygame.K_DOWN, pygame.K_w, pygame.K_s):
                    self.sim_speed_idx = (self.sim_speed_idx + 1) % len(self.SPEED_OPTIONS)
                elif key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._run_batch_simulation()

            elif self.state == "SETUP_PVC":
                if key == pygame.K_ESCAPE:
                    self.state = "MENU"
                elif key in (pygame.K_LEFT, pygame.K_a):
                    self.pvc_agent_idx = (self.pvc_agent_idx - 1) % len(self.COMP_AGENTS)
                elif key in (pygame.K_RIGHT, pygame.K_d):
                    self.pvc_agent_idx = (self.pvc_agent_idx + 1) % len(self.COMP_AGENTS)
                elif key in (pygame.K_UP, pygame.K_DOWN, pygame.K_w, pygame.K_s):
                    self.pvc_timer_idx = (self.pvc_timer_idx + 1) % len(self.TIMER_OPTIONS)
                elif key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._start_pvc()

            elif self.state == "SETUP_CVC":
                if key == pygame.K_ESCAPE:
                    self.state = "MENU"
                elif key in (pygame.K_LEFT, pygame.K_a):
                    self.cvc_p1_idx = (self.cvc_p1_idx + 1) % len(self.COMP_AGENTS)
                elif key in (pygame.K_RIGHT, pygame.K_d):
                    self.cvc_p2_idx = (self.cvc_p2_idx + 1) % len(self.COMP_AGENTS)
                elif key in (pygame.K_UP, pygame.K_DOWN, pygame.K_w, pygame.K_s):
                    self.cvc_timer_idx = (self.cvc_timer_idx + 1) % len(self.TIMER_OPTIONS)
                elif key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._start_cvc()

            elif self.state == "PLAY_1P":
                if key == pygame.K_ESCAPE:
                    self.state = "MENU"
                elif key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif key == pygame.K_TAB:
                    self._cycle_speed()
                elif key == pygame.K_r:
                    self._start_single_player()
                elif key in (pygame.K_UP, pygame.K_w):
                    self.pending_dir_1p = "UP"
                elif key in (pygame.K_DOWN, pygame.K_s):
                    self.pending_dir_1p = "DOWN"
                elif key in (pygame.K_LEFT, pygame.K_a):
                    self.pending_dir_1p = "LEFT"
                elif key in (pygame.K_RIGHT, pygame.K_d):
                    self.pending_dir_1p = "RIGHT"

            elif self.state == "SIM_REPLAY":
                if key == pygame.K_ESCAPE:
                    self.state = "MENU"
                elif key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif key == pygame.K_s:
                    self._cycle_speed()
                elif key == pygame.K_TAB:
                    self._load_replay(self.sim_best_idx + 1)
                elif key == pygame.K_v:
                    self.show_overlay = not self.show_overlay
                elif key == pygame.K_r:
                    self.replay_frame_idx = 0
                elif key == pygame.K_RETURN:
                    self.state = "SIM_SUMMARY"

            elif self.state == "SIM_SUMMARY":
                if key == pygame.K_ESCAPE:
                    self.state = "MENU"
                elif key in (pygame.K_LEFT, pygame.K_a):
                    self.summary_view_idx = (self.summary_view_idx - 1) % len(self.SUMMARY_VIEWS)
                elif key in (pygame.K_RIGHT, pygame.K_d):
                    self.summary_view_idx = (self.summary_view_idx + 1) % len(self.SUMMARY_VIEWS)
                elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    self.summary_view_idx = key - pygame.K_1
                elif key == pygame.K_TAB:
                    self._load_replay(self.sim_best_idx + 1)
                    self.state = "SIM_REPLAY"
                elif key == pygame.K_r:
                    self._load_replay(self.sim_best_idx)
                    self.state = "SIM_REPLAY"

            elif self.state in ("PLAY_PVC", "PLAY_CVC"):
                if key == pygame.K_ESCAPE:
                    self.state = "MENU"
                elif key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif key == pygame.K_TAB:
                    self._cycle_speed()
                elif key == pygame.K_v:
                    self.show_overlay = not self.show_overlay
                elif key == pygame.K_r:
                    if self.state == "PLAY_PVC":
                        self._start_pvc()
                    else:
                        self._start_cvc()
                elif self.state == "PLAY_PVC":
                    if key in (pygame.K_UP, pygame.K_w):
                        self.pending_dir_p1 = "UP"
                    elif key in (pygame.K_DOWN, pygame.K_s):
                        self.pending_dir_p1 = "DOWN"
                    elif key in (pygame.K_LEFT, pygame.K_a):
                        self.pending_dir_p1 = "LEFT"
                    elif key in (pygame.K_RIGHT, pygame.K_d):
                        self.pending_dir_p1 = "RIGHT"

    def _activate_menu(self):
        if self.menu_idx == 0:
            self._start_single_player()
        elif self.menu_idx == 1:
            self.state = "SETUP_SIM"
        elif self.menu_idx == 2:
            self.state = "SETUP_PVC"
        elif self.menu_idx == 3:
            self.state = "SETUP_CVC"
        elif self.menu_idx == 4:
            self.state = "CONTROLS"
        elif self.menu_idx == 5:
            pygame.quit()
            sys.exit()

    def _update(self, dt):
        if self.paused:
            return

        if self.state == "PLAY_1P" and self.game_1p and self.game_1p.alive:
            self.move_accum += dt
            step_interval = 1.0 / self.fps
            while self.move_accum >= step_interval and self.game_1p.alive:
                self.move_accum -= step_interval
                action = self.pending_dir_1p or self.game_1p.direction
                if action == OPPOSITE[self.game_1p.direction]:
                    action = self.game_1p.direction
                self.pending_dir_1p = None
                self.game_1p.step(action)
                if self.game_1p.score > self.session_best_1p:
                    self.session_best_1p = self.game_1p.score

        elif self.state == "SIM_REPLAY" and self.replay_frames:
            self.move_accum += dt
            step_interval = 1.0 / self.fps
            while self.move_accum >= step_interval:
                self.move_accum -= step_interval
                if self.replay_frame_idx + 1 < len(self.replay_frames):
                    self.replay_frame_idx += 1
                else:
                    self.state = "SIM_SUMMARY"
                    break

        elif self.state in ("PLAY_PVC", "PLAY_CVC") and self.arena_2p and self.arena_2p.winner is None:
            self.match_remaining = max(0.0, self.match_remaining - dt)
            if self.match_remaining <= 0.0:
                self.arena_2p.resolve_timeout()
                return

            self.move_accum += dt
            step_interval = 1.0 / self.fps
            while self.move_accum >= step_interval and self.arena_2p.winner is None:
                self.move_accum -= step_interval
                if self.state == "PLAY_PVC":
                    a1 = self.pending_dir_p1 or self.arena_2p.p1_dir
                    self.pending_dir_p1 = None
                else:
                    a1 = self._get_2p_action(self.p1_agent, 1)
                a2 = self._get_2p_action(self.p2_agent, 2)
                self.arena_2p.step(a1, a2)

    def _render_menu(self):
        self._draw_bezel_and_lcd("MAIN MENU")
        pulse = time.perf_counter() * 4.0

        row_h = 50
        row_gap = 18
        total_h = len(self.MAIN_OPTIONS) * row_h + (len(self.MAIN_OPTIONS) - 1) * row_gap
        start_y = ARENA_RECT.top + (GRID_PX - total_h) // 2
        row_w = GRID_PX - 180
        row_x = ARENA_RECT.left + (GRID_PX - row_w) // 2

        for i, opt in enumerate(self.MAIN_OPTIONS):
            selected = i == self.menu_idx
            row_rect = pygame.Rect(row_x, start_y + i * (row_h + row_gap), row_w, row_h)
            if selected:
                pygame.draw.rect(self.screen, (11, 32, 42), row_rect, border_radius=10)
                pygame.draw.rect(self.screen, COL_PELLET, row_rect, width=2, border_radius=10)
                cx = row_rect.left + 26
                cy = row_rect.centery
                r = 7 + int(1.5 * math.sin(pulse))
                pygame.draw.polygon(
                    self.screen,
                    COL_PELLET,
                    [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                )
                txt_col = COL_LCD_TEXT
            else:
                pygame.draw.rect(self.screen, (20, 128, 152), row_rect, border_radius=10)
                txt_col = (240, 252, 255)

            lbl = self.font_menu.render(opt, True, txt_col)
            self.screen.blit(lbl, (row_rect.left + 50, row_rect.centery - lbl.get_height() // 2))

            def _make_cb(idx):
                def _cb():
                    self.menu_idx = idx
                    self._activate_menu()
                return _cb

            self._register_button(row_rect, _make_cb(i), hover_idx=i)

        self._draw_side_card(14, 120, "DISPLAY")
        self._draw_display_toggle(68)

    def _render_controls_page(self):
        self._draw_bezel_and_lcd("SYSTEM & GAMEPLAY CONTROLS")
        panel = pygame.Rect(ARENA_RECT.left + 24, ARENA_RECT.top + 24, GRID_PX - 48, GRID_PX - 48)
        pygame.draw.rect(self.screen, (11, 22, 32), panel, border_radius=14)
        pygame.draw.rect(self.screen, COL_ACCENT, panel, width=2, border_radius=14)

        sections = [
            ("MOVEMENT & MENU NAVIGATION", [
                ("WASD / Arrow Keys", "Steer snake & navigate menus"),
                ("Left Click", "Select options, buttons & arrows"),
                ("Enter / Space", "Confirm selection or pause game"),
                ("Escape", "Return to Main Menu"),
            ]),
            ("SIMULATION & AI TELEMETRY", [
                ("V Key / Overlay Button", "Toggle AI target path line"),
                ("Tab Key / Speed Button", "Cycle game speed or best replays"),
                ("Left / Right Arrows", "Switch summary table & charts"),
                ("R Key / Restart Button", "Restart current game or replay"),
            ]),
            ("DISPLAY", [
                ("F11 / Fullscreen Button", "Toggle fullscreen & windowed mode"),
            ]),
        ]

        y = panel.top + 26
        for sec_title, rows in sections:
            hdr = self.font_ui.render(sec_title, True, COL_GOLD)
            self.screen.blit(hdr, (panel.left + 28, y))
            y += 30
            for key_lbl, desc in rows:
                k_surf = self.font_sm.render(f"{key_lbl:<24}", True, COL_LCD_TEXT)
                d_surf = self.font_sm.render(desc, True, COL_TEXT)
                self.screen.blit(k_surf, (panel.left + 36, y))
                self.screen.blit(d_surf, (panel.left + 250, y))
                y += 28
            y += 18

        self._draw_button(
            pygame.Rect(panel.centerx - 120, panel.bottom - 62, 240, 42),
            "MAIN MENU",
            lambda: setattr(self, "state", "MENU"),
            accent=COL_ACCENT,
            filled=True,
        )

        self._draw_side_card(14, 170, "NAVIGATION")
        self._draw_display_toggle(68)
        bx = SIDE_RECT.left + 16
        bw = SIDE_RECT.width - 32
        self._draw_button(pygame.Rect(bx, 118, bw, 40), "MAIN MENU", lambda: setattr(self, "state", "MENU"), accent=COL_BAD)

    def _render_setup(self):
        if self.state == "SETUP_SIM":
            self._draw_bezel_and_lcd("MODE 2: AGENTS SIMULATION")
            box = self._draw_center_banner("AGENTS BATCH SETUP", [], accent=COL_ACCENT, height_override=340)
            batch = self.BATCH_OPTIONS[self.sim_batch_idx]
            spd = self.SPEED_OPTIONS[self.sim_speed_idx]

            self._draw_selector_row(
                box.centerx,
                box.top + 76,
                "SEED BATCH SIZE",
                f"{batch} Seeds (100..{99 + batch})",
                lambda: setattr(self, "sim_batch_idx", (self.sim_batch_idx - 1) % len(self.BATCH_OPTIONS)),
                lambda: setattr(self, "sim_batch_idx", (self.sim_batch_idx + 1) % len(self.BATCH_OPTIONS)),
                accent=COL_ACCENT,
            )
            self._draw_selector_row(
                box.centerx,
                box.top + 152,
                "REPLAY PLAYBACK SPEED",
                f"{spd} FPS",
                lambda: setattr(self, "sim_speed_idx", (self.sim_speed_idx - 1) % len(self.SPEED_OPTIONS)),
                lambda: setattr(self, "sim_speed_idx", (self.sim_speed_idx + 1) % len(self.SPEED_OPTIONS)),
                accent=COL_ACCENT,
            )

            self._draw_button(
                pygame.Rect(box.centerx - 180, box.bottom - 62, 165, 40),
                "MAIN MENU",
                lambda: setattr(self, "state", "MENU"),
                accent=COL_MUTED,
            )
            self._draw_button(
                pygame.Rect(box.centerx + 15, box.bottom - 62, 165, 40),
                "START",
                self._run_batch_simulation,
                accent=COL_ACCENT,
                filled=True,
            )

        elif self.state == "SETUP_PVC":
            self._draw_bezel_and_lcd("MODE 3: PLAYER VS COMP")
            box = self._draw_center_banner("PLAYER VS COMP SETUP", [], accent=COL_GOLD, height_override=340)
            ai = self.COMP_AGENTS[self.pvc_agent_idx]
            tm = self.TIMER_OPTIONS[self.pvc_timer_idx]

            self._draw_selector_row(
                box.centerx,
                box.top + 76,
                "P2 RIVAL AI (PURPLE)",
                ai,
                lambda: setattr(self, "pvc_agent_idx", (self.pvc_agent_idx - 1) % len(self.COMP_AGENTS)),
                lambda: setattr(self, "pvc_agent_idx", (self.pvc_agent_idx + 1) % len(self.COMP_AGENTS)),
                accent=COL_GOLD,
            )
            self._draw_selector_row(
                box.centerx,
                box.top + 152,
                "MATCH TIMER",
                f"{tm} Seconds",
                lambda: setattr(self, "pvc_timer_idx", (self.pvc_timer_idx - 1) % len(self.TIMER_OPTIONS)),
                lambda: setattr(self, "pvc_timer_idx", (self.pvc_timer_idx + 1) % len(self.TIMER_OPTIONS)),
                accent=COL_GOLD,
            )

            self._draw_button(
                pygame.Rect(box.centerx - 180, box.bottom - 62, 165, 40),
                "MAIN MENU",
                lambda: setattr(self, "state", "MENU"),
                accent=COL_MUTED,
            )
            self._draw_button(
                pygame.Rect(box.centerx + 15, box.bottom - 62, 165, 40),
                "START",
                self._start_pvc,
                accent=COL_GOLD,
                filled=True,
            )

        elif self.state == "SETUP_CVC":
            self._draw_bezel_and_lcd("MODE 4: COMP VS COMP")
            box = self._draw_center_banner("COMP VS COMP SETUP", [], accent=COL_GOOD, height_override=410)
            p1 = self.COMP_AGENTS[self.cvc_p1_idx]
            p2 = self.COMP_AGENTS[self.cvc_p2_idx]
            tm = self.TIMER_OPTIONS[self.cvc_timer_idx]

            self._draw_selector_row(
                box.centerx,
                box.top + 72,
                "P1 AGENT (ORANGE)",
                p1,
                lambda: setattr(self, "cvc_p1_idx", (self.cvc_p1_idx - 1) % len(self.COMP_AGENTS)),
                lambda: setattr(self, "cvc_p1_idx", (self.cvc_p1_idx + 1) % len(self.COMP_AGENTS)),
                accent=COL_WARN,
            )
            self._draw_selector_row(
                box.centerx,
                box.top + 148,
                "P2 AGENT (PURPLE)",
                p2,
                lambda: setattr(self, "cvc_p2_idx", (self.cvc_p2_idx - 1) % len(self.COMP_AGENTS)),
                lambda: setattr(self, "cvc_p2_idx", (self.cvc_p2_idx + 1) % len(self.COMP_AGENTS)),
                accent=COL_ACCENT,
            )
            self._draw_selector_row(
                box.centerx,
                box.top + 224,
                "MATCH TIMER",
                f"{tm} Seconds",
                lambda: setattr(self, "cvc_timer_idx", (self.cvc_timer_idx - 1) % len(self.TIMER_OPTIONS)),
                lambda: setattr(self, "cvc_timer_idx", (self.cvc_timer_idx + 1) % len(self.TIMER_OPTIONS)),
                accent=COL_GOOD,
            )

            self._draw_button(
                pygame.Rect(box.centerx - 180, box.bottom - 62, 165, 40),
                "MAIN MENU",
                lambda: setattr(self, "state", "MENU"),
                accent=COL_MUTED,
            )
            self._draw_button(
                pygame.Rect(box.centerx + 15, box.bottom - 62, 165, 40),
                "START",
                self._start_cvc,
                accent=COL_GOOD,
                filled=True,
            )

        self._draw_side_card(14, 480, "ARENA RULES")
        rules = [
            "Multi-Pellet Arena: 6 golden",
            "pellets active simultaneously.",
            "",
            "Elimination:",
            "- Wall, self, or rival body hit",
            "- Head-on: longer snake wins",
            "- 200 idle steps starvation",
            "",
            "Timer Expiry:",
            "- Higher score wins the match",
            "- Tie-breaker: fewer idle steps",
        ]
        for idx, line in enumerate(rules):
            s = self.font_sm.render(line, True, COL_TEXT if line else COL_MUTED)
            self.screen.blit(s, (SIDE_RECT.left + 16, 66 + idx * 30))

        self._draw_side_card(510, 120, "DISPLAY")
        self._draw_display_toggle(564)

    def _render_1p(self):
        g = self.game_1p
        idle = getattr(g, "idle", 0)
        self._draw_bezel_and_lcd(f"SCORE: {g.score:04d}   |   IDLE: {idle:03d}/200")
        if g.food:
            self._draw_pellet(g.food[0], g.food[1], time.perf_counter() * 5.0)
        self._draw_snake(g.body, g.direction, COL_P1_HEAD, COL_P1_BODY)

        if not g.alive:
            box = self._draw_center_banner(
                "GAME OVER",
                [
                    f"Final Score: {g.score}   |   Steps: {g.steps}",
                    f"Session Best: {self.session_best_1p}",
                ],
                accent=COL_BAD,
                height_override=210,
            )
            self._draw_button(
                pygame.Rect(box.centerx - 180, box.bottom - 58, 165, 40),
                "RESTART",
                self._start_single_player,
                accent=COL_GOOD,
                filled=True,
            )
            self._draw_button(
                pygame.Rect(box.centerx + 15, box.bottom - 58, 165, 40),
                "MAIN MENU",
                lambda: setattr(self, "state", "MENU"),
                accent=COL_MUTED,
            )
        elif self.paused:
            self._draw_center_banner("PAUSED", ["Click Resume to continue"], accent=COL_GOLD)

        self._draw_side_card(14, 250, "SINGLE PLAYER VITALS")
        stats = [
            f"Score        : {g.score}",
            f"Session Best : {self.session_best_1p}",
            f"Steps Alive  : {g.steps}",
            f"Idle Counter : {idle} / 200",
            f"Snake Length : {len(g.body)}",
        ]
        for i, line in enumerate(stats):
            self.screen.blit(self.font_ui.render(line, True, COL_TEXT), (SIDE_RECT.left + 16, 66 + i * 34))

        self._draw_side_card(280, 330, "ACTIONS")
        bx = SIDE_RECT.left + 16
        bw = SIDE_RECT.width - 32
        self._draw_speed_toggle(336, h=40)
        self._draw_button(pygame.Rect(bx, 386, bw, 40), "RESUME" if self.paused else "PAUSE", lambda: setattr(self, "paused", not self.paused), accent=COL_GOLD)
        self._draw_button(pygame.Rect(bx, 436, bw, 40), "RESTART", self._start_single_player, accent=COL_GOOD)
        self._draw_display_toggle(486, h=40)
        self._draw_button(pygame.Rect(bx, 536, bw, 40), "MAIN MENU", lambda: setattr(self, "state", "MENU"), accent=COL_BAD)

    def _render_sim_replay(self):
        res = self.sim_results[self.sim_best_idx]
        body, direction, food, score, steps, idle, path = self.replay_frames[self.replay_frame_idx]
        self._draw_bezel_and_lcd(f"BEST: {res['name']} | SCORE: {score:03d} | SEED: {res['best_seed']}")

        if food:
            self._draw_pellet(food[0], food[1], time.perf_counter() * 5.0)
        self._draw_path_overlay(path, color=(255, 255, 255))
        self._draw_snake(body, direction, COL_P1_HEAD, COL_P1_BODY)

        if self.paused:
            self._draw_center_banner("REPLAY PAUSED", ["Click Resume to continue"], accent=COL_GOLD)

        self._draw_side_card(14, 280, "BEST RUN INSPECTOR")
        lines = [
            f"Agent        : {res['name']}",
            f"Best Seed    : #{res['best_seed']}",
            f"Peak Score   : {res['max_score']}",
            f"Batch Avg    : {res['avg_score']:.2f}",
            f"Nodes / Move : {res['nodes_mv']:.1f}",
            f"Frame        : {self.replay_frame_idx + 1}/{len(self.replay_frames)}",
        ]
        for i, line in enumerate(lines):
            self.screen.blit(self.font_ui.render(line, True, COL_TEXT), (SIDE_RECT.left + 16, 66 + i * 34))

        self._draw_side_card(310, 456, "REPLAY ACTIONS")
        bx = SIDE_RECT.left + 16
        bw = SIDE_RECT.width - 32
        ov_lbl = f"OVERLAY: {'ON' if self.show_overlay else 'OFF'}"
        self._draw_button(pygame.Rect(bx, 362, bw, 38), "NEXT AGENT BEST", lambda: self._load_replay(self.sim_best_idx + 1), accent=COL_ACCENT)
        self._draw_button(pygame.Rect(bx, 410, bw, 38), ov_lbl, lambda: setattr(self, "show_overlay", not self.show_overlay), accent=COL_GOLD)
        self._draw_speed_toggle(458, h=38)
        self._draw_button(pygame.Rect(bx, 506, bw, 38), "RESUME" if self.paused else "PAUSE", lambda: setattr(self, "paused", not self.paused), accent=COL_WARN)
        self._draw_button(pygame.Rect(bx, 554, bw, 38), "RESULTS & CHARTS", lambda: setattr(self, "state", "SIM_SUMMARY"), accent=COL_GOOD, filled=True)
        self._draw_button(pygame.Rect(bx, 602, bw, 38), "RESTART REPLAY", lambda: setattr(self, "replay_frame_idx", 0), accent=COL_MUTED)
        self._draw_display_toggle(650, h=38)
        self._draw_button(pygame.Rect(bx, 698, bw, 38), "MAIN MENU", lambda: setattr(self, "state", "MENU"), accent=COL_BAD)

    def _render_sim_summary(self):
        best_overall = max(self.sim_results, key=lambda r: r["avg_score"])
        self._draw_bezel_and_lcd(f"WINNER: {best_overall['name']} ({best_overall['avg_score']:.1f} AVG)")

        panel = pygame.Rect(ARENA_RECT.left + 12, ARENA_RECT.top + 12, GRID_PX - 24, GRID_PX - 24)
        pygame.draw.rect(self.screen, (11, 22, 32), panel, border_radius=12)
        pygame.draw.rect(self.screen, COL_ACCENT, panel, width=2, border_radius=12)

        available_views = [
            (idx, label, fname)
            for idx, (label, fname) in enumerate(self.SUMMARY_VIEWS)
            if fname is None or fname in self.chart_surfaces
        ]
        if not any(idx == self.summary_view_idx for idx, _, _ in available_views):
            self.summary_view_idx = 0

        tab_y = panel.top + 14
        tab_gap = 8
        tab_w = (panel.width - 32 - tab_gap * (len(available_views) - 1)) // len(available_views)
        for pos_i, (v_idx, label, _) in enumerate(available_views):
            t_rect = pygame.Rect(panel.left + 16 + pos_i * (tab_w + tab_gap), tab_y, tab_w, 32)
            is_active = v_idx == self.summary_view_idx
            self._draw_button(
                t_rect,
                label,
                lambda idx=v_idx: setattr(self, "summary_view_idx", idx),
                accent=COL_GOLD if is_active else COL_ACCENT,
                filled=is_active,
                font=self.font_xs,
            )

        _, active_file = self.SUMMARY_VIEWS[self.summary_view_idx]
        if active_file is None:
            hdr = self.font_ui.render("AGENT      AVG   STD   MED  MAX   NODES  SELF%  STRV%", True, COL_GOLD)
            self.screen.blit(hdr, (panel.left + 22, panel.top + 64))
            pygame.draw.line(self.screen, COL_CARD_BORDER, (panel.left + 18, panel.top + 90), (panel.right - 18, panel.top + 90), 1)

            for i, r in enumerate(self.sim_results):
                is_top = r["name"] == best_overall["name"]
                col = COL_GOOD if is_top else COL_TEXT
                row_str = (
                    f"{r['name']:<10} {r['avg_score']:>5.1f} {r['std_score']:>5.1f} "
                    f"{r['med_score']:>5.1f} {r['max_score']:>4d}  {r['nodes_mv']:>6.1f}  "
                    f"{r['self_pct']:>4.0f}%  {r['starve_pct']:>4.0f}%"
                )
                self.screen.blit(self.font_ui.render(row_str, True, col), (panel.left + 22, panel.top + 110 + i * 64))
        else:
            chart_surf = self.chart_surfaces[active_file]
            chart_box = pygame.Rect(panel.left + 16, panel.top + 58, panel.width - 32, panel.height - 126)
            cw, ch = chart_surf.get_size()
            scale = min(chart_box.width / cw, chart_box.height / ch)
            sw, sh = max(1, int(cw * scale)), max(1, int(ch * scale))
            scaled_chart = pygame.transform.smoothscale(chart_surf, (sw, sh))
            dst = scaled_chart.get_rect(center=chart_box.center)
            pygame.draw.rect(self.screen, (255, 255, 255), dst.inflate(6, 6), border_radius=8)
            self.screen.blit(scaled_chart, dst.topleft)

        self._draw_button(
            pygame.Rect(panel.left + 20, panel.bottom - 54, 240, 38),
            "WATCH REPLAY",
            lambda: (self._load_replay(self.sim_best_idx + 1), setattr(self, "state", "SIM_REPLAY")),
            accent=COL_ACCENT,
        )
        self._draw_button(
            pygame.Rect(panel.right - 240, panel.bottom - 54, 220, 38),
            "MAIN MENU",
            lambda: setattr(self, "state", "MENU"),
            accent=COL_BAD,
        )

        self._draw_side_card(14, 170, "SUMMARY HIGHLIGHTS")
        notes = [
            f"Winner (Avg) : {best_overall['name']}",
            f"Mean Score   : {best_overall['avg_score']:.2f}",
            f"Peak Score   : {max(r['max_score'] for r in self.sim_results)} apples",
        ]
        for i, line in enumerate(notes):
            self.screen.blit(self.font_ui.render(line, True, COL_TEXT), (SIDE_RECT.left + 16, 66 + i * 32))

        self._draw_side_card(200, 350, "ACTIONS")
        bx = SIDE_RECT.left + 16
        bw = SIDE_RECT.width - 32
        self._draw_button(
            pygame.Rect(bx, 256, bw, 42),
            "NEXT CHART / TABLE",
            lambda: setattr(self, "summary_view_idx", (self.summary_view_idx + 1) % len(self.SUMMARY_VIEWS)),
            accent=COL_GOLD,
        )
        self._draw_button(
            pygame.Rect(bx, 308, bw, 42),
            "PREV CHART / TABLE",
            lambda: setattr(self, "summary_view_idx", (self.summary_view_idx - 1) % len(self.SUMMARY_VIEWS)),
            accent=COL_ACCENT,
        )
        self._draw_button(
            pygame.Rect(bx, 360, bw, 42),
            "REPLAY BEST RUN",
            lambda: (self._load_replay(self.sim_best_idx), setattr(self, "state", "SIM_REPLAY")),
            accent=COL_GOOD,
        )
        self._draw_display_toggle(412)
        self._draw_button(
            pygame.Rect(bx, 464, bw, 42),
            "MAIN MENU",
            lambda: setattr(self, "state", "MENU"),
            accent=COL_BAD,
        )

    def _render_2p(self):
        a = self.arena_2p
        mins = int(self.match_remaining) // 60
        secs = int(self.match_remaining) % 60
        self._draw_bezel_and_lcd(f"P1: {a.p1_score:03d}   |   {mins:02d}:{secs:02d}   |   P2: {a.p2_score:03d}")

        pulse = time.perf_counter() * 5.0
        for fx, fy in a.foods:
            self._draw_pellet(fx, fy, pulse)

        if self.p1_agent:
            self._draw_path_overlay(self.p1_agent.last_path, color=(255, 220, 180))
        if self.p2_agent:
            self._draw_path_overlay(self.p2_agent.last_path, color=(220, 190, 255))

        if a.p1_alive:
            self._draw_snake(a.p1_body, a.p1_dir, COL_P1_HEAD, COL_P1_BODY)
        if a.p2_alive:
            self._draw_snake(a.p2_body, a.p2_dir, COL_P2_HEAD, COL_P2_BODY)

        if a.winner is not None:
            title = "MATCH DRAW!" if a.winner == "DRAW" else f"{a.winner} WINS!"
            accent = COL_GOLD if a.winner == "DRAW" else (COL_WARN if a.winner == "P1" else COL_ACCENT)
            box = self._draw_center_banner(
                title,
                [
                    a.end_reason,
                    f"P1 Score: {a.p1_score}   |   P2 Score: {a.p2_score}",
                ],
                accent=accent,
                height_override=210,
            )
            rematch_fn = self._start_pvc if self.state == "PLAY_PVC" else self._start_cvc
            self._draw_button(
                pygame.Rect(box.centerx - 180, box.bottom - 58, 165, 40),
                "REMATCH",
                rematch_fn,
                accent=COL_GOOD,
                filled=True,
            )
            self._draw_button(
                pygame.Rect(box.centerx + 15, box.bottom - 58, 165, 40),
                "MAIN MENU",
                lambda: setattr(self, "state", "MENU"),
                accent=COL_MUTED,
            )
        elif self.paused:
            self._draw_center_banner("MATCH PAUSED", ["Click Resume to continue"], accent=COL_GOLD)

        p1_label = "Human" if self.p1_agent is None else self.p1_agent.name
        p2_label = self.p2_agent.name

        self._draw_side_card(14, 280, "ARENA SCOREBOARD")
        lines = [
            f"P1 (Orange) : {p1_label}",
            f"  Score / Len: {a.p1_score} / {len(a.p1_body)}",
            f"  Idle Steps : {a.p1_idle} / 200",
            "",
            f"P2 (Purple) : {p2_label}",
            f"  Score / Len: {a.p2_score} / {len(a.p2_body)}",
            f"  Idle Steps : {a.p2_idle} / 200",
        ]
        for i, line in enumerate(lines):
            col = COL_WARN if i == 0 else (COL_ACCENT if i == 4 else COL_TEXT)
            self.screen.blit(self.font_sm.render(line, True, col), (SIDE_RECT.left + 16, 66 + i * 30))

        self._draw_side_card(310, 380, "MATCH ACTIONS")
        bx = SIDE_RECT.left + 16
        bw = SIDE_RECT.width - 32
        ov_lbl = f"AI OVERLAY: {'ON' if self.show_overlay else 'OFF'}"
        rematch_fn = self._start_pvc if self.state == "PLAY_PVC" else self._start_cvc
        self._draw_button(pygame.Rect(bx, 364, bw, 40), ov_lbl, lambda: setattr(self, "show_overlay", not self.show_overlay), accent=COL_GOLD)
        self._draw_speed_toggle(414, h=40)
        self._draw_button(pygame.Rect(bx, 464, bw, 40), "RESUME" if self.paused else "PAUSE", lambda: setattr(self, "paused", not self.paused), accent=COL_WARN)
        self._draw_button(pygame.Rect(bx, 514, bw, 40), "REMATCH", rematch_fn, accent=COL_GOOD)
        self._draw_display_toggle(564, h=40)
        self._draw_button(pygame.Rect(bx, 614, bw, 40), "MAIN MENU", lambda: setattr(self, "state", "MENU"), accent=COL_BAD)

    def run(self):
        self.last_tick_time = time.perf_counter()
        while True:
            now = time.perf_counter()
            dt = min(0.1, now - self.last_tick_time)
            self.last_tick_time = now

            self._handle_events()
            self._update(dt)

            self.click_targets = []
            if self.state == "MENU":
                self._render_menu()
            elif self.state == "CONTROLS":
                self._render_controls_page()
            elif self.state in ("SETUP_SIM", "SETUP_PVC", "SETUP_CVC"):
                self._render_setup()
            elif self.state == "PLAY_1P":
                self._render_1p()
            elif self.state == "SIM_REPLAY":
                self._render_sim_replay()
            elif self.state == "SIM_SUMMARY":
                self._render_sim_summary()
            elif self.state in ("PLAY_PVC", "PLAY_CVC"):
                self._render_2p()

            self._present()
            self.clock.tick(60)


def launch_gui():
    SnakeGUI().run()


if __name__ == "__main__":
    launch_gui()