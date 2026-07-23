"""Stylish top-down renderer for the Deep Ocean arena.

A neon, minimal aesthetic: a glowing circular arena on a deep radial-gradient
sea, fish drawn as little arrowheads with fading trails, and a sleek dark shark.
Works both on-screen (``human``) and off-screen (``rgb_array``) for recording.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from . import dynamics as dyn

_BG_INNER = (20, 27, 44)
_BG_OUTER = (5, 8, 15)
_ARENA = (0, 232, 224)
_SHARK = (226, 232, 244)
_SHARK_CORE = (120, 130, 150)
_EYE = (255, 78, 78)
_TEXT = (210, 232, 240)
_FISH_PALETTE = [
    (0, 232, 224), (255, 92, 176), (150, 255, 120), (255, 186, 72),
    (140, 150, 255), (120, 240, 255), (255, 128, 120), (200, 140, 255),
]


class Renderer:
    def __init__(self, mode: str, size: int = 720) -> None:
        self.mode = mode
        self.size = size
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Menlo,Consolas,monospace", 20, bold=True)
        if mode == "human":
            self.surface = pygame.display.set_mode((size, size))
            pygame.display.set_caption("Deep Ocean")
            self.clock = pygame.time.Clock()
        else:
            self.surface = pygame.Surface((size, size))
            self.clock = None

        self.margin = 40
        self.scale = (size / 2 - self.margin) / dyn.ARENA_RADIUS
        self.center = np.array([size / 2, size / 2])
        self._bg = self._make_background()
        self._trails: dict[int, deque] = {}

    # -- setup -------------------------------------------------------------
    def _make_background(self) -> pygame.Surface:
        bg = pygame.Surface((self.size, self.size))
        cx = cy = self.size / 2
        maxd = np.hypot(cx, cy)
        step = 4
        for r in range(int(maxd), 0, -step):
            t = r / maxd
            col = [int(_BG_OUTER[i] * t + _BG_INNER[i] * (1 - t)) for i in range(3)]
            pygame.draw.circle(bg, col, (int(cx), int(cy)), r)
        return bg

    def _to_screen(self, p: np.ndarray) -> tuple[int, int]:
        s = self.center + np.array([p[0], -p[1]]) * self.scale
        return int(s[0]), int(s[1])

    @staticmethod
    def _heading(vel: np.ndarray) -> float:
        if float(np.linalg.norm(vel)) < 1e-4:
            return np.pi / 2
        return float(np.arctan2(vel[1], vel[0]))

    # -- drawing -----------------------------------------------------------
    def draw(self, fish, shark, hud: str = ""):
        if self.mode == "human":
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()

        self.surface.blit(self._bg, (0, 0))
        self._draw_arena()

        for i, (pos, vel, alive) in enumerate(fish):
            if not alive:
                self._trails.pop(i, None)
                continue
            color = _FISH_PALETTE[i % len(_FISH_PALETTE)]
            self._draw_trail(i, pos, color)
            self._draw_arrow(pos, vel, color, 13)

        self._draw_shark(*shark)

        if hud:
            self.surface.blit(self.font.render(hud, True, _TEXT), (18, 14))
            alive_n = sum(1 for _, _, a in fish if a)
            if len(fish) > 1:
                tag = self.font.render(f"fish {alive_n}/{len(fish)}", True, _TEXT)
                self.surface.blit(tag, (18, 40))

        if self.mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(30)
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def _draw_arena(self) -> None:
        c = (int(self.center[0]), int(self.center[1]))
        r = int(dyn.ARENA_RADIUS * self.scale)
        for w, a in ((7, 22), (4, 45), (2, 130)):        # outward glow -> crisp edge
            ring = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*_ARENA, a), c, r, w)
            self.surface.blit(ring, (0, 0))

    def _draw_trail(self, idx: int, pos: np.ndarray, color) -> None:
        tr = self._trails.setdefault(idx, deque(maxlen=14))
        tr.append(self._to_screen(pos))
        n = len(tr)
        for j in range(1, n):
            a = int(120 * j / n)
            surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            pygame.draw.line(surf, (*color, a), tr[j - 1], tr[j], 2)
            self.surface.blit(surf, (0, 0))

    def _draw_arrow(self, pos, vel, color, size) -> None:
        h = self._heading(vel)
        cos_h, sin_h = np.cos(h), np.sin(h)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        shape = np.array([[1.0, 0.0], [-0.6, 0.55], [-0.3, 0.0], [-0.6, -0.55]]) * size
        pts = [self._to_screen(pos + (rot @ p) / self.scale) for p in shape]
        glow = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.polygon(glow, (*color, 70), pts)
        self.surface.blit(glow, (0, 0))
        pygame.draw.polygon(self.surface, color, pts)

    def _draw_shark(self, pos, vel) -> None:
        h = self._heading(vel)
        cos_h, sin_h = np.cos(h), np.sin(h)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        body = np.array([[1.9, 0.0], [0.4, 0.75], [-1.7, 0.35],
                         [-2.3, 0.0], [-1.7, -0.35], [0.4, -0.75]]) * 15
        pts = [self._to_screen(pos + (rot @ p) / self.scale) for p in body]
        glow = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.polygon(glow, (*_EYE, 40), pts)
        self.surface.blit(glow, (0, 0))
        pygame.draw.polygon(self.surface, _SHARK_CORE, pts)
        pygame.draw.polygon(self.surface, _SHARK, pts, 2)
        eye = pos + (rot @ np.array([1.1, 0.28])) / self.scale
        pygame.draw.circle(self.surface, _EYE, self._to_screen(eye), 3)

    def close(self) -> None:
        pygame.quit()
