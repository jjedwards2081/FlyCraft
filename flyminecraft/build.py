"""Ground to experiment on: a flat arena, a walled one, a maze, a patch to mine.

Each build is a few `fill` commands around the Agent, bounded to a small square so it cannot
run away with somebody's world. The functions only make the commands; the server runs them.
"""

import random

MIN_SIZE, MAX_SIZE = 5, 31  # blocks across, always centred on the Agent
HEADROOM = 4                # blocks of air above the floor
FLOOR = 'stone'
WALL = 'stone'

# key -> (label, what it does), for the page
BUILDS = [
    ['flat', 'Flat arena', 'Clears a square and floors it, so the fly walks on nothing but stone'],
    ['walled', 'Walled arena', 'A flat arena inside a wall, so the fly cannot wander out of it'],
    ['maze', 'Maze', 'Corridors one block wide around the fly, to watch it find its way'],
    ['patch', 'Patch to mine', 'Lays a patch of the block it seeks into the floor'],
    ['clear', 'Clear the air', 'Empties the air above the floor, leaving the ground as it is'],
]
BUILD_KEYS = [key for key, _, _ in BUILDS]


def square(position, size):
    """(x0, z0, x1, z1) of a square of `size` blocks centred on the Agent."""
    half = max(MIN_SIZE, min(MAX_SIZE, int(size))) // 2
    x, _, z = position
    return x - half, z - half, x + half, z + half


def fill(x0, y0, z0, x1, y1, z1, block):
    return f'fill {x0} {y0} {z0} {x1} {y1} {z1} {block}'


def maze_walls(position, size, seed=None):
    """(x0, z0, cells, open squares) of a maze whose corridors fall on the Agent's own square.

    Cells sit two blocks apart with a wall between them; a recursive backtracker knocks walls
    out until every cell is reachable, so there is always a way from the fly to anywhere in it.
    """
    cells = max(2, (min(MAX_SIZE, max(MIN_SIZE, int(size))) - 1) // 2)
    x, _, z = position
    x0, z0 = x - 2 * (cells // 2), z - 2 * (cells // 2)
    start = (cells // 2, cells // 2)  # the Agent's own square

    rng = random.Random(seed)
    seen = {start}
    opened = {(x0 + 2 * start[0], z0 + 2 * start[1])}
    stack = [start]
    while stack:
        i, j = stack[-1]
        nexts = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if 0 <= i + di <= cells and 0 <= j + dj <= cells and (i + di, j + dj) not in seen]
        if not nexts:
            stack.pop()
            continue
        step = rng.choice(nexts)
        seen.add(step)
        opened.add((x0 + 2 * step[0], z0 + 2 * step[1]))                      # the square itself
        opened.add((x0 + i + step[0], z0 + j + step[1]))                      # the wall between
        stack.append(step)
    return x0, z0, cells, sorted(opened)


def commands(kind, position, size, block='sand', seed=None):
    """The `fill` commands for one build, around the Agent at `position`."""
    x, y, z = position
    x0, z0, x1, z1 = square(position, size)
    floor, head = y - 1, y + HEADROOM

    if kind == 'clear':
        return [fill(x0, y, z0, x1, head, z1, 'air')]
    if kind == 'flat':
        return [fill(x0, floor, z0, x1, floor, z1, FLOOR),
                fill(x0, y, z0, x1, head, z1, 'air')]
    if kind == 'walled':
        return [fill(x0, floor, z0, x1, floor, z1, FLOOR),
                fill(x0, y, z0, x1, head, z1, 'air'),
                fill(x0, y, z0, x0, y + 2, z1, WALL),
                fill(x1, y, z0, x1, y + 2, z1, WALL),
                fill(x0, y, z0, x1, y + 2, z0, WALL),
                fill(x0, y, z1, x1, y + 2, z1, WALL)]
    if kind == 'patch':
        half = max(1, min(MAX_SIZE, max(MIN_SIZE, int(size))) // 4)
        return [fill(x - half, floor, z - half, x + half, floor, z + half, block)]
    if kind == 'maze':
        mx0, mz0, cells, opened = maze_walls(position, size, seed)
        mx1, mz1 = mx0 + 2 * cells, mz0 + 2 * cells
        built = [fill(mx0, floor, mz0, mx1, floor, mz1, FLOOR),      # floor
                 fill(mx0, y, mz0, mx1, y + 2, mz1, WALL),           # solid, then cut corridors
                 fill(mx0, y + 3, mz0, mx1, head, mz1, 'air')]
        built += [fill(cx, y, cz, cx, y + 2, cz, 'air') for cx, cz in opened]
        return built
    raise ValueError(f'unknown build: {kind}')
