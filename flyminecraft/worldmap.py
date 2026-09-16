"""What the fly has found out about the world: where it has been, the terrain it sensed, the blocks it mined."""

from .body import classify

MAX_TRACE = 50000    # route points kept; the oldest are dropped first
MAX_BLOCKS = 200000  # terrain blocks kept; blocks found beyond this are not mapped


class WorldMap:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trace = []   # [x, y, z] of each block the fly stood in, oldest first, without repeats
        self.blocks = {}  # (x, y, z) -> name of every non-air block sensed, as last sensed
        self.mined = []   # [x, y, z, block] mined, oldest first
        self.fly = None   # [x, y, z, dx, dz]: where the fly is, and the direction it faces

    def update(self, position, forward, sensed, mined):
        """Record one tick, and return the change as a page message.

        position: the fly's (x, y, z), or None if unknown; forward: (dx, dz) it faces
        sensed: (x, y, z) -> block around the fly; mined: [((x, y, z), block)] this tick
        Pages apply 'blocks' before 'cleared': a block mined this tick is in both.
        """
        step, changed, cleared = None, [], []
        sensed = dict(sensed)
        if position is not None:
            sensed[position] = 'air'  # the fly's own block
            self.fly = [*position, *forward]
            if not self.trace or self.trace[-1] != list(position):
                step = list(position)
                self.trace.append(step)
                del self.trace[:-MAX_TRACE]
        for point, block in [*sensed.items(), *((point, 'air') for point, _ in mined)]:
            if block == 'unknown':
                continue
            if classify(block) == 'air':
                if self.blocks.pop(point, None) is not None:
                    cleared.append(list(point))
            elif self.blocks.get(point) != block and (point in self.blocks or len(self.blocks) < MAX_BLOCKS):
                self.blocks[point] = block
                changed.append([*point, block])
        new_mined = [[*point, block] for point, block in mined]
        self.mined.extend(new_mined)
        return {'type': 'map_step', 'step': step, 'fly': self.fly,
                'blocks': changed, 'cleared': cleared, 'mined': new_mined}

    def snapshot(self):
        """The whole map, as a page message."""
        return {'type': 'map', 'trace': self.trace, 'blocks': [[*point, block] for point, block in self.blocks.items()],
                'mined': self.mined, 'fly': self.fly, 'max_trace': MAX_TRACE}
