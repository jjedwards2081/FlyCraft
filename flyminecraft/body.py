"""The Agent robot as the fly's body.

Senses: what surrounds the Agent in the game becomes Poisson input to real sensory neurons.
Drives: walking, seeking the chosen block, homing to the player and landing, onto descending neurons.
Actions: the most active descending/motor neuron pool picks the Agent's next command.
No behaviour is scripted: senses and drives compete inside the connectome.
"""

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field

from .minecraft import succeeded

log = logging.getLogger(__name__)

# Input rates (Hz). 200 Hz sugar matches Shiu et al.; the rest are the same order of magnitude.
TASTE_HZ = 200.0
TOUCH_HZ = 100.0
HEAT_HZ = 100.0
GRAVITY_HZ = 20.0   # tonic graviception: a walking fly always senses gravity
SOUND_HZ = 150.0    # the thud of bumping into something
LOOMING_HZ = 40.0   # an approaching mob: take-off with a turn away from that side

FORWARD_DRIVE_HZ = 30.0   # onto DNp09; senses must out-compete this to stop walking
STEER_DRIVE_HZ = 35.0     # onto DNa on one side: towards the sought block, or back to the player
LAND_DRIVE_HZ = 40.0      # onto MDN when nothing is below the fly

HOME_RANGE = 10       # blocks from the player before the homing drive starts
STEER_REST_TICKS = 3  # ticks steering rests after a turn, so a 90-degree body walks between turns
MOB_RANGE = 8         # blocks within which a hostile mob looms
ITEM_RANGE = 3        # blocks within which dropped items can be tasted
ITEM_REST_TICKS = 10  # ticks item taste adapts after collecting found nothing to mine, so unreachable drops don't hold the fly
FAR = 256             # reach of the side boxes used to find the player
NIGHT = range(13000, 23000)   # daytime ticks that count as night
TIME_EVERY = 20       # fly ticks between time-of-day queries
SEEK_EVERY = 3        # fly ticks between scans for the sought block
SEEK_RINGS = (2, 3, 5)  # distances scanned along the eight compass directions
COLLECT_WAIT_S = 0.5    # seconds for mined blocks' drops to land before agent collect all

AIR = {'', 'air', 'cave_air', 'void_air'}
ORES = {'ancient_debris'}               # plus every "<something>_ore" (coal, iron, deepslate_diamond...)
WOOD = {'log', 'wood', 'stem', 'hyphae'}
HAZARDS = ('water', 'magma', 'fire', 'cactus', 'bedrock', 'barrier', 'powder_snow')
LIQUIDS = ('water', 'lava')
# Offsets are (right, up, ahead) from the Agent, in its own frame
FACES = {'forward': (0, 0, 1), 'back': (0, 0, -1), 'left': (-1, 0, 0), 'right': (1, 0, 0),
         'up': (0, 1, 0), 'down': (0, -1, 0)}
CONTACT = tuple(FACES)  # blocks touching the fly's faces: the ones agent destroy can mine
# Every block of the 3x3x3 cube around the Agent: its own layer, and the layers above and below
AROUND = tuple((r, u, a) for u in (1, 0, -1) for a in (1, 0, -1) for r in (-1, 0, 1) if (r, u, a) != (0, 0, 0))

# Blocks the fly can be set to seek from the page: key -> (label, block names that count, scan heights)
TARGETS = {
    'grass_block': ('Grass Block', ('grass_block', 'grass'), (-1,)),
    'sand': ('Sand', ('sand',), (-1,)),
    'oak_log': ('Oak Log', ('oak_log',), (0, 1)),
    'stone': ('Stone', ('stone',), (-1, 0, 1)),
    'coal_ore': ('Coal Ore', ('coal_ore', 'deepslate_coal_ore'), (-1, 0, 1)),
    'iron_ore': ('Iron Ore', ('iron_ore', 'deepslate_iron_ore'), (-1, 0, 1)),
    'copper_ore': ('Copper Ore', ('copper_ore', 'deepslate_copper_ore'), (-1, 0, 1)),
    'none': ('Nothing (just explore)', (), ()),
}
DEFAULT_TARGET = 'sand'

# Page labels for every input group, in display order
INPUT_LABELS = [
    ['drive_forward', 'Walking drive'],
    ['drive_seek_left', 'Seeking drive, left'],
    ['drive_seek_right', 'Seeking drive, right'],
    ['drive_home_left', 'Homing drive, left'],
    ['drive_home_right', 'Homing drive, right'],
    ['drive_land', 'Landing drive'],
    ['gravity', 'Gravity'],
    ['sugar', 'Sugar taste (sought block, items)'],
    ['bitter', 'Bitter taste (hazards)'],
    ['heat', 'Heat (lava)'],
    ['touch_left', 'Touch, left'],
    ['touch_right', 'Touch, right'],
    ['sound', 'Bump (sound)'],
    ['looming_left', 'Mob looming, left'],
    ['looming_right', 'Mob looming, right'],
]


def classify(block):
    """air | ore | wood | lava | hazard | solid (any other block: stone, dirt, sand...)."""
    name = block.removeprefix('minecraft:')
    tokens = name.split('_')
    if name in AIR:
        return 'air'
    if 'lava' in tokens:
        return 'lava'
    if any(key in name for key in HAZARDS):
        return 'hazard'
    if tokens[-1] == 'ore' or name in ORES:
        return 'ore'
    if WOOD & set(tokens):
        return 'wood'
    return 'solid'


def is_target(block, target):
    return block.removeprefix('minecraft:') in TARGETS.get(target, TARGETS['none'])[1]


def kind(block, target):
    """classify(), with the block the fly seeks marked 'target'."""
    return 'target' if is_target(block, target) else classify(block)


@dataclass
class Senses:
    forward: str = ''         # blocks touching each face (see FACES)
    back: str = ''
    left: str = ''
    right: str = ''
    up: str = ''
    down: str = ''
    cube: dict = field(default_factory=dict)  # (right, up, ahead) offset -> block, all 26 around the fly
    target: str = 'none'      # key of the block the fly seeks
    eating: tuple = ()        # contact sides holding that block
    fresh: tuple = ('left', 'right')  # sides whose contact is new since the last tick (touch adapts)
    seen: tuple | None = None # (blocks away, bearing) of the nearest one scanned
    seek: str | None = None   # side the seeking drive steers to
    bumped: bool = False      # the last move failed
    mob_left: bool = False    # a hostile mob within MOB_RANGE on that side
    mob_right: bool = False
    items_near: bool = False  # dropped items within ITEM_RANGE
    items_fresh: bool = True  # False while item taste adapts (see ITEM_REST_TICKS)
    player: str = 'unknown'   # near | left | right | behind | ahead | unknown
    homing: str | None = None # side the homing drive steers to
    night: bool | None = None


def steer_towards(where):
    """Side to turn for something on that bearing: behind turns right; near or ahead needs none."""
    return {'left': 'left', 'right': 'right', 'behind': 'right'}.get(where)


# Only the sought block tastes (sweet, so the fly eats it); plain blocks, ore and wood are just felt.
# No smell: any food odour, even 5 Hz on either side, locks the left DNa steering neurons on for
# 30+ ticks in this connectome model, so the fly would circle left long after passing the source.
# Cold and humidity do the same, so night and water are not felt that way either.
def stimulus(senses):
    """Map what surrounds the Agent, and its internal drives, onto input rates."""
    rates = {'gravity': GRAVITY_HZ}
    if senses.bumped:
        rates['sound'] = SOUND_HZ  # a fly that has walked into something stops walking
    elif not senses.eating:
        rates['drive_forward'] = FORWARD_DRIVE_HZ  # a fly tasting food stops walking to eat
    ahead = classify(senses.forward)
    if senses.eating or (ahead == 'air' and senses.items_near and senses.items_fresh):
        rates['sugar'] = TASTE_HZ  # flies taste what their legs touch: the sought block, or dropped items
    elif ahead in ('hazard', 'lava'):
        rates['bitter'] = TASTE_HZ
    if any(classify(b) == 'lava' for b in senses.cube.values()):
        rates['heat'] = HEAT_HZ  # lava anywhere in the cube around the fly radiates heat
    for side in ('left', 'right'):
        block = getattr(senses, side)
        # Touch is an obstacle sense that adapts: only a new contact, and never the food being tasted
        # (sustained touch on both sides suppresses feeding and climbing in this connectome model)
        if (side in senses.fresh and classify(block) != 'air' and not is_target(block, senses.target)
                and not any(liquid in block for liquid in LIQUIDS)):
            rates[f'touch_{side}'] = TOUCH_HZ
        if getattr(senses, f'mob_{side}'):
            rates[f'looming_{side}'] = LOOMING_HZ
    if senses.homing:
        rates[f'drive_home_{senses.homing}'] = STEER_DRIVE_HZ
    elif senses.seek:
        rates[f'drive_seek_{senses.seek}'] = STEER_DRIVE_HZ
    if classify(senses.down) == 'air' and not senses.bumped:
        rates['drive_land'] = LAND_DRIVE_HZ  # a fly still clinging to an obstacle keeps climbing
    return rates


PLAYER_TEXT = {
    'near': f'within {HOME_RANGE} blocks',
    'left': 'far away, to the left',
    'right': 'far away, to the right',
    'behind': 'far away, behind',
    'ahead': 'far away, ahead',
    'unknown': 'not found',
}
BEARING_TEXT = {'ahead': 'ahead', 'left': 'to the left', 'right': 'to the right', 'behind': 'behind'}


def world_facts(senses):
    """[label, text] pairs describing the game around the fly, for the page."""
    label = TARGETS.get(senses.target, TARGETS['none'])[0]
    if senses.target == 'none':
        seeking = label
    elif senses.eating:
        seeking = f'{label}: touching it, mining'
    elif senses.seen:
        blocks = 'block' if senses.seen[0] == 1 else 'blocks'
        seeking = f'{label}: nearest {senses.seen[0]} {blocks} away, {BEARING_TEXT[senses.seen[1]]}'
    else:
        seeking = f'{label}: none seen nearby'
    if senses.mob_left and senses.mob_right:
        mobs = f'within {MOB_RANGE} blocks'
    elif senses.mob_left or senses.mob_right:
        mobs = f"within {MOB_RANGE} blocks, on the {'left' if senses.mob_left else 'right'}"
    else:
        mobs = 'none nearby'
    time = 'unknown' if senses.night is None else ('night' if senses.night else 'day')
    footing = 'in the air' if classify(senses.down) == 'air' else f"on {senses.down.replace('_', ' ')}"
    return [
        ['Seeking', seeking],
        ['Player', PLAYER_TEXT[senses.player]],
        ['Hostile mobs', mobs],
        ['Dropped items', ('nearby' if senses.items_fresh else 'nearby, out of reach (ignored for now)')
         if senses.items_near else 'none nearby'],
        ['Footing', footing],
        ['Time', time],
    ]


ACTIONS = ('forward', 'descend', 'turn_left', 'turn_right', 'feed', 'takeoff')


def decide(motor_hz, threshold_hz):
    """Winner-take-all over motor pools (rates already baseline-subtracted)."""
    turn = motor_hz['turn_left'] - motor_hz['turn_right']
    scores = {
        'forward': (motor_hz['forward_left'] + motor_hz['forward_right']) / 2,
        'descend': motor_hz['backward'],  # MDN, the moonwalker neurons, lower the fly
        'turn_left': max(turn, 0.0),
        'turn_right': max(-turn, 0.0),
        'feed': motor_hz.get('feed', 0.0),
        'takeoff': motor_hz['takeoff'],
    }
    action, score = max(scores.items(), key=lambda kv: kv[1])
    return (action if score >= threshold_hz else 'idle'), scores


COMMANDS = {
    'forward': 'agent move forward',
    'descend': 'agent move down',
    'turn_left': 'agent turn left',
    'turn_right': 'agent turn right',
}
MOVES = ('forward', 'descend')  # take-off climbs (AgentBody._climb); feeding mines (AgentBody._eat)

# Bedrock yaw -> (dx, dz) of the block in front: 0 = south (+z), 90 = west, 180 = north, 270 = east.
# Education's agent inspect/detect return no block data, so blocks are read with testforblock.
HEADINGS = [(0, 1), (-1, 0), (0, -1), (1, 0)]
SIDE_OFFSET = {'ahead': 0, 'right': 1, 'behind': 2, 'left': 3}  # quarter turns clockwise from ahead
COMPASS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
# Failure message names the block that is there, e.g. "The block at 1,2,3 is Iron Ore (expected: Air)."
FOUND_BLOCK = re.compile(r' is (.+?) \(expected', re.IGNORECASE)
NO_TARGETS = re.compile(r'no targets|not found|no entit', re.IGNORECASE)


def neighbourhood(position, yaw):
    """World coordinates of the 26 blocks around the Agent, keyed by (right, up, ahead) offset."""
    x, y, z = position
    i = round(yaw / 90) % 4
    (fx, fz), (rx, rz) = HEADINGS[i], HEADINGS[(i + 1) % 4]
    return {(r, u, a): (x + a * fx + r * rx, y + u, z + a * fz + r * rz) for r, u, a in AROUND}


def bearing(position, yaw, point):
    """'ahead' or 'behind' (within 45 degrees), else 'left' or 'right', of a world point from the Agent."""
    i = round(yaw / 90) % 4
    (fx, fz), (lx, lz) = HEADINGS[i], HEADINGS[i - 1]
    dx, dz = point[0] - position[0], point[2] - position[2]
    along, across = dx * fx + dz * fz, dx * lx + dz * lz
    if along > 0 and abs(across) <= along:
        return 'ahead'
    if along < 0 and abs(across) <= -along:
        return 'behind'
    return 'left' if across > 0 else 'right'


def side_box(position, yaw, side, reach, height):
    """Target-selector box from 1 to `reach` blocks on one side of the Agent, `reach` wide either way."""
    x, y, z = position
    ax, az = HEADINGS[(round(yaw / 90) + SIDE_OFFSET[side]) % 4]
    if ax:
        x0, z0, dx, dz = (x + 1 if ax > 0 else x - reach), z - reach, reach - 1, 2 * reach
    else:
        x0, z0, dx, dz = x - reach, (z + 1 if az > 0 else z - reach), 2 * reach, reach - 1
    return f'x={x0},y={y - height},z={z0},dx={dx},dy={2 * height},dz={dz}'


class AgentBody:
    def __init__(self, client, seek):
        """seek: shared {'target': key of TARGETS}, changed from the page while the fly runs."""
        self.client = client
        self.seek = seek
        self.bumped = False
        self.collected = {}   # block name -> blocks mined and picked up
        self.position = None  # (x, y, z) from the last sense, reused as the pre-move position
        self.around = None    # contact coordinates from the last sense
        self._tick = 0
        self._target = None
        self._sought = None   # world coordinates of the nearest sought block last scanned
        self._next_scan = 0
        self._steer_rest = 0  # ticks left before steering may turn the fly again
        self._item_rest = 0   # ticks left before dropped items taste sweet again
        self._contacts = {}   # side -> (coordinates, block) touched last tick
        self._night = None
        self._logged = set()

    @property
    def carried(self):
        return sum(self.collected.values())

    def _log_once(self, key, message, *args):
        if key not in self._logged:
            self._logged.add(key)
            log.info(message, *args)

    async def spawn(self):
        await self.client.command('agent create')

    async def _locate(self):
        """(position, yaw) of the Agent, or (None, 0) if Minecraft does not report one."""
        body = await self.client.command('agent getposition')
        pos = body.get('position')
        if not isinstance(pos, dict):
            self._log_once('position', 'agent getposition returned no position: %s', body)
            return None, 0.0
        return tuple(math.floor(pos[k]) for k in 'xyz'), float(body.get('y-rot', 0.0))

    async def _block_at(self, x, y, z):
        body = await self.client.command(f'testforblock {x} {y} {z} air')
        if succeeded(body):
            return 'air'
        match = FOUND_BLOCK.search(body.get('statusMessage', ''))
        if not match:
            self._log_once('testforblock', 'Could not read a block name from testforblock: %s', body)
            return 'unknown'
        return match.group(1).strip().lower().replace(' ', '_')

    async def _found(self, kind_, selector):
        """True if a target selector matches anything."""
        body = await self.client.command(f'testfor {selector}')
        self._log_once(f'testfor {kind_}', 'testfor %s (%s) -> %s', kind_, selector, body)
        return succeeded(body) and not NO_TARGETS.search(body.get('statusMessage', ''))

    async def _mobs(self, yaw):
        """(left, right): hostile mobs within MOB_RANGE; one straight ahead or behind looms on both sides."""
        x, y, z = self.position
        if not await self._found('mobs', f'@e[family=monster,x={x},y={y},z={z},r={MOB_RANGE}]'):
            return False, False
        left, right = await asyncio.gather(
            *(self._found('mob side', f'@e[family=monster,{side_box(self.position, yaw, side, MOB_RANGE, 4)}]')
              for side in ('left', 'right')))
        return (left, right) if left or right else (True, True)

    async def _player(self, yaw):
        x, y, z = self.position
        if await self._found('player near', f'@a[x={x},y={y},z={z},r={HOME_RANGE}]'):
            return 'near'
        for side in ('left', 'right', 'behind', 'ahead'):
            if await self._found('player side', f'@a[{side_box(self.position, yaw, side, FAR, FAR)}]'):
                return side
        return 'unknown'

    async def _items(self):
        x, y, z = self.position
        return await self._found('items', f'@e[type=item,x={x},y={y},z={z},r={ITEM_RANGE}]')

    async def _time_of_day(self):
        """Night or day, queried every TIME_EVERY ticks."""
        if self._tick % TIME_EVERY == 1:
            body = await self.client.command('time query daytime')
            self._log_once('time', 'time query daytime -> %s', body)
            match = re.search(r'(\d+)', body.get('statusMessage', ''))
            self._night = int(match.group(1)) % 24000 in NIGHT if match else None
        return self._night

    async def _scan(self, target):
        """Nearest block of the sought kind on rings around the Agent, or None."""
        _, names, heights = TARGETS[target]
        x, y, z = self.position
        points = [(x + dx * r, y + h, z + dz * r) for r in SEEK_RINGS for dx, dz in COMPASS for h in heights]
        blocks = await asyncio.gather(*(self._block_at(*p) for p in points))
        found = [p for p, block in zip(points, blocks) if block in names]
        return min(found, key=lambda p: math.dist(p, self.position)) if found else None

    async def sense(self):
        self._tick += 1
        target = self.seek['target'] if self.seek.get('target') in TARGETS else 'none'
        if target != self._target:
            self._target, self._sought, self._next_scan = target, None, 0
        self.position, yaw = await self._locate()
        if self.position is None:
            self.around = None
            return Senses(target=target, bumped=self.bumped)
        coords = neighbourhood(self.position, yaw)
        self.around = {side: coords[offset] for side, offset in FACES.items()}
        scan = bool(TARGETS[target][1]) and self._tick >= self._next_scan
        *blocks, (mob_left, mob_right), items_near, player, night, sought = await asyncio.gather(
            *(self._block_at(*point) for point in coords.values()),
            self._mobs(yaw), self._items(), self._player(yaw), self._time_of_day(),
            self._scan(target) if scan else asyncio.sleep(0, self._sought))
        if scan:
            self._sought, self._next_scan = sought, self._tick + SEEK_EVERY
        cube = dict(zip(coords, blocks))
        # The scan starts 2 blocks out, so a sought block on a diagonal next to the fly is found here
        near = [coords[offset] for offset, block in cube.items() if is_target(block, target)]
        if near:
            self._sought = min(near, key=lambda p: math.dist(p, self.position))
        blocks = {side: cube[offset] for side, offset in FACES.items()}
        self._log_once('senses', 'Agent at %s yaw %.0f senses %s', self.position, yaw, blocks)

        eating = tuple(side for side in CONTACT if is_target(blocks[side], target))
        contacts = {side: (self.around[side], blocks[side]) for side in ('left', 'right')}
        fresh = tuple(side for side in contacts if self._contacts.get(side) != contacts[side])
        self._contacts = contacts
        steering_free = not self._steer_rest
        homing = steer_towards(player) if steering_free else None
        seen = seek = None
        if self._sought and not eating:
            where = bearing(self.position, yaw, self._sought)
            seen = (round(math.dist(self.position, self._sought)), where)
            if steering_free and not homing:
                seek = steer_towards(where)
        return Senses(**blocks, cube=cube, target=target, eating=eating, fresh=fresh, seen=seen, seek=seek, bumped=self.bumped,
                      mob_left=mob_left, mob_right=mob_right, items_near=items_near,
                      items_fresh=not self._item_rest, player=player,
                      homing=homing, night=night)

    async def act(self, action, senses):
        """Perform the action; return True if it happened in the world."""
        if action == 'feed':
            ok, bumped = await self._eat(senses), False
        elif action == 'takeoff':
            ok, bumped = await self._climb()
        else:
            ok = await self._perform(action, senses)
            bumped = action in MOVES and not ok
        self.bumped = bumped
        if action == 'feed' and not senses.eating:
            self._item_rest = ITEM_REST_TICKS  # collecting found nothing to mine: stop tasting those drops
        elif self._item_rest:
            self._item_rest -= 1
        if action in ('turn_left', 'turn_right') and (senses.homing or senses.seek):
            self._steer_rest = STEER_REST_TICKS  # walk a little before steering again
        elif self._steer_rest:
            self._steer_rest -= 1
        return ok

    async def _perform(self, action, senses):
        if action == 'idle':
            return True
        if action == 'descend' and classify(senses.down) != 'air':
            return False  # a block below: the fly cannot go any lower
        ok = succeeded(await self.client.command(COMMANDS[action]))
        # A blocked move still reports success, so a move only counts if the Agent actually moved
        if ok and action in MOVES and self.position is not None:
            ok = await self._moved()
        return ok

    async def _eat(self, senses):
        """Feeding: mine every touching block of the sought kind, then pick up what dropped."""
        mined = 0
        for side in senses.eating:
            await self.client.command(f'agent destroy {side}')
            # agent commands report success even when nothing happens, and the block breaks a moment
            # later, so wait for it to be gone before counting it
            if self.around is None or await self._gone(self.around[side]):
                block = getattr(senses, side)
                self.collected[block] = self.collected.get(block, 0) + 1
                mined += 1
                if self.around and self.around[side] == self._sought:
                    self._sought, self._next_scan = None, 0
            else:
                self._log_once('destroy', 'agent destroy %s left the %s in place', side, getattr(senses, side))
        if mined:
            await asyncio.sleep(COLLECT_WAIT_S)  # let the drops land before picking them up
        await self.client.command('agent collect all')
        return mined > 0 or not senses.eating

    async def _gone(self, point, rechecks=3, wait_s=0.25):
        """True once the block at a world point has become air."""
        for attempt in range(rechecks + 1):
            if await self._block_at(*point) == 'air':
                return True
            if attempt < rechecks:
                await asyncio.sleep(wait_s)
        return False

    async def _moved(self, rechecks=2, wait_s=0.25):
        """True once the Agent has left self.position (then updated); it may still be mid-step at first."""
        for attempt in range(rechecks + 1):
            position = (await self._locate())[0]
            if position is not None and position != self.position:
                self.position = position
                return True
            if attempt < rechecks:
                await asyncio.sleep(wait_s)
        return False

    async def _climb(self):
        """Take-off: rise one block, then step forward onto whatever was in the way.

        Returns (rose, still_blocked). A blocked step leaves the fly bumped, so it climbs again
        next tick rather than landing, and so scales walls taller than one block.
        """
        if self.position is None:
            return succeeded(await self.client.command('agent move up')), False
        if not succeeded(await self.client.command('agent move up')) or not await self._moved():
            return False, True
        stepped = succeeded(await self.client.command('agent move forward')) and await self._moved()
        return True, not stepped
