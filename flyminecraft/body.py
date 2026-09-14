"""The Agent robot as the fly's body.

Senses: what surrounds the Agent in the game becomes Poisson input to real sensory neurons.
Drives: walking, homing to the player, landing and egg-laying, onto descending neurons.
Actions: the most active descending/motor neuron pool picks the Agent's next command.
No behaviour is scripted: senses and drives compete inside the connectome.
"""

import asyncio
import logging
import math
import re
from dataclasses import dataclass

from .minecraft import succeeded

log = logging.getLogger(__name__)

# Input rates (Hz). 200 Hz sugar matches Shiu et al.; the rest are the same order of magnitude.
TASTE_HZ = 200.0
TOUCH_HZ = 100.0
HEAT_HZ = 100.0
GRAVITY_HZ = 20.0   # tonic graviception: a walking fly always senses gravity
SOUND_HZ = 150.0    # the thud of bumping into something
LOOMING_HZ = 40.0   # an approaching mob: take-off with a turn away from that side

FORWARD_DRIVE_HZ = 30.0        # onto DNp09; senses must out-compete this to stop walking
HOME_DRIVE_HZ = 35.0           # onto DNa on the player's side when the fly has strayed
LAND_DRIVE_HZ = 40.0           # onto MDN when nothing is below the fly
EGG_DRIVE_HZ_PER_BLOCK = 2.0   # onto oviDNs, per block carried
EGG_DRIVE_MAX_HZ = 60.0

HOME_RANGE = 10     # blocks from the player before the homing drive starts
HOME_REST_TICKS = 3 # ticks the homing drive rests after a turn, so a 90-degree body walks between turns
MOB_RANGE = 8       # blocks within which a hostile mob looms
ITEM_RANGE = 3      # blocks within which dropped items can be tasted
FAR = 256           # reach of the side boxes used to find the player
NIGHT = range(13000, 23000)   # daytime ticks that count as night
TIME_EVERY = 20     # fly ticks between time-of-day queries

AIR = {'', 'air', 'cave_air', 'void_air'}
ORES = {'ancient_debris'}               # plus every "<something>_ore" (coal, iron, deepslate_diamond...)
WOOD = {'log', 'wood', 'stem', 'hyphae'}
HAZARDS = ('water', 'magma', 'fire', 'cactus', 'bedrock', 'barrier', 'powder_snow')
LIQUIDS = ('water', 'lava')

# Page labels for every input group, in display order
INPUT_LABELS = [
    ['drive_forward', 'Walking drive'],
    ['drive_home_left', 'Homing drive, left'],
    ['drive_home_right', 'Homing drive, right'],
    ['drive_land', 'Landing drive'],
    ['drive_egg', 'Egg-laying drive'],
    ['gravity', 'Gravity'],
    ['sugar', 'Sugar taste (ore, items)'],
    ['low_salt', 'Low-salt taste (wood)'],
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


@dataclass
class Senses:
    forward: str
    left: str
    right: str
    down: str
    bumped: bool = False      # the last move failed
    carried: int = 0          # blocks collected and not yet placed
    mob_left: bool = False    # a hostile mob within MOB_RANGE on that side
    mob_right: bool = False
    items_near: bool = False  # dropped items within ITEM_RANGE
    player: str = 'unknown'   # near | left | right | behind | ahead | unknown
    homing: str | None = None # side the homing drive steers to, or None (near, ahead or resting)
    night: bool | None = None


# Valuable ore tastes sweet, wood mildly salty, hazards bitter; plain stone and dirt have no taste.
# No smell: any food odour, even 5 Hz on either side, locks the left DNa steering neurons on for
# 30+ ticks in this connectome model, so the fly would circle left long after passing the source.
# Cold and humidity do the same, so night and water are not felt that way either.
TASTE = {'ore': 'sugar', 'wood': 'low_salt', 'hazard': 'bitter', 'lava': 'bitter'}


def home_side(player):
    """Side to steer towards the player: behind turns right, near or ahead needs no steering."""
    return {'left': 'left', 'right': 'right', 'behind': 'right'}.get(player)


def stimulus(senses):
    """Map what surrounds the Agent, and its internal drives, onto input rates."""
    rates = {'gravity': GRAVITY_HZ}
    if senses.carried:
        rates['drive_egg'] = min(EGG_DRIVE_HZ_PER_BLOCK * senses.carried, EGG_DRIVE_MAX_HZ)
    if senses.bumped:
        rates['sound'] = SOUND_HZ  # a fly that has walked into something stops walking
    else:
        rates['drive_forward'] = FORWARD_DRIVE_HZ
    ahead = classify(senses.forward)
    if ahead in TASTE:
        rates[TASTE[ahead]] = TASTE_HZ  # flies taste what their legs touch
    elif ahead == 'air' and senses.items_near:
        rates['sugar'] = TASTE_HZ       # dropped items: feeding collects them
    if any(classify(b) == 'lava' for b in (senses.forward, senses.left, senses.right, senses.down)):
        rates['heat'] = HEAT_HZ
    for side in ('left', 'right'):
        block = getattr(senses, side)
        if classify(block) != 'air' and not any(liquid in block for liquid in LIQUIDS):
            rates[f'touch_{side}'] = TOUCH_HZ
        if getattr(senses, f'mob_{side}'):
            rates[f'looming_{side}'] = LOOMING_HZ
    if senses.homing:
        rates[f'drive_home_{senses.homing}'] = HOME_DRIVE_HZ
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


def world_facts(senses):
    """[label, text] pairs describing the game around the fly, for the page."""
    if senses.mob_left and senses.mob_right:
        mobs = f'within {MOB_RANGE} blocks'
    elif senses.mob_left or senses.mob_right:
        mobs = f"within {MOB_RANGE} blocks, on the {'left' if senses.mob_left else 'right'}"
    else:
        mobs = 'none nearby'
    time = 'unknown' if senses.night is None else ('night' if senses.night else 'day')
    footing = 'in the air' if classify(senses.down) == 'air' else f"on {senses.down.replace('_', ' ')}"
    return [
        ['Player', PLAYER_TEXT[senses.player]],
        ['Hostile mobs', mobs],
        ['Dropped items', 'nearby' if senses.items_near else 'none nearby'],
        ['Footing', footing],
        ['Time', time],
    ]


ACTIONS = ('forward', 'descend', 'turn_left', 'turn_right', 'feed', 'lay_egg', 'takeoff')


def decide(motor_hz, threshold_hz):
    """Winner-take-all over motor pools (rates already baseline-subtracted)."""
    turn = motor_hz['turn_left'] - motor_hz['turn_right']
    scores = {
        'forward': (motor_hz['forward_left'] + motor_hz['forward_right']) / 2,
        'descend': motor_hz['backward'],  # MDN, the moonwalker neurons, lower the fly
        'turn_left': max(turn, 0.0),
        'turn_right': max(-turn, 0.0),
        'feed': motor_hz.get('feed', 0.0),
        'lay_egg': motor_hz['lay_egg'],
        'takeoff': motor_hz['takeoff'],
    }
    action, score = max(scores.items(), key=lambda kv: kv[1])
    return (action if score >= threshold_hz else 'idle'), scores


# The first command is the action; any others are best-effort follow-ups
COMMANDS = {
    'forward': ['agent move forward'],
    'descend': ['agent move down'],
    'turn_left': ['agent turn left'],
    'turn_right': ['agent turn right'],
    'feed': ['agent destroy forward', 'agent collect all'],
}
MOVES = ('forward', 'descend')  # take-off climbs: see AgentBody._climb

INVENTORY_SLOTS = 27
PLACE_TRIES = 6  # inventory slots tried per egg; the search resumes there on the next egg

# Bedrock yaw -> (dx, dz) of the block in front: 0 = south (+z), 90 = west, 180 = north, 270 = east.
# Education's agent inspect/detect return no block data, so blocks are read with testforblock.
HEADINGS = [(0, 1), (-1, 0), (0, -1), (1, 0)]
SIDE_OFFSET = {'ahead': 0, 'right': 1, 'behind': 2, 'left': 3}  # quarter turns clockwise from ahead
# Failure message names the block that is there, e.g. "The block at 1,2,3 is Iron Ore (expected: Air)."
FOUND_BLOCK = re.compile(r' is (.+?) \(expected', re.IGNORECASE)
NO_TARGETS = re.compile(r'no targets|not found|no entit', re.IGNORECASE)


def neighbours(position, yaw):
    """World coordinates of the blocks forward, left, right, behind and below the Agent."""
    x, y, z = position
    i = round(yaw / 90) % 4
    (fx, fz), (lx, lz) = HEADINGS[i], HEADINGS[i - 1]
    (rx, rz), (bx, bz) = HEADINGS[(i + 1) % 4], HEADINGS[(i + 2) % 4]
    return {'forward': (x + fx, y, z + fz), 'left': (x + lx, y, z + lz),
            'right': (x + rx, y, z + rz), 'down': (x, y - 1, z), 'back': (x + bx, y, z + bz)}


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
    def __init__(self, client):
        self.client = client
        self.carried = 0
        self.bumped = False
        self.position = None  # (x, y, z) from the last sense, reused as the pre-move position
        self.around = None    # neighbour coordinates from the last sense
        self._tick = 0
        self._home_rest = 0   # ticks left before the homing drive may steer again
        self._night = None
        self._egg_slot = 1
        self._slots_tried = 0  # consecutive slots that placed nothing
        self._logged = set()

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

    async def _found(self, kind, selector):
        """True if a target selector matches anything."""
        body = await self.client.command(f'testfor {selector}')
        self._log_once(f'testfor {kind}', 'testfor %s (%s) -> %s', kind, selector, body)
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

    async def sense(self):
        self._tick += 1
        self.position, yaw = await self._locate()
        if self.position is None:
            self.around = None
            return Senses('', '', '', '', bumped=self.bumped, carried=self.carried)
        self.around = neighbours(self.position, yaw)
        sides = ('forward', 'left', 'right', 'down')
        *blocks, (mob_left, mob_right), items_near, player, night = await asyncio.gather(
            *(self._block_at(*self.around[side]) for side in sides),
            self._mobs(yaw), self._items(), self._player(yaw), self._time_of_day())
        blocks = dict(zip(sides, blocks))
        self._log_once('senses', 'Agent at %s yaw %.0f senses %s', self.position, yaw, blocks)
        homing = home_side(player) if not self._home_rest else None
        return Senses(**blocks, bumped=self.bumped, carried=self.carried, mob_left=mob_left,
                      mob_right=mob_right, items_near=items_near, player=player, homing=homing,
                      night=night)

    async def act(self, action, senses):
        """Perform the action; return True if it happened in the world."""
        bumped = False
        if action == 'lay_egg':
            result = await self._lay_egg()
            ok = result == 'placed'
            if ok:
                self.carried = max(self.carried - 1, 0)
            elif result == 'empty':
                self.carried = 0  # a full pass over the inventory placed nothing
        elif action == 'takeoff':
            ok, bumped = await self._climb()
        else:
            ok = await self._perform(action, senses)
            if action == 'feed' and ok and classify(senses.forward) != 'air':
                self.carried += 1
            bumped = action in MOVES and not ok
        self.bumped = bumped
        if action in ('turn_left', 'turn_right') and senses.homing:
            self._home_rest = HOME_REST_TICKS  # walk a little before steering home again
        elif self._home_rest:
            self._home_rest -= 1
        return ok

    async def _perform(self, action, senses):
        if action == 'idle':
            return True
        if action == 'descend' and classify(senses.down) != 'air':
            return False  # a block below: the fly cannot go any lower
        lines = COMMANDS[action]
        ok = succeeded(await self.client.command(lines[0]))
        # A blocked move still reports success, so a move only counts if the Agent actually moved
        if ok and action in MOVES and self.position is not None:
            ok = await self._moved()
        for line in lines[1:]:
            await self.client.command(line)
        return ok

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

    async def _lay_egg(self):
        """Place a carried block behind the Agent: 'placed', 'blocked' (no room) or 'empty'.

        agent place reports success even when nothing is placed, so the spot behind is checked.
        """
        back = self.around['back'] if self.around else None
        if back is not None and await self._block_at(*back) != 'air':
            return 'blocked'
        for _ in range(PLACE_TRIES):
            slot = self._egg_slot
            body = await self.client.command(f'agent place {slot} back')
            if succeeded(body) and (back is None or await self._block_at(*back) != 'air'):
                self._slots_tried = 0
                self._log_once('placed', 'Laid an egg: placed slot %d at %s', slot, back)
                return 'placed'
            self._log_once('place', 'agent place %d back placed nothing: %s', slot, body)
            self._egg_slot = slot % INVENTORY_SLOTS + 1
            self._slots_tried += 1
            if self._slots_tried >= INVENTORY_SLOTS:
                self._slots_tried = 0
                return 'empty'
        return 'blocked'
