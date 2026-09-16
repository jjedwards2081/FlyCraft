"""Run the fly brain as a Minecraft Education websocket server.

    python -m flyminecraft [--port 8080] [--dashboard_port 8081] [--tick_ms 200] [--threshold_hz 5]

In Minecraft Education: Settings > General > turn off "Require Encrypted Websockets",
open a world with cheats on, then type  /connect localhost:8080  in chat.
Chat "fly pause" / "fly resume" to control it.
Watch the brain, map where the fly has been, choose the block it seeks and reset, at http://localhost:8081
"""

import argparse
import asyncio
import json
import logging
import threading
from time import perf_counter

from . import build, config
from .body import (CONTACT, DEFAULT_TARGET, INPUT_LABELS, TARGETS, TUNABLE, AgentBody, adjust, decide, facing,
                   kind, stimulus, world_blocks, world_facts)
from .dashboard import Dashboard, build_layout
from .minecraft import run_server
from .neurons import load_groups
from .worldmap import WorldMap

log = logging.getLogger('flyminecraft')

SEEK_OPTIONS = [[key, label] for key, (label, _, _) in TARGETS.items()]
FAMILIES = {key for key, _, _, _ in TUNABLE}          # what a slider or an off switch may name
GROUP_NAMES = {name for name, _ in INPUT_LABELS}      # what a poke may name
# A tick cut short by a disconnect can still be thinking in its thread when the next game starts
BRAIN_LOCK = threading.Lock()


def settings_message(settings):
    """The page's knobs as they stand, and what they can be set to."""
    return {
        'type': 'settings',
        'tick_ms': settings['tick_ms'], 'threshold_hz': settings['threshold_hz'],
        'paused': settings['paused'], 'step': settings['step'],
        'sleep': settings['sleep'], 'force_time': settings['force_time'],
        'off': sorted(settings['off']), 'tuning': dict(settings['tuning']),
        'poke': {name: poke['ticks'] for name, poke in settings['poke'].items()},
        'tunable': [[key, label, default] for key, label, default, _ in TUNABLE],
        'groups': [list(pair) for pair in INPUT_LABELS],
        'defaults': settings['defaults'],  # what "reset knobs" puts back
        'builds': [list(entry) for entry in build.BUILDS],
        'build_size': [build.MIN_SIZE, build.MAX_SIZE],
    }


def apply_settings(settings, message):
    """Take the knobs the page sent, within sane limits. Fields it leaves out are left alone."""
    def number(name, low, high):
        value = message.get(name)
        return float(min(high, max(low, value))) if isinstance(value, (int, float)) else None

    if 'tick_ms' in message and (value := number('tick_ms', 20.0, 1000.0)) is not None:
        settings['tick_ms'] = value
    if 'threshold_hz' in message and (value := number('threshold_hz', 0.0, 200.0)) is not None:
        settings['threshold_hz'] = value
    for flag in ('paused', 'sleep'):
        if isinstance(message.get(flag), bool):
            settings[flag] = message[flag]
    if 'force_time' in message and message['force_time'] in (None, 'day', 'night'):
        settings['force_time'] = message['force_time']
    if isinstance(message.get('off'), list):
        settings['off'] = [family for family in message['off'] if family in FAMILIES]
    if isinstance(message.get('tuning'), dict):
        settings['tuning'] = {family: float(min(1000.0, max(0.0, hz)))
                              for family, hz in message['tuning'].items()
                              if family in FAMILIES and isinstance(hz, (int, float))}


def chat_text(body):
    """PlayerMessage text across protocol versions."""
    props = body.get('properties', {})
    return (body.get('message') or props.get('Message') or '').strip().lower()


def think(brain, rates, tick_ms):
    """Feed in the senses and advance the brain one tick, never two at once."""
    with BRAIN_LOCK:
        brain.set_input(rates)
        return brain.run(tick_ms)


def rest(brain):
    with BRAIN_LOCK:
        brain.reset()


async def play(client, brain, motor_names, baseline, args, dashboard, seek, world_map, game, settings):
    """settings: the page's shared knobs (rates, threshold, tick length, pause, overrides, pokes)."""
    body = AgentBody(client, seek, settings)

    def status(connected):
        dashboard.publish({'type': 'status', 'connected': connected, 'paused': settings['paused'],
                           'port': args.port})

    def on_chat(event):
        text = chat_text(event)
        if text in ('fly pause', 'fly resume'):
            settings['paused'] = text == 'fly pause'
            log.info('Chat: %s', text)
            status(True)

    game.update(client=client, task=asyncio.current_task(), body=body)
    tick = 0
    try:
        await client.subscribe('PlayerMessage', on_chat)
        await body.spawn()
        log.info('Connected. The fly is alive.')
        status(True)

        while True:
            if settings['paused'] and not settings['step']:
                await asyncio.sleep(0.2)
                continue
            stepping = bool(settings['step'])
            senses = await body.sense()
            tick_ms = settings['tick_ms']
            rates = adjust(stimulus(senses), settings)

            start = perf_counter()
            counts = await asyncio.to_thread(think, brain, rates, tick_ms)
            think_s = perf_counter() - start

            firing = {name: hz[0] for name, hz in
                      brain.group_rates(counts, tick_ms, list(brain.groups)).items()}
            motor_hz = {m: firing[m] - baseline.get(m, 0.0) for m in motor_names}
            action, scores = decide(motor_hz, settings['threshold_hz'])
            ok = await body.act(action, senses)

            for name, poke in list(settings['poke'].items()):  # a poke lasts the ticks it was given
                poke['ticks'] -= 1
                if poke['ticks'] <= 0:
                    del settings['poke'][name]
            if stepping:
                settings['step'] = max(0, settings['step'] - 1)
            if stepping or settings['poke']:
                dashboard.publish(settings_message(settings))

            tick += 1
            top = ' '.join(f'{k}={v:.0f}' for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:3])
            log.info('#%d seek=%s eat=%s ahead=%s inputs=%s collected=%d -> %s%s  [%s] think=%.2fs',
                     tick, senses.target, ','.join(senses.eating) or '-', senses.forward or '?',
                     ','.join(sorted(rates)), body.carried, action, '' if ok else ' (failed)', top, think_s)

            fired, spike_counts = brain.spikes(counts)
            dashboard.publish({
                'type': 'tick', 'tick': tick, 'tick_ms': tick_ms, 'think_s': think_s,
                'senses': {side: getattr(senses, side) for side in CONTACT},
                'kinds': {side: kind(getattr(senses, side), senses.target) for side in CONTACT},
                'cube': [[*offset, block, kind(block, senses.target)] for offset, block in senses.cube.items()],
                'eating': list(senses.eating),
                'bumped': senses.bumped, 'carried': body.carried, 'collected': body.collected,
                'target': senses.target, 'world': world_facts(senses),
                'asleep': senses.asleep, 'night': senses.night, 'daytime': senses.daytime,
                'inputs': rates, 'firing': firing, 'motor_hz': motor_hz,
                'scores': scores, 'threshold_hz': settings['threshold_hz'],
                'action': action, 'ok': ok,
                'fired': fired, 'spike_counts': spike_counts,
            })
            dashboard.publish(world_map.update(senses.position, facing(senses.yaw), world_blocks(senses),
                                               body.just_mined), replay=False)
            label = TARGETS[senses.target][0]
            await client.command(f'title @s actionbar Fly: {action}  seeking {label}  collected {body.carried}')
    finally:
        game.update(client=None, task=None, body=None)
        status(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--dashboard_port', type=int, default=8081,
                        help='Port of the live brain web page')
    parser.add_argument('--tick_ms', type=float, default=200.0,
                        help='Brain time simulated per decision (ms)')
    parser.add_argument('--threshold_hz', type=float, default=5.0,
                        help='Motor pool rate above baseline needed to act')
    parser.add_argument('--seek', choices=list(TARGETS), default=DEFAULT_TARGET,
                        help='Block the fly seeks and mines at start (change it on the page)')
    parser.add_argument('--debug', action='store_true', help='Log raw Minecraft responses')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(asctime)s %(name)s %(message)s', datefmt='%H:%M:%S')
    if not args.debug:
        logging.getLogger('websockets').setLevel(logging.WARNING)

    if not config.CALIBRATION.exists():
        raise SystemExit('Run  python -m flyminecraft.calibrate  first.')
    baseline = json.loads(config.CALIBRATION.read_text())['baseline_hz']

    from .brain import LiveBrain

    sensory, motor, flyid2i, annotations = load_groups()
    stimulated = sorted({i for idx in sensory.values() for i in idx})
    log.info('Loading the connectome...')
    brain = LiveBrain({**sensory, **motor}, stimulated)
    log.info('Brain ready on %s: %d neurons', brain.device, brain.num_neurons)

    seek = {'target': args.seek}  # shared with the fly's body; the page changes it
    world_map = WorldMap()
    game = {'client': None, 'task': None, 'body': None, 'resetting': None, 'building': None}
    # The page's knobs, shared with the fly so they take effect on the next tick
    settings = {
        'tick_ms': args.tick_ms, 'threshold_hz': args.threshold_hz,
        'paused': False, 'step': 0,      # step runs that many ticks while paused
        'sleep': True, 'force_time': None,
        'off': [], 'tuning': {}, 'poke': {},
        'defaults': {'tick_ms': args.tick_ms, 'threshold_hz': args.threshold_hz},
    }

    def publish_seek():
        dashboard.publish({'type': 'seek', 'block': seek['target'], 'options': SEEK_OPTIONS})

    def publish_settings():
        dashboard.publish(settings_message(settings))

    async def reset():
        """Disconnect Minecraft and start over: an empty map, a resting brain, cleared panels."""
        try:
            log.info('Reset from the page: disconnecting Minecraft, clearing the map and resting the brain')
            client, task = game['client'], game['task']
            if client:
                await client.close()
                await asyncio.wait({task})  # run_server stops the fly once the game has gone
            await asyncio.to_thread(rest, brain)
            world_map.reset()
            dashboard.reset()
            dashboard.publish(world_map.snapshot(), replay=False)
            log.info('Reset done. Type  /connect localhost:%d  in the game chat to start again', args.port)
        finally:
            game['resetting'] = None

    async def run_build(kind_, size):
        """Build test ground around the Agent. This changes blocks in the player's world."""
        def note(state, text):
            dashboard.publish({'type': 'build', 'state': state, 'kind': kind_, 'note': text}, replay=False)

        try:
            client, body = game['client'], game['body']
            if client is None or body is None or body.position is None:
                note('idle', 'Minecraft is not connected, so there is nothing to build on')
                return
            names = TARGETS.get(seek['target'], TARGETS['none'])[1]
            lines = build.commands(kind_, body.position, size, block=names[0] if names else 'sand')
            log.info('Building %s (%d blocks across) around %s: %d commands',
                     kind_, size, body.position, len(lines))
            note('running', f'building {kind_}: {len(lines)} commands')
            for line in lines:
                await client.command(line)
            log.info('Built %s', kind_)
            note('done', f'built {kind_} around the fly')
        except ConnectionError:
            note('idle', 'Minecraft went away while building')
        finally:
            game['building'] = None

    def on_page_message(message):
        if message.get('type') == 'seek' and message.get('block') in TARGETS:
            seek['target'] = message['block']
            log.info('Now seeking: %s', TARGETS[seek['target']][0])
            publish_seek()
        elif message.get('type') == 'reset' and not game['resetting']:
            game['resetting'] = asyncio.create_task(reset())
        elif message.get('type') == 'settings':
            apply_settings(settings, message)
            log.info('From the page: %s', ', '.join(f'{k}={settings[k]}' for k in message if k in settings))
            publish_settings()
        elif message.get('type') == 'step':
            settings['step'] = min(50, settings['step'] + 1)
            publish_settings()
        elif message.get('type') == 'build' and message.get('kind') in build.BUILD_KEYS:
            if game['building']:
                log.info('Already building; ignoring %s', message['kind'])
            else:
                size = message.get('size', 15)
                game['building'] = asyncio.create_task(
                    run_build(message['kind'], size if isinstance(size, (int, float)) else 15))
        elif message.get('type') == 'poke' and message.get('group') in GROUP_NAMES:
            hz, ticks = message.get('hz'), message.get('ticks')
            if isinstance(hz, (int, float)) and isinstance(ticks, (int, float)):
                settings['poke'][message['group']] = {'hz': float(min(1000.0, max(0.0, hz))),
                                                      'ticks': int(min(100, max(1, ticks)))}
                log.info('Poking %s at %.0f Hz for %d ticks', message['group'],
                         settings['poke'][message['group']]['hz'],
                         settings['poke'][message['group']]['ticks'])
                publish_settings()

    layout = build_layout(annotations, flyid2i, brain.num_neurons, sensory, motor, INPUT_LABELS)
    # Settings reach a new page through the replay of the last published one, so on_open adds only the map
    dashboard = Dashboard(layout, on_message=on_page_message, on_open=lambda: [world_map.snapshot()])
    publish_seek()
    publish_settings()
    dashboard.publish({'type': 'status', 'connected': False, 'paused': False, 'port': args.port})

    async def serve():
        await asyncio.gather(
            dashboard.serve(args.host, args.dashboard_port),
            run_server(args.host, args.port,
                       lambda client: play(client, brain, list(motor), baseline, args, dashboard, seek,
                                           world_map, game, settings)),
        )

    asyncio.run(serve())


if __name__ == '__main__':
    main()
