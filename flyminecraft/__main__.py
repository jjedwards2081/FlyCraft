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

from . import config
from .body import (CONTACT, DEFAULT_TARGET, INPUT_LABELS, TARGETS, AgentBody, decide, facing, kind, stimulus,
                   world_blocks, world_facts)
from .dashboard import Dashboard, build_layout
from .minecraft import run_server
from .neurons import load_groups
from .worldmap import WorldMap

log = logging.getLogger('flyminecraft')

SEEK_OPTIONS = [[key, label] for key, (label, _, _) in TARGETS.items()]
# A tick cut short by a disconnect can still be thinking in its thread when the next game starts
BRAIN_LOCK = threading.Lock()


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


async def play(client, brain, motor_names, baseline, args, dashboard, seek, world_map, game):
    """game: shared {'client', 'task'} of the connected game, so the page can disconnect it."""
    body = AgentBody(client, seek)
    state = {'paused': False}

    def status(connected):
        dashboard.publish({'type': 'status', 'connected': connected, 'paused': state['paused'], 'port': args.port})

    def on_chat(event):
        text = chat_text(event)
        if text in ('fly pause', 'fly resume'):
            state['paused'] = text == 'fly pause'
            log.info('Chat: %s', text)
            status(True)

    game.update(client=client, task=asyncio.current_task())
    tick = 0
    try:
        await client.subscribe('PlayerMessage', on_chat)
        await body.spawn()
        log.info('Connected. The fly is alive.')
        status(True)

        while True:
            if state['paused']:
                await asyncio.sleep(0.5)
                continue
            senses = await body.sense()
            rates = stimulus(senses)

            start = perf_counter()
            counts = await asyncio.to_thread(think, brain, rates, args.tick_ms)
            think_s = perf_counter() - start

            firing = {name: hz[0] for name, hz in
                      brain.group_rates(counts, args.tick_ms, list(brain.groups)).items()}
            motor_hz = {m: firing[m] - baseline.get(m, 0.0) for m in motor_names}
            action, scores = decide(motor_hz, args.threshold_hz)
            ok = await body.act(action, senses)

            tick += 1
            top = ' '.join(f'{k}={v:.0f}' for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:3])
            log.info('#%d seek=%s eat=%s ahead=%s inputs=%s collected=%d -> %s%s  [%s] think=%.2fs',
                     tick, senses.target, ','.join(senses.eating) or '-', senses.forward or '?',
                     ','.join(sorted(rates)), body.carried, action, '' if ok else ' (failed)', top, think_s)

            fired, spike_counts = brain.spikes(counts)
            dashboard.publish({
                'type': 'tick', 'tick': tick, 'tick_ms': args.tick_ms, 'think_s': think_s,
                'senses': {side: getattr(senses, side) for side in CONTACT},
                'kinds': {side: kind(getattr(senses, side), senses.target) for side in CONTACT},
                'cube': [[*offset, block, kind(block, senses.target)] for offset, block in senses.cube.items()],
                'eating': list(senses.eating),
                'bumped': senses.bumped, 'carried': body.carried, 'collected': body.collected,
                'target': senses.target, 'world': world_facts(senses),
                'asleep': senses.asleep, 'night': senses.night, 'daytime': senses.daytime,
                'inputs': rates, 'firing': firing, 'motor_hz': motor_hz,
                'scores': scores, 'threshold_hz': args.threshold_hz,
                'action': action, 'ok': ok,
                'fired': fired, 'spike_counts': spike_counts,
            })
            dashboard.publish(world_map.update(senses.position, facing(senses.yaw), world_blocks(senses),
                                               body.just_mined), replay=False)
            label = TARGETS[senses.target][0]
            await client.command(f'title @s actionbar Fly: {action}  seeking {label}  collected {body.carried}')
    finally:
        game.update(client=None, task=None)
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
    game = {'client': None, 'task': None, 'resetting': None}

    def publish_seek():
        dashboard.publish({'type': 'seek', 'block': seek['target'], 'options': SEEK_OPTIONS})

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

    def on_page_message(message):
        if message.get('type') == 'seek' and message.get('block') in TARGETS:
            seek['target'] = message['block']
            log.info('Now seeking: %s', TARGETS[seek['target']][0])
            publish_seek()
        elif message.get('type') == 'reset' and not game['resetting']:
            game['resetting'] = asyncio.create_task(reset())

    layout = build_layout(annotations, flyid2i, brain.num_neurons, sensory, motor, INPUT_LABELS)
    dashboard = Dashboard(layout, on_message=on_page_message, on_open=lambda: [world_map.snapshot()])
    publish_seek()
    dashboard.publish({'type': 'status', 'connected': False, 'paused': False, 'port': args.port})

    async def serve():
        await asyncio.gather(
            dashboard.serve(args.host, args.dashboard_port),
            run_server(args.host, args.port,
                       lambda client: play(client, brain, list(motor), baseline, args, dashboard, seek,
                                           world_map, game)),
        )

    asyncio.run(serve())


if __name__ == '__main__':
    main()
