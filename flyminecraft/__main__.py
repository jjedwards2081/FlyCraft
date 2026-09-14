"""Run the fly brain as a Minecraft Education websocket server.

    python -m flyminecraft [--port 8080] [--dashboard_port 8081] [--tick_ms 200] [--threshold_hz 5]

In Minecraft Education: Settings > General > turn off "Require Encrypted Websockets",
open a world with cheats on, then type  /connect localhost:8080  in chat.
Chat "fly pause" / "fly resume" to control it.
Watch the brain at http://localhost:8081
"""

import argparse
import asyncio
import json
import logging
from time import perf_counter

from . import config
from .body import INPUT_LABELS, AgentBody, classify, decide, stimulus, world_facts
from .dashboard import Dashboard, build_layout
from .minecraft import run_server
from .neurons import load_groups

log = logging.getLogger('flyminecraft')

SIDES = ('forward', 'left', 'right', 'down')


def chat_text(body):
    """PlayerMessage text across protocol versions."""
    props = body.get('properties', {})
    return (body.get('message') or props.get('Message') or '').strip().lower()


async def play(client, brain, motor_names, baseline, args, dashboard):
    body = AgentBody(client)
    state = {'paused': False}

    def status(connected):
        dashboard.publish({'type': 'status', 'connected': connected, 'paused': state['paused']})

    def on_chat(event):
        text = chat_text(event)
        if text in ('fly pause', 'fly resume'):
            state['paused'] = text == 'fly pause'
            log.info('Chat: %s', text)
            status(True)

    await client.subscribe('PlayerMessage', on_chat)
    await body.spawn()
    log.info('Connected. The fly is alive.')
    status(True)

    tick = 0
    try:
        while True:
            if state['paused']:
                await asyncio.sleep(0.5)
                continue
            senses = await body.sense()
            rates = stimulus(senses)
            brain.set_input(rates)

            start = perf_counter()
            counts = await asyncio.to_thread(brain.run, args.tick_ms)
            think_s = perf_counter() - start

            firing = {name: hz[0] for name, hz in
                      brain.group_rates(counts, args.tick_ms, list(brain.groups)).items()}
            motor_hz = {m: firing[m] - baseline.get(m, 0.0) for m in motor_names}
            action, scores = decide(motor_hz, args.threshold_hz)
            ok = await body.act(action, senses)

            tick += 1
            top = ' '.join(f'{k}={v:.0f}' for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:3])
            log.info('#%d ahead=%s inputs=%s carried=%d -> %s%s  [%s] think=%.2fs',
                     tick, senses.forward or '?', ','.join(sorted(rates)), body.carried, action,
                     '' if ok else ' (failed)', top, think_s)

            fired, spike_counts = brain.spikes(counts)
            dashboard.publish({
                'type': 'tick', 'tick': tick, 'tick_ms': args.tick_ms, 'think_s': think_s,
                'senses': {side: getattr(senses, side) for side in SIDES},
                'kinds': {side: classify(getattr(senses, side)) for side in SIDES},
                'bumped': senses.bumped, 'carried': body.carried, 'world': world_facts(senses),
                'inputs': rates, 'firing': firing, 'motor_hz': motor_hz,
                'scores': scores, 'threshold_hz': args.threshold_hz,
                'action': action, 'ok': ok,
                'fired': fired, 'spike_counts': spike_counts,
            })
            await client.command(f'title @s actionbar Fly: {action}  carrying {body.carried}  ({top} Hz)')
    finally:
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
    dashboard = Dashboard(build_layout(annotations, flyid2i, brain.num_neurons, sensory, motor, INPUT_LABELS))

    async def serve():
        await asyncio.gather(
            dashboard.serve(args.host, args.dashboard_port),
            run_server(args.host, args.port,
                       lambda client: play(client, brain, list(motor), baseline, args, dashboard)),
        )

    asyncio.run(serve())


if __name__ == '__main__':
    main()
