"""Probe the connectome before playing.

Runs every sensory condition as a parallel batch and reports how each motor pool
responds. It also identifies the feeding motor neurons (MN9 is unnamed in the
FlyWire table) as the motor neurons most driven by sugar, as in Shiu et al., and
records baseline motor rates. Results go to data/calibration.json.

    python -m flyminecraft.calibrate [--duration_ms 1000] [--top_feed 2]
"""

import argparse
import json
from time import perf_counter

from . import config
from .body import GRAVITY_HZ, HEAT_HZ, ODOR_HZ, SOUND_HZ, TASTE_HZ, TOUCH_HZ
from .neurons import FEED_CANDIDATES, SENSORY_GROUPS, load_groups, select_ids

TEST_RATES = {'sugar': TASTE_HZ, 'low_salt': TASTE_HZ, 'bitter': TASTE_HZ, 'heat': HEAT_HZ,
              'touch_left': TOUCH_HZ, 'touch_right': TOUCH_HZ,
              'odor_left': ODOR_HZ, 'odor_right': ODOR_HZ, 'sound': SOUND_HZ}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--duration_ms', type=float, default=1000.0)
    parser.add_argument('--top_feed', type=int, default=2)
    args = parser.parse_args()

    from .brain import LiveBrain  # torch import is slow; keep --help fast

    sensory, motor, flyid2i, ann = load_groups()
    motor.pop('feed', None)
    for name, idx in {**sensory, **motor}.items():
        print(f'{name:14s} {len(idx):4d} neurons' + ('   <-- EMPTY' if not idx else ''))

    candidate_ids = [i for i in select_ids(ann, *FEED_CANDIDATES) if i in flyid2i]
    candidates = [flyid2i[i] for i in candidate_ids]

    conditions = {'baseline': {'gravity': GRAVITY_HZ}}
    conditions.update({name: {'gravity': GRAVITY_HZ, name: hz} for name, hz in TEST_RATES.items()})
    assert set(TEST_RATES) <= set(SENSORY_GROUPS)

    stimulated = sorted({i for idx in sensory.values() for i in idx})
    brain = LiveBrain({**sensory, **motor, 'feed_candidates': candidates}, stimulated, batch=len(conditions))
    for trial, rates in enumerate(conditions.values()):
        brain.set_input(rates, trial=trial)

    start = perf_counter()
    counts = brain.run(args.duration_ms)
    elapsed = perf_counter() - start
    print(f'\nSimulated {args.duration_ms:.0f} ms x {len(conditions)} brains in {elapsed:.1f} s '
          f'({args.duration_ms / 1000 * len(conditions) / elapsed:.2f}x realtime per brain-second)')

    rates = brain.group_rates(counts, args.duration_ms, motor)
    active = (counts.sum(dim=1) > 0).float()
    names = list(conditions)
    print('\nMotor pool rates (Hz)')
    print(f'{"condition":12s}' + ''.join(f'{m:>14s}' for m in motor) + f'{"active":>10s}')
    for t, cond in enumerate(names):
        n_active = int((counts[t] > 0).sum())
        print(f'{cond:12s}' + ''.join(f'{rates[m][t]:14.1f}' for m in motor) + f'{n_active:10d}')

    scale = 1000.0 / args.duration_ms
    cand = counts[:, brain.groups['feed_candidates']] * scale
    sugar_gain = cand[names.index('sugar')] - cand[names.index('baseline')]
    order = sugar_gain.argsort(descending=True)[:args.top_feed].tolist()
    feed_ids = [candidate_ids[i] for i in order if sugar_gain[i] > 0]
    by_id = ann.set_index('root_id')
    print('\nMotor neurons most driven by sugar (feeding candidates):')
    for i in order:
        fid = candidate_ids[i]
        row = by_id.loc[fid]
        print(f'  {fid}  {row["cell_type"]:>8s} {row["side"]:>6s}  +{float(sugar_gain[i]):.1f} Hz')
    if not feed_ids:
        print('  none responded: the fly will not feed')

    baseline = {m: rates[m][0] for m in motor}
    if feed_ids:
        feed_idx = [flyid2i[i] for i in feed_ids]
        baseline['feed'] = float(counts[0, feed_idx].mean() * scale)

    config.CALIBRATION.write_text(json.dumps({
        'duration_ms': args.duration_ms,
        'feed_ids': feed_ids,
        'baseline_hz': baseline,
        'motor_hz': {cond: {m: rates[m][t] for m in motor} for t, cond in enumerate(names)},
        'realtime_factor': args.duration_ms / 1000 * len(conditions) / elapsed,
    }, indent=2))
    print(f'\nWrote {config.CALIBRATION}')


if __name__ == '__main__':
    main()
