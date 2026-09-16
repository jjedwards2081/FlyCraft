"""Sensory and motor neuron groups of the fly, resolved to connectome indices.

Groups are selected from the FlyWire v783 annotation table
(flyconnectome/flywire_annotations, Supplemental_file1_neuron_annotations.tsv).
"""

import json

import pandas as pd

from . import config
from benchmark import EXPERIMENTS, path_comp

# Sugar gustatory receptor neurons used by Shiu et al. to evoke feeding (MN9 / proboscis extension)
SUGAR_GRN_IDS = EXPERIMENTS['sugar']['neu_exc']

ORN_ATTRACTIVE = ['ORN_DM1', 'ORN_DM4']  # Or42b / Or59b: attractive food odours

# name: (annotation column, value or values, side or None for both sides)
SENSORY_GROUPS = {
    'sugar':       ('root_id', SUGAR_GRN_IDS, None),
    'low_salt':    ('cell_sub_class', 'low-salt', None),
    'bitter':      ('cell_sub_class', 'bitter', None),
    'heat':        ('cell_sub_class', 'heating', None),
    'gravity':     ('cell_sub_class', 'wind_gravity', None),
    'touch_left':  ('cell_sub_class', 'head bristle', 'left'),
    'touch_right': ('cell_sub_class', 'head bristle', 'right'),
    'odor_left':   ('cell_type', ORN_ATTRACTIVE, 'left'),
    'odor_right':  ('cell_type', ORN_ATTRACTIVE, 'right'),
    'sound':       ('cell_sub_class', 'auditory', None),           # Johnston's organ
    # Looming-sensitive visual projection neurons: an approaching mob, on that side of the fly
    'looming_left':  ('cell_type', ['LC4', 'LPLC2'], 'left'),
    'looming_right': ('cell_type', ['LC4', 'LPLC2'], 'right'),
    # The fly's own state rather than the world: its body clock, how long it has been awake, and
    # the sleep neurons. Driving these does not quiet this model (measured; see README), so the
    # body withdraws its drives to rest the fly while these still run.
    'clock':       ('cell_type', ['l-LNv', 's-LNv_a', 's-LNv_b', 'LNd_a', 'LNd_b', 'LNd_c',
                                  'DN1a', 'DN1pA', 'DN1pB', 'DN1-l'], None),   # circadian clock
    'sleep_need':  ('cell_type', 'ER5', None),                                 # sleep pressure builds here
    # Dorsal fan-shaped body: the 23E10 sleep-promoting types that this annotation table names
    'sleep_drive': ('cell_type', ['FB6A', 'FB6C', 'FB6H', 'FB6I', 'FB6M', 'FB6V',
                                  'FB7A', 'FB7B', 'FB7K'], None),
}

# Internal drives: the only inputs that do not come from senses. No sensory class in the
# connectome drives walking, egg-laying, steering home or landing (see README), so drives
# onto the descending neurons stand in for motivation.
DRIVE_GROUPS = {
    'drive_forward':    ('cell_type', 'DNp09', None),
    'drive_egg':        ('cell_type', ['oviDNa_a', 'oviDNa_b', 'oviDNb'], None),
    'drive_home_left':  ('cell_type', ['DNa01', 'DNa02'], 'left'),    # steer back towards the player
    'drive_home_right': ('cell_type', ['DNa01', 'DNa02'], 'right'),
    'drive_seek_left':  ('cell_type', ['DNa01', 'DNa02'], 'left'),    # steer towards the sought block
    'drive_seek_right': ('cell_type', ['DNa01', 'DNa02'], 'right'),
    # Menotaxis: hold one compass heading, so the fly leaves the ground it has already searched.
    # A fly does this in its central complex, comparing a goal (FC2) against its heading (E-PG) to
    # steer through PFL3. That comparison needs the E-PG bump, and this model holds no activity at
    # all once a drive stops (measured: 0 Hz within 400 ms), so driving FC2 by side steers nothing
    # (measured: the turn it produces flips side with dose). The heading is kept in the body instead,
    # and pushed onto the same steering neurons the other drives use.
    'drive_goal_left':  ('cell_type', ['DNa01', 'DNa02'], 'left'),
    'drive_goal_right': ('cell_type', ['DNa01', 'DNa02'], 'right'),
    'drive_land':       ('cell_type', 'MDN', None),                   # nothing solid below: come down
}

MOTOR_GROUPS = {
    'forward_left':  ('cell_type', 'DNp09', 'left'),               # P9: forward walking
    'forward_right': ('cell_type', 'DNp09', 'right'),
    'turn_left':     ('cell_type', ['DNa01', 'DNa02'], 'left'),    # ipsilateral steering
    'turn_right':    ('cell_type', ['DNa01', 'DNa02'], 'right'),
    'backward':      ('cell_type', 'MDN', None),                   # moonwalker: backward walking
    'lay_egg':       ('cell_type', ['oviDNa_a', 'oviDNa_b', 'oviDNb'], None),  # oviposition
    'takeoff':       ('cell_type', 'DNp01', None),                 # giant fibre escape
}

# Candidates for the feeding motor neuron (MN9 is not named in the table); see calibrate.py
FEED_CANDIDATES = ('super_class', 'motor', None)


def load_annotations():
    columns = ['root_id', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type', 'side',
               'pos_x', 'pos_y', 'pos_z']
    return pd.read_csv(config.ANNOTATIONS, sep='\t', usecols=columns,
                       dtype={'root_id': 'int64'}, low_memory=False)


def connectome_index():
    """FlyWire root ID -> row index in the fly-brain model."""
    ids = pd.read_csv(path_comp, index_col=0).index
    return {int(fid): i for i, fid in enumerate(ids)}


def select_ids(ann, column, value, side):
    values = value if isinstance(value, (list, tuple)) else [value]
    mask = ann[column].isin(values)
    if side:
        mask &= ann['side'] == side
    return [int(i) for i in ann.loc[mask, 'root_id']]


def resolve(specs, ann, flyid2i):
    return {
        name: [flyid2i[i] for i in select_ids(ann, *spec) if i in flyid2i]
        for name, spec in specs.items()
    }


def load_groups():
    """Return (input groups, motor groups, flyid2i, annotations).

    Input groups are the sensory groups plus the internal drives.
    The 'feed' motor group only exists once calibrate.py has identified it.
    """
    ann = load_annotations()
    flyid2i = connectome_index()
    sensory = resolve({**SENSORY_GROUPS, **DRIVE_GROUPS}, ann, flyid2i)
    motor = resolve(MOTOR_GROUPS, ann, flyid2i)
    if config.CALIBRATION.exists():
        calibration = json.loads(config.CALIBRATION.read_text())
        motor['feed'] = [flyid2i[i] for i in calibration['feed_ids']]
    return sensory, motor, flyid2i, ann
