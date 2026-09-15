# FlyCraft

A whole *Drosophila* brain playing Minecraft Education. (The Python package is
called `flyminecraft`.)

The [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) leaky
integrate-and-fire model of the FlyWire connectome (138,639 neurons) runs live on
the GPU as a websocket server. Minecraft Education connects to it and the brain
drives the **Agent** robot: pick a block on the web page (grass, sand, oak logs,
stone, coal, iron or copper ore) and the fly steers towards it, mines and collects
every one it touches, climbs when it bumps into things, and comes back towards you
when it strays.

No behaviour is scripted. Senses and two labelled internal drives are injected
as Poisson spikes into real neurons, and whichever real motor neuron pool fires
most becomes the Agent's next command.

## How the fly is embodied

Each tick: the Agent inspects the 26 blocks of the 3×3×3 cube around it (its own
layer, and the layers above and below) → those senses are fed into
the brain for `--tick_ms` of brain time → the most active motor pool (above its
calibrated baseline and `--threshold_hz`) is performed.

### Inputs

| Game situation | Neurons stimulated (FlyWire annotations) |
|---|---|
| Always | Johnston's organ gravity/wind neurons, 20 Hz |
| The sought block touching a face: ahead, behind, left, right, above or below (chosen on the page) | Sugar GRNs (the 21 used by Shiu et al.), 200 Hz |
| Water, magma, fire, cactus, bedrock... or lava ahead | Bitter GRNs, 200 Hz |
| Lava anywhere in the 3×3×3 cube around the fly | Heating thermosensory neurons |
| New contact with a solid block left / right (not the sought block; touch adapts while the contact stays the same) | Head bristle mechanosensory neurons, that side |
| Any other block (stone, ore, wood, dirt...) | No taste: only touch when beside the fly |
| Last move failed (bump) | Johnston's organ auditory neurons |
| Hostile mob within 8 blocks, that side (both if ahead or behind) | Looming-sensitive LC4 and LPLC2 visual neurons, 40 Hz |
| Dropped items within 3 blocks, air ahead | Sugar GRNs, 200 Hz (feeding runs `agent collect all`); adapts for 10 ticks when collecting finds nothing to mine, so drops out of reach don't hold the fly in place |
| **Drive:** unless the last move was blocked, or it is tasting the sought block | DNp09 (P9) forward-walking neurons, 30 Hz |
| **Drive:** player more than 10 blocks away | DNa01/DNa02 steering neurons on the player's side (right if behind), 35 Hz; rests for 3 ticks after each turn so the fly walks a staircase towards the player instead of flipping left and right |
| **Drive:** nothing solid below, and not still blocked | MDN (moonwalker) neurons, 40 Hz |
| **Drive:** a sought block seen more than 45° off course (on a diagonal of the cube around the fly, or scanned every 3 ticks at 2, 3 and 5 blocks in 8 directions) | DNa01/DNa02 steering neurons on that side, 35 Hz, with the same 3-tick rest; homing to the player takes priority |

Game senses are read with `testfor` target selectors around the Agent (mobs,
dropped items, the player's side) and `time query daytime`, run concurrently with
the block checks each tick. Cold, humidity and ocellar light were tested and not
used: cold and humidity lock the left steering neurons on like smell does, and
the ocelli have no effect on the motor pools.

### Outputs

| Motor neurons | Agent command |
|---|---|
| DNp09 (P9) | `agent move forward` |
| DNa01 + DNa02, left minus right | `agent turn left` / `right` |
| MDN (moonwalker) | `agent move down`, only when the block below is air |
| Feeding motor neurons (the CB0700 pair; found by calibration as the motor neurons most driven by sugar) | mine: `agent destroy <side>` (forward, back, left, right, up or down) for every touching sought block, then `agent collect all` |
| DNp01 (giant fibre escape) | climb: `agent move up`, then `agent move forward` onto the obstacle |

Mappings: [flyminecraft/neurons.py](flyminecraft/neurons.py),
[flyminecraft/body.py](flyminecraft/body.py).

### Why there are drives

Stimulating each of the 63 sensory classes in the connectome for one second
(150 Hz) never activates forward walking (DNp09), backward walking (MDN) or egg
laying (oviDN). Senses do drive feeding (sugar, low-salt, pharyngeal taste),
turning (most odours, temperature, humidity) and giant-fibre takeoff (auditory).
Without a drive the fly only turns and eats in place, so a steady walking drive
and a carried-blocks egg-laying drive stand in for motivation. Everything else
is the brain.

### What it does (5 × 200 ms ticks per situation, full brain)

| Situation | Brain's choices |
|---|---|
| Open ground | forward 5/5 |
| Iron ore / oak log / stone ahead | feed 5/5 |
| Lava ahead | forward 4, turn left 1 |
| Water ahead | forward 5/5 |
| Ore on left / right | forward 5 / turn left 2, forward 3 |
| Stone walls left and right | forward 4, turn left 1 |
| Bumped into bedrock | forward 3, takeoff 1, turn left 1 |
| Carrying 10 / 20+ blocks | forward / lay egg 5/5 |
| Carrying 30, ore ahead | feed 5/5 |

Known quirks, straight from the connectome model: bitter taste and heat turn
the fly (~30 Hz) but rarely beat the walking drive, so it often walks into
water or lava; and the DNa steering neurons respond almost only on the left
side, so the fly turns left.

Smell is deliberately not used. Any attractive food odour (DM1/DM4 ORNs), even at
5 Hz on either side for 400 ms, switches the left DNa steering neurons on at
~30–40 Hz and they stay on for 30+ ticks after the odour stops, so the fly
circled left long after passing a log. Touch alone does not do this, and neither
do the tastes.

## Setup

Windows, Python 3.12, NVIDIA GPU (tested on an RTX 3070, 8 GB).

```powershell
git clone --recursive https://github.com/jjedwards2081/FlyCraft.git
cd FlyCraft
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
curl.exe -L -o data\flywire_neuron_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
```

If pip fails on download.pytorch.org with an SSL handshake error, download the
wheel with `curl.exe` and `pip install` the file.

## Run

```powershell
# 1. Probe the brain once: finds the feeding motor neurons, records baselines
#    (first run also builds fly-brain's ~600 MB weight cache)
.\.venv\Scripts\python.exe -m flyminecraft.calibrate

# 2. Start the server
.\.venv\Scripts\python.exe -m flyminecraft
```

Or double-click **start.bat** (in PowerShell, `.\start.bat`): it calibrates first
if needed, starts the server, and opens the dashboard once it is up. Options pass
through, e.g. `.\start.bat --seek oak_log`.

In Minecraft Education:

1. **Settings → General → turn off "Require Encrypted Websockets"**.
2. Open a world with cheats enabled.
3. Type `/connect localhost:8080` in chat.

Chat `fly pause` / `fly resume` to control it. The action bar shows what the fly
is doing and how many blocks it carries. `--debug` logs raw game messages.

### Watching the brain

Open **http://localhost:8081** while the server runs (`--dashboard_port` to change
it). Use its **Seek block** drop-down to choose what the fly looks for and mines
(Grass Block, Sand, Oak Log, Stone, Coal Ore, Iron Ore, Copper Ore, or nothing);
it starts on Sand, and `--seek` sets another choice at start. The fly never places blocks. The dark page
shows every neuron of the connectome in 3D (drag to rotate, scroll to zoom),
lighting up as it fires; how many blocks of each kind it has collected; what the
Agent senses, as a 3D model of the cube of blocks around it (drag to rotate, hover a block), and the
game facts (player, mobs, items, footing, time); the input rate into each
sensory group and drive; the motor pool scores that pick each command; and the
history of decisions. It updates every tick.

If `/connect localhost` cannot reach the server, Windows may be blocking the
app from connecting to localhost. From an administrator prompt:

```powershell
CheckNetIsolation LoopbackExempt -a -n=Microsoft.MinecraftEducationEdition_8wekyb3d8bbwe
```

### Speed

One brain simulates at about 0.13× realtime on an RTX 3070: a 200 ms tick
takes ~1.6 s, so the Agent acts roughly every 2 s. `--tick_ms 100` halves that
at the cost of noisier decisions. (CUDA graphs were tried: only 1.3× faster.)

## Layout

```
flyminecraft/
  config.py      paths; puts fly-brain/code on the import path (fly-brain is used unmodified)
  neurons.py     sensory, drive and motor neuron groups from the FlyWire annotation table
  brain.py       LiveBrain: fly-brain's PyTorch LIF model, stepped continuously
  body.py        Agent senses -> input rates; motor rates -> Agent commands
  minecraft.py   Minecraft websocket protocol (commands, events)
  calibrate.py   probe motor responses to each sense; writes data/calibration.json
  dashboard.py   live brain web page server (layout + per-tick websocket feed)
  dashboard.html the brain web page
  __main__.py    the server
fly-brain/       git submodule: eonsystemspbc/fly-brain, used unmodified (GPL-2.0-or-later)
data/            FlyWire annotations (downloaded), calibration results
```

If you cloned without `--recursive`, fetch the model with
`git submodule update --init`.

## Credits

FlyCraft is a thin layer on top of other people's science and code; the brain
itself is theirs.

- **[fly-brain](https://github.com/eonsystemspbc/fly-brain)** (GPL-2.0-or-later),
  included as a submodule. FlyCraft imports its PyTorch leaky integrate-and-fire
  model (`TorchModel`), model parameters and connectome weight loading, and the
  sugar-GRN neuron list from its sugar experiment. `flyminecraft/brain.py` only
  changes how that model is stepped (continuously, one tick at a time).
- **Shiu et al.**, *A Drosophila computational brain model reveals sensorimotor
  processing*, [Nature (2024)](https://www.nature.com/articles/s41586-024-07763-9):
  the whole-brain LIF model FlyCraft runs. Their original Brian2 code ships in
  `fly-brain/code/paper-phil-drosophila/` under the MIT License,
  copyright (c) 2023 Philip Shiu and Nico Spiller.
- **[FlyWire](https://flywire.ai/)**: the adult fly brain connectome
  (release 783), supplied with fly-brain.
- **[flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)**:
  the neuron annotation table FlyCraft uses to find sensory, drive and motor
  neurons by type and to place neurons in the 3D view.

## License

FlyCraft is licensed under the GNU General Public License version 2 or any
later version (`GPL-2.0-or-later`), the same as fly-brain, which it builds on.
See [LICENSE](LICENSE). Third-party components keep their own notices: the
Shiu et al. Brian2 materials in fly-brain remain under the MIT License, and
fly-brain's adapted NEST GPU files keep their GPL notices.
