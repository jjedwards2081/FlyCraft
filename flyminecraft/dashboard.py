"""Live web view of the fly brain, served next to the Minecraft server.

    http://localhost:8081

GET /             the page (dashboard.html)
GET /layout.json  every neuron's projected position and super class, and the input/motor groups
GET /logo.png     the Minecraft Education logo shown in the page header
WS  /ws           one JSON message per brain tick, plus connection status
"""

import json
import logging
from collections import deque
from http import HTTPStatus
from pathlib import Path

import numpy as np
from websockets.asyncio.server import broadcast, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name('dashboard.html')
LOGO = Path(__file__).with_name('minecraft-education-logo.png')  # Minecraft Education, by Mojang/Microsoft
MAP_SIZE = 1000  # positions are scaled so the longer axis spans 0..MAP_SIZE
HISTORY = 60     # recent tick summaries replayed to pages opened mid-session


def build_layout(annotations, flyid2i, num_neurons, inputs, motors, input_labels=None):
    """Brain coordinates (x right, y down, z depth) for every neuron in the connectome.

    input_labels: ordered [group name, label] pairs for the page's input bars.
    """
    ann = annotations.dropna(subset=['pos_x', 'pos_y', 'pos_z'])
    ann = ann[ann['root_id'].isin(flyid2i.keys())].drop_duplicates('root_id')
    rows = ann['root_id'].map(flyid2i).to_numpy()
    # FlyWire positions are in 4 x 4 x 40 nm voxels, so depth is scaled to the same units
    px, py, pz = ann['pos_x'].to_numpy(), ann['pos_y'].to_numpy(), ann['pos_z'].to_numpy() * 10
    span = max(np.ptp(px), np.ptp(py), np.ptp(pz))

    x = np.full(num_neurons, -1, dtype=np.int32)
    y = np.full(num_neurons, -1, dtype=np.int32)
    z = np.full(num_neurons, -1, dtype=np.int32)
    x[rows] = np.round((px - px.min()) / span * MAP_SIZE)
    y[rows] = np.round((py - py.min()) / span * MAP_SIZE)
    z[rows] = np.round((pz - pz.min()) / span * MAP_SIZE)

    classes = sorted(ann['super_class'].dropna().unique())
    cls = np.full(num_neurons, -1, dtype=np.int32)
    cls[rows] = ann['super_class'].map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int)

    return {
        'num_neurons': num_neurons,
        'width': int(x.max()) + 1,
        'height': int(y.max()) + 1,
        'depth': int(z.max()) + 1,
        'classes': classes,
        'x': x.tolist(),   # -1 for neurons without a position
        'y': y.tolist(),
        'z': z.tolist(),
        'cls': cls.tolist(),
        'inputs': inputs,   # group name -> neuron indices
        'motors': motors,
        'input_labels': [list(pair) for pair in (input_labels or [])],
    }


class Dashboard:
    def __init__(self, layout, on_message=None, on_open=None):
        """on_message(dict) receives JSON objects sent by pages, e.g. the chosen block to seek.

        on_open() returns messages that bring a newly opened page up to date, e.g. the whole map.
        """
        self._on_message = on_message
        self._on_open = on_open
        self._page = PAGE.read_bytes()
        self._logo = LOGO.read_bytes() if LOGO.exists() else b''
        self._layout = json.dumps(layout, separators=(',', ':')).encode()
        self._clients = set()
        self._latest = {}  # message type -> last message, replayed to pages that open later
        self._history = deque(maxlen=HISTORY)

    def publish(self, message, replay=True):
        """Send to every page. replay=False for changes that only apply on top of on_open's messages."""
        text = json.dumps(message, separators=(',', ':'))
        if message['type'] == 'tick':
            summary = {k: v for k, v in message.items() if k not in ('fired', 'spike_counts')}
            summary['active'] = len(message['fired'])
            self._history.append(summary)
        if replay:
            self._latest[message['type']] = text
        broadcast(self._clients, text)

    def reset(self):
        """Forget the ticks so far, and tell pages to clear them."""
        self._history.clear()
        self._latest.pop('tick', None)
        self.publish({'type': 'reset'}, replay=False)

    def _http(self, connection, request):
        path = request.path.split('?', 1)[0]
        if path == '/ws':
            return None  # continue the websocket handshake
        if path == '/':
            body, kind = self._page, 'text/html; charset=utf-8'
        elif path == '/layout.json':
            body, kind = self._layout, 'application/json'
        elif path == '/logo.png' and self._logo:
            body, kind = self._logo, 'image/png'
        else:
            return connection.respond(HTTPStatus.NOT_FOUND, 'Not found\n')
        headers = Headers([('Content-Type', kind), ('Content-Length', str(len(body))),
                           ('Cache-Control', 'no-store'), ('Connection', 'close')])
        return Response(HTTPStatus.OK.value, HTTPStatus.OK.phrase, headers, body)

    async def _handler(self, ws):
        try:
            # Recent decisions without spikes (the latest tick, with spikes, follows)
            await ws.send(json.dumps({'type': 'history', 'ticks': list(self._history)[:-1]},
                                     separators=(',', ':')))
            for text in self._latest.values():
                await ws.send(text)
            # No await between building these and joining, and send writes before it yields,
            # so every later change reaches this page after them
            opening = [json.dumps(m, separators=(',', ':')) for m in self._on_open()] if self._on_open else []
            self._clients.add(ws)
            for text in opening:
                await ws.send(text)
            async for raw in ws:
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(message, dict) and self._on_message:
                    self._on_message(message)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    async def serve(self, host, port):
        async with serve(self._handler, host, port, process_request=self._http,
                         max_size=None) as server:
            log.info('Brain dashboard: http://%s:%d', host, port)
            await server.serve_forever()
