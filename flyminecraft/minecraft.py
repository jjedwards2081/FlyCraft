"""Minecraft Education / Bedrock websocket protocol (the game connects to us via /connect)."""

import asyncio
import json
import logging
import uuid

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)

MAX_PENDING = 100  # Minecraft rejects commands beyond 100 awaiting a response


class MinecraftClient:
    def __init__(self, ws):
        self.ws = ws
        self._pending = {}
        self._slots = asyncio.Semaphore(MAX_PENDING)
        self._handlers = {}

    async def listen(self):
        try:
            async for raw in self.ws:
                self._dispatch(json.loads(raw))
        except ConnectionClosed:
            pass
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError('Minecraft disconnected'))

    def _dispatch(self, message):
        header, body = message.get('header', {}), message.get('body', {})
        if header.get('messagePurpose') == 'event':
            name = header.get('eventName') or body.get('eventName')
            log.debug('event %s: %s', name, body)
            for handler in self._handlers.get(name, []):
                handler(body)
            return
        future = self._pending.get(header.get('requestId'))
        if future and not future.done():
            future.set_result(body)

    async def _send(self, purpose, body, request_id=None):
        await self.ws.send(json.dumps({
            'header': {
                'version': 1,
                'requestId': request_id or str(uuid.uuid4()),
                'messageType': 'commandRequest',
                'messagePurpose': purpose,
            },
            'body': body,
        }))

    async def command(self, line, timeout=10.0):
        """Run a command and return the response body (statusCode < 0 means it failed)."""
        async with self._slots:
            request_id = str(uuid.uuid4())
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            try:
                await self._send('commandRequest', {
                    'version': 1,
                    'commandLine': line,
                    'origin': {'type': 'player'},
                }, request_id)
                body = await asyncio.wait_for(future, timeout)
            finally:
                self._pending.pop(request_id, None)
        log.debug('%s -> %s', line, body)
        return body

    async def subscribe(self, event_name, handler):
        self._handlers.setdefault(event_name, []).append(handler)
        await self._send('subscribe', {'eventName': event_name})


def succeeded(body):
    return body.get('statusCode', 0) >= 0


async def run_server(host, port, on_connect):
    """Serve forever; on_connect(client) runs for each game connection."""
    busy = asyncio.Lock()

    async def handler(ws):
        if busy.locked():
            log.warning('Rejecting a second Minecraft connection; the fly has only one body')
            await ws.close()
            return
        async with busy:
            client = MinecraftClient(ws)
            listener = asyncio.create_task(client.listen())
            try:
                await on_connect(client)
            except ConnectionError as e:
                log.info('%s', e)
            finally:
                listener.cancel()

    # Minecraft does not reliably answer websocket pings
    async with serve(handler, host, port, ping_interval=None) as server:
        log.info('Waiting for Minecraft: type  /connect localhost:%d  in the game chat', port)
        await server.serve_forever()
