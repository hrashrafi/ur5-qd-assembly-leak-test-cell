"""A Modbus TCP slave running in a background thread of a synchronous program.

The simulation is one blocking single-threaded loop with no event loop of its
own, and it must not acquire one: its timing is already tuned and its physics
step cannot yield. So the server gets its own thread with its own asyncio
loop, and the sim publishes into the live register storage directly (see
qd_plc.datastore for why that is safe).

`mj_step` releases the GIL, so this thread genuinely gets CPU while physics is
running - this is not a case where Python threading buys nothing.

Shutdown matters more than it looks. If the thread is left running, the port
stays bound and the next run fails to start with an unhelpful address-in-use;
the session's try/finally must call stop().
"""

from __future__ import annotations

import asyncio
import threading

from pymodbus.server import ModbusTcpServer

from .datastore import Datastore, build_device
from .tags import TagMap


class SlaveServer:
    """A tag-addressed Modbus slave on its own thread.

    Typical use, from synchronous code:

        srv = SlaveServer(MAP_A, port=5020)
        srv.start()
        srv.data.set("PHASE_CODE", Phase.SCREW_IN)
        ...
        srv.stop()
    """

    def __init__(self, tagmap: TagMap, host: str = "0.0.0.0", port: int = 5020,
                 device_id: int = 1):
        # Default to 0.0.0.0, not loopback: OpenPLC reaches this from inside a
        # Docker container, and binding 127.0.0.1 is the single most common
        # reason it reports the slave as offline.
        self.tags = tagmap
        self.host = host
        self.port = port
        self.device_id = device_id
        self.data = Datastore(tagmap, device_id)
        self._device = build_device(tagmap, device_id)
        self._server: ModbusTcpServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = 10.0) -> None:
        """Start serving and block until the socket is actually listening.

        Waiting for a real listen rather than sleeping a fixed interval means
        a caller that immediately publishes cannot race the server's startup.
        """
        if self._thread is not None:
            raise RuntimeError("server already started")
        self._thread = threading.Thread(target=self._run, name="modbus-slave",
                                        daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError(
                f"Modbus slave did not start listening on "
                f"{self.host}:{self.port} within {timeout}s"
            )
        if self._error is not None:
            raise RuntimeError(
                f"Modbus slave failed to start on {self.host}:{self.port}: "
                f"{self._error}"
            ) from self._error

    def _run(self) -> None:
        async def serve():
            self._loop = asyncio.get_running_loop()
            self._stopping = asyncio.Event()
            try:
                self._server = ModbusTcpServer(
                    self._device, address=(self.host, self.port))
                # background=True listens and returns, instead of blocking
                # here forever. Calling listen() ourselves first would make
                # serve_forever() refuse ("already running server object")
                # and leave a bound socket that accepts connections and
                # answers nothing.
                await self._server.serve_forever(background=True)
                # Bind to the storage the server actually serves from: the
                # SimDevice builds its own copy, so publishing anywhere else
                # would silently go nowhere.
                self.data.bind(self._server.context.devices[self.device_id])
            except BaseException as exc:
                self._error = exc
                self._ready.set()
                return
            self._ready.set()
            await self._stopping.wait()
            await self._server.shutdown()

        try:
            asyncio.run(serve())
        except BaseException as exc:
            # Record rather than swallow. An exception discarded here leaves
            # the caller with a server that looks started and never answers.
            self._error = exc
            self._ready.set()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop serving and release the port. Safe to call twice.

        The session's try/finally must call this: a thread left running
        keeps the port bound, and the next run dies with an address-in-use
        error that points nowhere near the real cause.
        """
        if self._thread is None:
            return
        loop, stopping = self._loop, self._stopping
        if loop is not None and stopping is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stopping.set)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"Modbus slave thread did not stop within {timeout}s; "
                f"port {self.port} may still be bound"
            )
        self._thread = None
        self._server = None
        self._loop = None
        self._stopping = None
        self._ready.clear()

