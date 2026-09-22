"""The live register storage behind a Modbus slave, addressed by tag name.

Everything above this module talks in names and engineering units
("INSERT_DEPTH", 8.42 mm); everything below it is pymodbus' packed 16-bit
storage. This is the only place the two meet.

Why it pokes pymodbus' internals
--------------------------------
pymodbus 3.15 exposes its datastore through `async_getValues`/`async_setValues`
coroutines. Our sim loop is a plain blocking `for` loop that cannot await
anything, and marshalling every publish onto the server's event loop with
`run_coroutine_threadsafe` would add a round-trip per signal on a loop that
publishes several times a second while also stepping physics and rendering.

Underneath those coroutines, `SimRuntime.block[k]` holds a plain `list[int]`
that the server reads from directly. Assigning an element of a Python list is
a single bytecode operation and therefore atomic under the GIL, so the sim
thread can publish into it without a lock and the server thread will serve the
new value on its next request. Verified end-to-end against a live server and a
real client.

Every value in the maps is a single register, so no read can ever catch a
value half-written.
"""

from __future__ import annotations

import threading

from pymodbus.simulator import DataType, SimData, SimDevice

from . import scaling
from .tags import DType, Kind, Tag, TagMap

#: SimRuntime's internal block keys, one per Modbus object type.
_BLOCK_KEY = {
    Kind.COIL: "c",
    Kind.DISCRETE_INPUT: "d",
    Kind.HOLDING_REG: "h",
    Kind.INPUT_REG: "i",
}


def build_device(tagmap: TagMap, device_id: int = 1) -> SimDevice:
    """A SimDevice sized from the map's DECLARED block sizes.

    Sized from the declared sizes rather than the addresses actually in use,
    so the OpenPLC slave-device configuration stays valid when a tag is later
    added into the spare space.
    """
    def bits(kind):
        size = tagmap.size(kind)
        return [SimData(0, count=max(size, 1), values=False,
                        datatype=DataType.BITS)]

    def regs(kind):
        size = tagmap.size(kind)
        return [SimData(0, count=max(size, 1), values=0,
                        datatype=DataType.REGISTERS)]

    # SimDevice wants (coils, discrete inputs, holding, input registers).
    return SimDevice(
        id=device_id,
        simdata=(bits(Kind.COIL), bits(Kind.DISCRETE_INPUT),
                 regs(Kind.HOLDING_REG), regs(Kind.INPUT_REG)),
    )


class Datastore:
    """Tag-addressed view of one slave's live registers.

    It starts with storage of its own, and `bind()` then points it at the
    running server's, so every write from then on is what the server serves.
    """

    def __init__(self, tagmap: TagMap, device_id: int = 1):
        self.tags = tagmap
        self.device_id = device_id
        self._blocks: dict[str, list[int]] = {}
        #: Guards only the read-modify-write of bit registers, where two
        #: different bits share one word and `|=` is not atomic. Whole-register
        #: writes do not take it.
        self._bit_lock = threading.Lock()
        self._detach()

    # -- storage binding ---------------------------------------------------

    def _detach(self) -> None:
        """Private storage, no server - what the store holds until bind()."""
        for kind, key in _BLOCK_KEY.items():
            size = self.tags.size(kind)
            words = (size + 15) // 16 if kind in Kind.BITS else size
            self._blocks[key] = [0] * max(words, 1)

    def bind(self, runtime) -> None:
        """Point at a live server's SimRuntime so writes are served."""
        for key in _BLOCK_KEY.values():
            self._blocks[key] = runtime.block[key][2]

    def _block(self, tag: Tag) -> list[int]:
        return self._blocks[_BLOCK_KEY[tag.kind]]

    # -- raw access --------------------------------------------------------

    def get_raw(self, name: str) -> int | bool:
        tag = self.tags[name]
        if tag.is_bit:
            return scaling.read_bit(self._block(tag), tag.addr)
        return self._block(tag)[tag.addr]

    def set_raw(self, name: str, value: int | bool) -> None:
        tag = self.tags[name]
        if tag.is_bit:
            with self._bit_lock:
                scaling.write_bit(self._block(tag), tag.addr, bool(value))
        else:
            self._block(tag)[tag.addr] = int(value) & 0xFFFF

    # -- engineering units -------------------------------------------------

    def get(self, name: str):
        """Raw register -> engineering value, honouring scale and sign."""
        tag = self.tags[name]
        raw = self.get_raw(name)
        if tag.is_bit:
            return bool(raw)
        if tag.dtype == DType.INT16:
            raw = scaling.from_wire_i16(raw)
        return raw * tag.scale if tag.scale != 1.0 else raw

    def set(self, name: str, value) -> None:
        """Engineering value -> raw register, clamped into range.

        Clamping here rather than at the call sites means a runaway physical
        value saturates instead of wrapping into a small, plausible-looking
        number on the operator panel.
        """
        tag = self.tags[name]
        if tag.is_bit:
            self.set_raw(name, bool(value))
            return
        raw = value / tag.scale if tag.scale != 1.0 else value
        if tag.dtype == DType.INT16:
            self.set_raw(name, scaling.to_wire_i16(scaling.clamp_i16(raw)))
        else:
            self.set_raw(name, scaling.clamp_u16(raw))

    # -- sticky bits -------------------------------------------------------
    # Set-only. The sim never clears these; only clear_sticky() does, and the
    # PLC is the only thing that calls it. That is what makes a sim running
    # far faster than the PLC polls safe: a condition raised and gone within
    # one poll interval still reaches the supervisor.

    def latch(self, name: str) -> None:
        """Raise a sticky bit. Never lowers it."""
        if not self.tags[name].sticky:
            raise ValueError(f"{name} is not a sticky bit; use set()")
        self.set_raw(name, True)

    def clear_sticky(self) -> None:
        """Lower every sticky bit."""
        for tag in self.tags.sticky_bits():
            self.set_raw(tag.name, False)
