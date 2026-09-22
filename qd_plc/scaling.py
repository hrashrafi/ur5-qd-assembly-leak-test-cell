"""Engineering units <-> 16-bit Modbus registers.

Modbus carries 16-bit words and nothing else: no floats, no signs unless you
choose them, no units. Every physical quantity on the wire is therefore an
integer with an agreed scale factor, and both ends have to agree exactly.
This module is the only place the Python side writes that conversion down;
the scale of each tag lives in qd_plc.tags, and the ST program compares raw
integers against limits already expressed in the same units.

Conventions used throughout (see qd_plc.tags for which tag uses which):

  * lengths in 0.01 mm   -> 8.42 mm is 842
  * angles  in 0.01 deg  -> 0.51 deg is 51, signed

Clamping is deliberate and silent at the boundary: a register physically
cannot hold 70000, so a runaway value saturates rather than wrapping round to
a small number that would look plausible on the HMI.
"""

from __future__ import annotations

U16_MAX = 0xFFFF
I16_MIN, I16_MAX = -0x8000, 0x7FFF


def clamp_u16(value: float) -> int:
    """Round to the nearest integer and saturate into unsigned 16-bit."""
    return max(0, min(U16_MAX, int(round(value))))


def clamp_i16(value: float) -> int:
    """Round to the nearest integer and saturate into signed 16-bit."""
    return max(I16_MIN, min(I16_MAX, int(round(value))))


def to_wire_i16(value: int) -> int:
    """Two's-complement a signed value into the unsigned word Modbus sends."""
    return value & U16_MAX


def from_wire_i16(word: int) -> int:
    """Undo to_wire_i16 - interpret a raw register as signed."""
    word &= U16_MAX
    return word - 0x10000 if word & 0x8000 else word


# --- bit packing for coil / discrete-input blocks -------------------------
# pymodbus stores bit blocks packed 16-to-a-register, so bit n lives at
# register n // 16, bit n % 16. Verified against pymodbus 3.15.

def bit_location(bit_index: int) -> tuple[int, int]:
    """Bit number -> (register index, bit offset within that register)."""
    return bit_index // 16, bit_index % 16


def read_bit(registers: list[int], bit_index: int) -> bool:
    reg, off = bit_location(bit_index)
    return bool(registers[reg] & (1 << off))


def write_bit(registers: list[int], bit_index: int, value: bool) -> None:
    reg, off = bit_location(bit_index)
    if value:
        registers[reg] |= 1 << off
    else:
        registers[reg] &= ~(1 << off) & U16_MAX
