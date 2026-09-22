"""Which addresses the operator panel is allowed to touch.

In OpenPLC, %IX100+/%QX100+/%IW100+/%QW100+ belong to the master mapping - the
link down to the robot controller. The panel writing there would collide with
the supervisor's own traffic and produce a permit race that looks exactly like
a PLC bug. Nothing in the platform prevents it, so it is enforced here.

Only the PLC writes the robot's registers; only the panel writes the panel's
command range. That split is a rule, not a mechanism, which is precisely why
it is worth a guard.
"""

from __future__ import annotations

from ..tags import MAP_B, PLC_RESERVED_COIL, PLC_RESERVED_REG, Tag

#: The only addresses the panel may write: its pushbutton coils. Everything
#: else it can read but not modify - status is the PLC's to publish.
WRITABLE_COILS = range(80, 96)


class NotWritable(PermissionError):
    pass


def check_readable(tag: Tag) -> None:
    limit = PLC_RESERVED_COIL if tag.is_bit else PLC_RESERVED_REG
    if tag.addr >= limit:
        raise NotWritable(
            f"{tag.name} at {tag.kind} {tag.addr} is inside the OpenPLC master "
            f"mapping, which belongs to the robot link. The panel must stay "
            f"below {limit}."
        )


def check_writable(name: str) -> Tag:
    """Raise unless the panel owns this address."""
    tag = MAP_B[name]
    check_readable(tag)
    if not tag.is_bit or tag.addr not in WRITABLE_COILS:
        raise NotWritable(
            f"{name} is published by the PLC, not written by the panel. "
            f"The panel may only write coils 80-95."
        )
    return tag
