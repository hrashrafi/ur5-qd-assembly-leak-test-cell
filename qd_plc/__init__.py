"""Supervisory control layer for the QD manifold assembly cell.

Three processes talk over Modbus TCP, mirroring a real industrial cell:

    panel server --Modbus--> OpenPLC (Structured Text) --Modbus--> MuJoCo sim
    operator panel               cell PLC                      robot controller

The browser reaches the panel server over plain HTTP, since a browser cannot
open a Modbus connection itself (see qd_plc.hmi).

The sim keeps motion and vision internal, exactly as a real robot controller
keeps its control loop off the fieldbus; Modbus carries only handshake and
status. See docs/ARCHITECTURE.md for the architecture and the mapping to
real hardware, and qd_plc.tags for the register maps themselves.
"""
