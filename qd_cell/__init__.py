"""The session: what the robot actually does, and how the PLC gates it.

Three packages make up the cell:

    qd_sim    the robot - motion, vision, task sequences
    qd_plc    the supervisor - Modbus maps, the gate handshake, the panel
    qd_cell   this one - the session that drives the robot through the gates

See docs/ARCHITECTURE.md.
"""
