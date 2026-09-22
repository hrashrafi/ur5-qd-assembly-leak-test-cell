"""The operator panel: a browser front end over the PLC's Modbus server.

A browser cannot open a raw TCP Modbus socket, so a small Python process sits
in between - one persistent Modbus client polling the PLC, and a JSON endpoint
the page fetches. N open tabs therefore cost one poll, not N.
"""
