"""
Build the path the leak-test probe traces around an installed QD: an ordered
ring of world waypoints that hugs the REAL hex flange - closer at the flats,
farther at the corners - rather than a plain circle.

Why not just rotate a fixed-radius tool around the port's center: that only
ever traces a circle. Picking the circle's radius to clear the corners leaves
a visible gap at the flats; picking it to hug the flats lets the corners poke
through. The QD's flange is a real hexagon, so the seam being inspected
genuinely isn't a circle. Since the probe's own tip sits at a truly FIXED
offset from wherever the arm's 6 joints put it (rigidly, no extra joint),
driving the ARM's own TCP (the "leak/probe_tip" site, via ordinary 6-DOF IK -
the same ArmController motion commands every other move in this project
uses) through a hex-shaped ring reproduces that same shape at the tip
directly; no extra joint or controller needed, just a different set of
waypoints.

The flange's centre and orientation come from the leak camera
(shape_detector.detect_flange_hexagon()); its corner radius is a known
constant of the part. world_yaw_from_pixel_angle() turns the detected pixel
orientation into a world yaw using ray_cast.pixel_to_world_on_plane(), the
same "known Z-plane, refine X/Y" approach used throughout the vision
pipeline.
"""

import numpy as np

from qd_sim.vision.ray_cast import pixel_to_world_on_plane

# The flange's own real corner-to-center radius - matches qd_*_col_flange's
# size="0.0156 ..." in scene/world.xml (31.2mm corner-to-corner, i.e. 15.6mm
# corner radius, consistent with the connector's datasheet - see
# qd_cell/constants.py's thread spec). A physical constant of the part, not something that needs
# re-detecting per QD the way its position/orientation do - see
# hex_ring_from_known_geometry() below.
FLANGE_CORNER_RADIUS_M = 0.0156


def world_yaw_from_pixel_angle(center_px, angle_deg, K, cam_world_pos, cam_world_mat_mj,
                                seam_z, probe_px=20.0):
    """Convert a PIXEL-frame direction (e.g.
    shape_detector.detect_flange_hexagon()'s corner angle) at center_px
    into a world-frame yaw in radians.

    Not a constant rotation: the camera's own roll about its optical axis
    sets the pixel-to-world angle offset, and that's a property of the
    arm's pose, not something to hardcode. So this ray-casts two points
    along the direction onto the known seam_z plane and takes the resulting
    world-frame angle, rather than trying to derive it from the camera
    matrix by hand.

    probe_px is just how far along the pixel direction the second point is
    taken; anything comfortably above sub-pixel noise works, and the
    result doesn't depend on it (a straight line stays straight through
    this ray-cast, both endpoints lying in the same single Z-plane)."""
    a = np.radians(angle_deg)
    p0 = np.asarray(center_px, dtype=float)
    p1 = p0 + probe_px * np.array([np.cos(a), np.sin(a)])
    w0 = pixel_to_world_on_plane(p0[0], p0[1], K, cam_world_pos, cam_world_mat_mj, seam_z)[:2]
    w1 = pixel_to_world_on_plane(p1[0], p1[1], K, cam_world_pos, cam_world_mat_mj, seam_z)[:2]
    d = w1 - w0
    return float(np.arctan2(d[1], d[0]))


def hex_ring_from_known_geometry(center_world_xy, yaw_rad, outward_offset_m,
                                  corner_radius_m=FLANGE_CORNER_RADIUS_M):
    """Build a hex-hugging ring from a KNOWN/DETECTED center and
    orientation plus the flange's own real corner radius (a physical CAD
    constant, not something that needs re-detecting per QD). The centre
    and orientation come from detect_flange_hexagon() on the leak camera;
    combining them with this known radius gives the ring directly, rather
    than tracing whatever outline a detector happened to return.

    Vertices sit at yaw_rad + k*60deg for k in 0..5 - a property of this
    part's geometry (see qd_sim/tasks/pick_sequence.py's module
    docstring: "6 corners at exactly 0/60/120/180/-60/-120deg in the QD's own
    frame"), at corner_radius_m from center_world_xy, and each corner is pushed
    outward from the centre by outward_offset_m for clearance. Returns the 6
    corners in order; the probe traces the straight edges between them."""
    center = np.asarray(center_world_xy, dtype=float)
    angles = yaw_rad + np.radians(np.arange(6) * 60)
    vertices = center + corner_radius_m * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    ring = []
    for vertex in vertices:
        outward = vertex - center
        ring.append(vertex + outward / np.linalg.norm(outward) * outward_offset_m)
    return ring
