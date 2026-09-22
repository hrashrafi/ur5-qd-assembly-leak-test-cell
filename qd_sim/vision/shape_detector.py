"""Shape-based detectors for the arm cameras: recognize a QD's own
hexagonal flange, and the manifold's own port holes, directly from their
silhouette - not a printed ArUco marker. Used for the close-range "final
adjustment" step: the caller already flies to a coarse, assumed hover position
above the tray or manifold; these detectors refine that into exact pixel
centroids for the real features, which qd_sim/vision/ray_cast.py then turns
into world positions.

The hexagon detectors all follow the same shape: threshold for contrast
against the background, find contours, filter by shape/size, return each
match's pixel centroid (via image moments - the standard OpenCV way to get a
contour's centroid, robust to the shape's own asymmetry). The hole detector is
the exception - a shaded hole has no clean silhouette to threshold, so it uses
a Hough circle transform instead (see detect_holes()).

Default size thresholds are pixel-count based, calibrated against a 960x720
render at the specific hover heights the session's detect_qds()/detect_ports()
use (TRAY_HOVER_Z=0.50, MANIFOLD_HOVER_Z=0.30), where they reliably find all
4 QDs/holes (see TRAY_HOVER_Z's comment in qd_cell/constants.py). They do NOT
automatically scale to a different render resolution or hover height (a QD's
real pixel footprint quadruples if the render resolution doubles in each
dimension, for instance - enough to detect nothing at all against a
1920x1440 render). The vision renderer is deliberately kept at 960x720 -
smaller and faster than the 1920x1440 recorded video, and plenty of
resolution for these two shapes - specifically so these defaults stay valid;
re-measure before changing either the renderer resolution or the hover
heights.
"""

import cv2
import numpy as np


def _centroid(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def detect_hexagons(image_bgr, min_area_px=400, max_area_px=6000,
                     min_vertices=5, max_vertices=8, min_circularity=0.6):
    """QDs render as a bright, achromatic (white/gray) hex flange (plus a
    smaller circle - the shaft top - inside it) against the blue-tinted
    floor. Thresholding on brightness alone isn't enough to isolate it - the
    floor's own grid lines are bright too (they show up as a single huge,
    near-frame-spanning contour) - so this also requires low chroma (max
    channel - min channel),
    which the blue-tinted floor/grid lines fail and the white QD passes.

    A contour is accepted as a QD if BOTH its polygon-approximation vertex
    count is roughly hexagonal (5-8, not exactly 6 - render noise/shadows
    routinely push a real hex a vertex or two off exact) AND its circularity
    (4*pi*area/perimeter^2) is at least 0.6 - a real hex measures ~0.77-0.89
    here; the two failure modes that occur (the floor's grid-line
    contour, and a QD partly merged with an arm shadow) both measure well
    under that, so this combination rejects them even though vertex count
    alone doesn't. Returns a list of ((cx, cy), polygon_points) pairs, one
    per detected QD: the pixel centroid, plus the same approxPolyDP result
    already computed for the vertex-count check - the actual recognized
    outline, for the display overlay to draw, not a generic circle standing
    in for it."""
    chroma = np.max(image_bgr.astype(np.int16), axis=2) - np.min(image_bgr.astype(np.int16), axis=2)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = ((chroma < 20) & (gray > 150)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area_px < area < max_area_px):
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri ** 2)
        if circularity < min_circularity:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if not (min_vertices <= len(approx) <= max_vertices):
            continue
        centroid = _centroid(c)
        if centroid is not None:
            results.append((centroid, approx.reshape(-1, 2)))
    return results


def detect_holes(image_bgr, min_radius_px=10, max_radius_px=45, min_dist_px=50, accumulator_thresh=20):
    """Manifold ports render as a shaded circular hole (a bright specular
    highlight over roughly half of it, a dark crescent over the rest, not a
    flat dark disc) cut into the manifold's flat top face. A plain
    intensity threshold only catches the dark crescent half (the resulting
    contours have circularity as low as 0.12, an unusable signal), so this
    uses cv2.HoughCircles instead - built for exactly this "well-defined
    circular edge, shaded interior" case, and reliable at finding all 4
    holes across a range of heights where a crescent-contour approach finds
    none. Returns a list of (cx, cy, radius) in pixels, one per detected
    hole - the radius so an overlay can draw the circle at the size
    actually found, not a guessed fixed radius.

    accumulator_thresh (Hough's own param2 - lower accepts weaker/less
    complete circular edge evidence) is 20: at 25, a borderline but
    otherwise-clean frame (visually indistinguishable from ones that work)
    misses one of the 4 holes, while 22 and below reliably find all 4.

    min_dist_px is 50 because once a port already has a QD installed in it,
    its complex hex-flange-plus-crescent shape (not a plain empty hole
    anymore) can register as TWO overlapping Hough circles instead of one,
    ~32px apart - so at a 30px minimum an occupied port reads as two holes.
    50 finds exactly 4 across 0, 1, 2 and 3 already-occupied ports (the
    realistic range across a 4-QD session) while staying comfortably below
    the real spacing between actual distinct ports (~100-120px in the
    calibrated hover setup), so it never merges real, separate ports
    together either."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist_px,
        param1=80, param2=accumulator_thresh, minRadius=min_radius_px, maxRadius=max_radius_px,
    )
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


def detect_flange_hexagon(image_bgr, cx, cy, patch_radius_px=80, min_area_px=200,
                           chroma_max=25, gray_min=140):
    """Center AND corner orientation of an installed QD's hex flange, from
    a NEARLY TOP-DOWN view of it, in one pass.

    It is a contour fit rather than a Hough CIRCLE fit plus a handful of Hough
    LINE segments: with the camera looking straight down the probe's own axis,
    the flange renders as a clean, centered, genuinely hexagonal silhouette,
    so its own contour can be used directly, which is both simpler and far
    more accurate. Measured on the same four installed QDs, against ground
    truth read straight out of the sim (the circle+lines figures are from a
    camera ~145mm off to one side, where the flange is a ~20deg-oblique blob
    and that is about all the view supports):

        circle+lines:  center err 4.7mm,     yaw err 0.7-28deg
        this:          center err 0.4mm,     yaw err 0.08-0.32deg

    (detect_holes() in fact finds NOTHING at all in a top-down center crop -
    a hexagon seen face-on is a poor circle - so a circle-based chain isn't
    merely less accurate here, it doesn't work.)

    Mask: the same bright + low-chroma test detect_hexagons() uses, which
    separates the pale gray flange from the manifold's own tan top face.
    Applied in a local crop around (cx, cy) so that exactly one port is in view
    (the whole-frame version is detect_flange_hexagons_all()). Of the resulting
    contours, the one whose own centroid is NEAREST the crop's center wins -
    with the camera centered on the port being inspected, that's this port's
    flange and not a neighbor's 45mm away.

    Center comes from that contour's moment centroid (the natural center of
    the actual detected shape, not a circle fitted to a hexagon).

    Orientation comes from the 6th harmonic of the contour's own radius
    function about that centroid: r(theta) for a hexagon peaks at its 6
    corners, so the phase of sum(w * r * exp(6j*theta)) / 6 is the corner
    direction directly, folded into the 60deg of real distinction a
    6-fold-symmetric shape has, driven by EVERY contour pixel instead of
    however many line segments a Hough transform happens to return - which
    is why it's ~10-100x more accurate than a line-segment approach).
    Points are arc-length weighted so a densely-sampled straight run
    doesn't outvote a sparsely-sampled one.

    Returns ((px, py), corner_angle_deg) in the ORIGINAL image's own pixel
    coordinates and pixel-frame angle, or None if no suitable contour was
    found. The pixel-frame angle is converted to a world yaw by the caller
    (see seam_path.world_yaw_from_pixel_angle())."""
    h, w = image_bgr.shape[:2]
    x0, y0 = max(0, int(cx - patch_radius_px)), max(0, int(cy - patch_radius_px))
    x1, y1 = min(w, int(cx + patch_radius_px)), min(h, int(cy + patch_radius_px))
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    found = _flange_contours(crop, min_area_px, chroma_max, gray_min)
    if not found:
        return None
    crop_cx, crop_cy = cx - x0, cy - y0
    contour, centroid = min(found, key=lambda fc: np.hypot(fc[1][0] - crop_cx, fc[1][1] - crop_cy))
    angle = _hex_corner_angle(contour, centroid)
    if angle is None:
        return None
    return (centroid[0] + x0, centroid[1] + y0), angle


def _flange_contours(image_bgr, min_area_px, chroma_max, gray_min):
    """Every contour in image_bgr big enough and pale/achromatic enough to
    be a flange, as (contour, centroid) pairs. The mask is the same
    bright + low-chroma test detect_hexagons() uses."""
    chroma = np.max(image_bgr.astype(np.int16), axis=2) - np.min(image_bgr.astype(np.int16), axis=2)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = ((chroma < chroma_max) & (gray > gray_min)).astype(np.uint8) * 255
    # CHAIN_APPROX_NONE, not SIMPLE: _hex_corner_angle() wants every
    # boundary pixel, and SIMPLE would collapse each straight hex side to
    # its two endpoints.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        centroid = _centroid(c)
        if centroid is not None:
            out.append((c, centroid))
    return out


def _hex_corner_angle(contour, centroid):
    """Corner direction (degrees, pixel frame) from the 6th harmonic of the
    contour's radius function about centroid - see detect_flange_hexagon()'s
    docstring for why this is the orientation estimate."""
    pts = contour.reshape(-1, 2).astype(float)
    dx, dy = pts[:, 0] - centroid[0], pts[:, 1] - centroid[1]
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    step = np.hypot(np.diff(pts[:, 0], append=pts[0, 0]), np.diff(pts[:, 1], append=pts[0, 1]))
    weight = np.roll(step, 1) * r
    total = np.sum(weight)
    if total <= 0:
        return None
    harmonic = np.sum(weight * np.exp(1j * 6 * theta)) / total
    return np.degrees(np.angle(harmonic)) / 6.0


def detect_flange_hexagons_all(image_bgr, min_area_px=200, max_area_px=20000,
                                chroma_max=25, gray_min=140, min_circularity=0.6):
    """Every installed QD flange visible in the WHOLE frame, as
    ((px, py), corner_angle_deg) pairs - the survey counterpart to
    detect_flange_hexagon() above, which refines one already-located
    flange.

    This serves the roster-style scheduling both arms use: fly to ONE fixed
    survey pose that frames the entire port row, detect every flange at
    once, then choose the next one to work on by excluding the ones already
    done. That needs "find all of them", not "refine this one".

    Frame-wide works here for the same reason the local version's crop does
    NOT need to be tight: the mask keys on pale + achromatic, and the
    manifold's own top face is a warm tan (mat_manifold in scene/world.xml,
    chroma well above the threshold), so the slab itself is rejected and
    only the flanges survive. That's specific to a top-down view of the
    MANIFOLD; over the tray, use detect_hexagons() instead.

    max_area_px rejects a blob far too large to be a single flange - e.g.
    two neighbours merged, or an arm link crossing the frame - rather than
    silently returning its centroid as if it were a real part.

    min_circularity is the same 4*pi*area/perimeter^2 test detect_hexagons()
    already uses, and for the same reason. Frame-wide, the pale+achromatic
    mask also catches thin bright slivers that aren't parts at all (the
    floor's own grid lines, a lit edge of either arm). Measured on a real
    4-port survey frame, the separation is not close: the four genuine
    flanges come out at 0.81-0.83 (matching detect_hexagons()'s own
    "a real hex measures ~0.77-0.89"), every sliver at 0.01-0.02. This
    rejects them on SHAPE rather than on where they happen to lie, so it
    doesn't quietly bake the manifold's own position into the detector."""
    out = []
    for contour, centroid in _flange_contours(image_bgr, min_area_px, chroma_max, gray_min):
        area = cv2.contourArea(contour)
        if area > max_area_px:
            continue
        peri = cv2.arcLength(contour, True)
        if peri <= 0 or 4 * np.pi * area / (peri ** 2) < min_circularity:
            continue
        angle = _hex_corner_angle(contour, centroid)
        if angle is not None:
            out.append((centroid, angle))
    return out
