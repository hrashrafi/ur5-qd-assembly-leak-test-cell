"""
Kinematic weld: rigidly attaches a mocap body to a site (normally the
gripper's pinch site) once engaged, so it follows the site exactly every
tick - a real weld, not a simulated hold. This is used instead of relying on
contact friction/mass/inertia to keep a grasped part in the gripper (see
world.xml's QD body comment for why: a friction grip on this part is unstable
at several grasp heights, however much force or ramp-timing is applied).

The weld itself is nothing more than "remember the target's pose in the
site's local frame, then rebuild the target's world pose from that local
pose every time the site moves" - the same parent/child pose composition
used throughout qd_sim.transforms, just applied every tick instead of
once.
"""

from qd_sim.robot.kinematics import site_pose
from qd_sim import transforms as tf


class KinematicWeld:
    def __init__(self, model):
        self.model = model
        self.mocap_id = None
        self.site_id = None
        self._site_T_target = None  # constant 4x4: target's pose in the site's local frame

    @property
    def engaged(self):
        return self.mocap_id is not None

    def engage(self, data, mocap_body_name, site_id):
        """Weld the named mocap body to the given site, at their current
        relative pose - call this the instant the gripper finishes closing
        on it."""
        body_id = self.model.body(mocap_body_name).id
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id < 0:
            raise ValueError(f"'{mocap_body_name}' is not a mocap body - check world.xml")
        self.mocap_id = mocap_id
        self.site_id = site_id

        site_pos, site_quat = site_pose(data, site_id)
        target_pos = data.mocap_pos[mocap_id].copy()
        target_quat = data.mocap_quat[mocap_id].copy()

        world_T_site = tf.pose_to_matrix(site_pos, site_quat)
        world_T_target = tf.pose_to_matrix(target_pos, target_quat)
        self._site_T_target = tf.compose(tf.invert(world_T_site), world_T_target)

    def update(self, data):
        """Recompute the welded body's world pose from the site's current
        pose - call every physics tick once engaged. No-op if not engaged."""
        if not self.engaged:
            return
        site_pos, site_quat = site_pose(data, self.site_id)
        world_T_site = tf.pose_to_matrix(site_pos, site_quat)
        world_T_target = tf.compose(world_T_site, self._site_T_target)
        pos, quat = tf.matrix_to_pose(world_T_target)
        data.mocap_pos[self.mocap_id] = pos
        data.mocap_quat[self.mocap_id] = quat

    def release(self):
        """Stop tracking - the mocap body holds whatever pose it last had
        (mocap bodies never move on their own)."""
        self.mocap_id = None
        self.site_id = None
        self._site_T_target = None
