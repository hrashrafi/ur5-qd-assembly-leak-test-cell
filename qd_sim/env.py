"""Loading the scene and stepping physics - the one place that knows how.

Used by the session, so nothing else has to know where the scene lives or
how a reset works.
"""

from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = ROOT / "scene" / "scene_real.xml"


class SimEnv:
    def __init__(self, scene_path=DEFAULT_SCENE, keyframe="ready"):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.keyframe_name = keyframe
        self.reset()

    def reset(self):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
