from pathlib import Path
from shutil import copy2, copytree
import xml.etree.ElementTree as ET
import mujoco
from wuji_description import ROOT as WUJI_ROOT
from dex_supermimic.assets.wuji_hand2 import _get_spec, wuji_hand2_xml_path

project = Path.cwd()
out = project / 'src/unisupermimic/assets/robots/wuji_hand2'
(out / 'model_assets/meshes').mkdir(parents=True, exist_ok=True)
(out / 'model_assets/textures').mkdir(parents=True, exist_ok=True)

# Copy hand mesh assets supplied by wuji-description.
hand_mesh_dir = WUJI_ROOT / 'hand2/hand2_2/body/meshes/right'
for src in hand_mesh_dir.iterdir():
    if src.is_file(): copy2(src, out / 'model_assets/meshes' / src.name)
# Copy cube mesh and textures.
cube_root = project / 'src/unisupermimic/assets/robots/wuji_hand2/inhand_cube'
copy2(cube_root / 'meshes/dex_cube.obj', out / 'model_assets/meshes/dex_cube.obj')
copytree(cube_root / 'textures', out / 'model_assets/textures', dirs_exist_ok=True)

hand_spec = _get_spec(wuji_hand2_xml_path('right'))
hand_root = ET.fromstring(hand_spec.to_xml())
cube_root_xml = ET.parse(cube_root / 'xmls/cube.xml').getroot()
hand_root.set('model', 'wuji_hand2_reorient')
compiler = hand_root.find('compiler')
assert compiler is not None
compiler.attrib.clear()
compiler.set('angle', 'radian')
compiler.set('meshdir', 'model_assets/meshes')
compiler.set('texturedir', 'model_assets')
# Use calibrated hand root pose from Dex reorient constants.
hand_body = hand_root.find('./worldbody/body[@name="right_palm_link"]')
assert hand_body is not None
hand_body.set('pos', '-0.0284999 0.00300004 0.500250158')
hand_body.set('quat', '0.5 0.5 0.5 0.5')
# Add cube defaults, assets, and body into the same MJCF root.
defaults = hand_root.find('default')
cube_defaults = cube_root_xml.find('default')
if defaults is not None and cube_defaults is not None:
    for child in list(cube_defaults): defaults.append(child)
assets = hand_root.find('asset')
cube_assets = cube_root_xml.find('asset')
assert assets is not None and cube_assets is not None
for child in list(cube_assets):
    if child.tag == 'mesh' and child.get('name') == 'cube_mesh':
        child.set('file', 'dex_cube.obj')
    assets.append(child)
worldbody = hand_root.find('worldbody')
assert worldbody is not None
floor = ET.Element('geom', {'name':'floor', 'type':'plane', 'size':'5 5 0.1', 'pos':'0 0 0', 'contype':'1', 'conaffinity':'1'})
worldbody.insert(0, floor)
cube_body = cube_root_xml.find('./worldbody/body[@name="cube"]')
assert cube_body is not None
cube_body.set('pos', '-0.108 0.0072 0.5616')
cube_body.set('quat', '1 0 0 0')
worldbody.append(cube_body)

# Match Dex-SuperMimic's reorient grasp pose.  Keeping this as a MuJoCo
# keyframe makes it the UniSim default qpos, so action defaults and reset
# state use the same bent-hand configuration.
hand2_default_joint_pos = [
    0.8, -0.3, 0.5, 0.4,
    0.45, -0.15, 0.8, 0.5,
    0.45, 0.0, 0.9, 0.3,
    0.6, 0.15, 1.1, 0.2,
    1.2, 0.2, 0.8, 0.4,
]
keyframe = hand_root.find('keyframe')
if keyframe is None:
    keyframe = ET.SubElement(hand_root, 'keyframe')
ET.SubElement(keyframe, 'key', {
    'name': 'hand2_default',
    'qpos': ' '.join(str(value) for value in (*hand2_default_joint_pos, -0.108, 0.0072, 0.5616, 1.0, 0.0, 0.0, 0.0)),
    'qvel': ' '.join('0.0' for _ in range(26)),
})

# Add the calibrated wrist-tag frame used by Dex-Supermimic observations.
def _mul_pose(pos1, quat1, pos2, quat2):
    out_pos = __import__("numpy").zeros(3, dtype=float)
    out_quat = __import__("numpy").zeros(4, dtype=float)
    mujoco.mju_mulPose(out_pos, out_quat, pos1, quat1, pos2, quat2)
    return out_pos, out_quat

mount = next(body for body in worldbody.iter("body") if body.get("name") == "right_mount")
mount_pos = [float(x) for x in mount.get("pos", "0 0 0").split()]
mount_quat = [float(x) for x in mount.get("quat", "1 0 0 0").split()]
inv_mount_pos = __import__("numpy").zeros(3, dtype=float)
inv_mount_quat = __import__("numpy").zeros(4, dtype=float)
mujoco.mju_negPose(inv_mount_pos, inv_mount_quat, mount_pos, mount_quat)
calibrated_pos = (-0.0030000041, -0.000250110803, 0.0285)
calibrated_quat = (1.0, 0.0, 0.0, -0.0000081995)
tag_mount_pos = (0.0, 0.037408, 0.034737)
tag_mount_quat = (0.5, -0.5, -0.5, -0.5)
calibrated_tag_pos, calibrated_tag_quat = _mul_pose(calibrated_pos, calibrated_quat, tag_mount_pos, tag_mount_quat)
tag_pos, tag_quat = _mul_pose(inv_mount_pos, inv_mount_quat, calibrated_tag_pos, calibrated_tag_quat)
tag_body = ET.SubElement(mount, "body", {
    "name": "right_wrist_tag_frame",
    "pos": " ".join(f"{float(x):.10g}" for x in tag_pos),
    "quat": " ".join(f"{float(x):.10g}" for x in tag_quat),
})
ET.SubElement(tag_body, "site", {
    "name": "right_wrist_tag", "type": "box",
    "size": "0.0252 0.0252 0.0006", "group": "1",
    "rgba": "1 0.9 0 0.5",
})

# Touch sensors provide a backend-neutral first contact signal for the five fingertips.
sensors = hand_root.find("sensor")
if sensors is None:
    sensors = ET.SubElement(hand_root, "sensor")
for finger in range(1, 6):
    ET.SubElement(sensors, "touch", {
        "name": f"right_finger{finger}_tip_touch",
        "site": f"right_finger{finger}_tip",
    })
# Use deterministic simulator settings matching the Dex reorient task.
option = hand_root.find('option')
assert option is not None
option.set('timestep', '0.01')
option.set('integrator', 'Euler')
option.set('iterations', '5')
option.set('ls_iterations', '8')
option.set('tolerance', '1e-8')
path = out / 'scene_hand2_reorient.xml'
ET.ElementTree(hand_root).write(path, encoding='utf-8', xml_declaration=False)
model = mujoco.MjModel.from_xml_path(str(path))
print(path)
print('nq',model.nq,'nv',model.nv,'nu',model.nu,'nbody',model.nbody,'ngeom',model.ngeom)
print('joints', [model.joint(i).name for i in range(model.njnt)][:5], '...', [model.joint(i).name for i in range(model.njnt)][-3:])
print('actuators', [model.actuator(i).name for i in range(model.nu)][:3], '...', [model.actuator(i).name for i in range(model.nu)][-3:])
