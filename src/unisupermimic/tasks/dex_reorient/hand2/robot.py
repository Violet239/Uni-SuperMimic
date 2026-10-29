"""Wuji Hand 2 asset declarations for the first UniLab smoke task."""

HAND2_JOINT_NAMES = [
    f"right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]
HAND2_ACTUATOR_NAMES = [f"{name}_actuator" for name in HAND2_JOINT_NAMES]
HAND2_BODY_NAMES = [
    "right_palm_link",
    *[f"right_finger{finger}_link{link}" for finger in range(1, 6) for link in range(1, 5)],
    *[f"right_finger{finger}_tip_sensor" for finger in range(1, 6)],
]
HAND2_SITE_NAMES = [f"right_finger{finger}_tip" for finger in range(1, 6)]

__all__ = [
    "HAND2_ACTUATOR_NAMES",
    "HAND2_BODY_NAMES",
    "HAND2_JOINT_NAMES",
    "HAND2_SITE_NAMES",
]
