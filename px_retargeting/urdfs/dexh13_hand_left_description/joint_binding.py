
import math


def calculate_dip_from_pip(pip_joint):
    inp = [pip_joint, 5.107837115649, 17.334936, 4.999989, 16.799683, 18.179658962698, 4.999975, 18.179231, 5.000014, 13.389921583041, 22.243296728003, 14.704423681722]
    t2 = inp[1] * inp[1]
    t3 = inp[5] * inp[5]
    t4 = inp[2] * inp[2]
    t5 = inp[9] * inp[9]
    t8 = inp[3] * inp[3]
    t10 = inp[8] * inp[8]
    t12 = 1.0 / inp[5]
    t13 = 1.0 / inp[2]
    t14 = 1.0 / inp[9]
    t16 = 1.0 / inp[3]
    t17 = 1.0 / inp[8]
    t41 = t2 + t4
    t38 = inp[1] * inp[2] * math.cos(inp[0] + math.acos(1.0 / inp[1] * t13 * (t41 - t5) / 2.0)) * 2.0
    t41 -= t38
    t42 = 1.0 / math.sqrt(t41)
    t51 = inp[4] * inp[4]
    t51_tmp = t3 + inp[6] * inp[6]
    t51 = inp[5] * inp[6] * math.cos((((math.acos(t13 * t14 * ((t4 + t5) - t2) / 2.0) + math.acos(t12 * (1.0 / inp[6]) * (t51_tmp - inp[10] * inp[10]) / 2.0)) - math.acos(t14 * t16 * ((t5 + t8) - t51) / 2.0)) - math.acos(t13 * (t4 * 2.0 - t38) * t42 / 2.0)) + math.acos(t16 * t42 * ((t8 - t51) + t41) / 2.0)) * 2.0
    t41 = t51_tmp - t51
    t42 = 1.0 / math.sqrt(t41)
    return (math.acos(t17 * t42 * ((t10 + t41) - inp[7] * inp[7]) / 2.0) - math.acos(t12 * t17 * ((t3 + t10) - inp[11] * inp[11]) / 2.0)) + math.acos(t12 * t42 * (t3 * 2.0 - t51) / 2.0)



joint_bindings = {
    "left_index_joint_3": {
        "source_joint": "left_index_joint_2",
        "mapping_func": calculate_dip_from_pip,
    },
    "left_middle_joint_3": {
        "source_joint": "left_middle_joint_2",
        "mapping_func": calculate_dip_from_pip,
    },
    "left_ring_joint_3": {
        "source_joint": "left_ring_joint_2",
        "mapping_func": calculate_dip_from_pip,
    },
    "left_thumb_joint_3": {
        "source_joint": "left_thumb_joint_2",
        "mapping_func": calculate_dip_from_pip,
    },
}