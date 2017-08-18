import numpy as np

M_srgb_to_xyz = np.array(
    [[0.4124564,  0.3575761,  0.1804375],
     [0.2126729,  0.7151522,  0.0721750],
     [0.0193339,  0.1191920,  0.9503041]])
xyz_ref_white = np.array([0.95047, 1., 1.08883])
# using reference white D65

def rgb_to_xyz(rgb):
    _rgb = rgb.copy()
    mask = _rgb > 0.04045
    _rgb[mask] = np.power(
        (_rgb[mask] + 0.055) / 1.055, 2.4)
    _rgb[~mask] /= 12.92
    return _rgb.dot(M_srgb_to_xyz.T)

def xyz_to_lab(rgb):
    _rgb = rgb.copy()
    _rgb /= xyz_ref_white

    mask = _rgb > 0.008856
    _rgb[mask] = np.power(_rgb[mask], 1. / 3.)
    _rgb[~mask] = 7.787 * _rgb[~mask] + 16. / 116.

    L = (116. * _rgb[1]) - 16.
    a = 500.0 * (_rgb[0] - _rgb[1])
    b = 200.0 * (_rgb[1] - _rgb[2])

    return np.array([L, a, b])

def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))

