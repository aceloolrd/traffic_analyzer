# utils/geometry.py

def should_calc_axle(
    x2: int,
    y2: int,
    frame_w: int,
    frame_h: int,
    axle_for_all: bool,
    hz_start: int,
    hz_stop: int,
    vz_start: int,
    vz_stop: int
) -> bool:
    """
    Определяет, нужно ли считать оси, в зависимости от зон (гориз./верт.).
    """
    if axle_for_all:
        return True
    ok_h = True
    ok_v = True
    if hz_start is not None and hz_stop is not None:
        d_right = frame_w - x2
        ok_h = (hz_stop <= d_right <= hz_start)
    if vz_start is not None and vz_stop is not None:
        d_bot = frame_h - y2
        ok_v = (vz_stop <= d_bot <= vz_start)
    return ok_h and ok_v
