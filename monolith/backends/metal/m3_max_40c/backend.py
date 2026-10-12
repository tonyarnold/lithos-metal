from ..base import MetalBackend


class Backend(MetalBackend):
    """Independent Apple9/40-core backend using shared native shaders.

    Kernel geometry is calibrated on this variant. No fusion recipes or
    serving context limit are inherited from the 30-core M3 Max.
    """
    id = "m3_max_40c"
