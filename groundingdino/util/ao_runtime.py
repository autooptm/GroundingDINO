import os


def _flag(name, default):
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "false", "False", "")


OPT = _flag("AO_OPT", True)
OPT_1 = OPT and _flag("AO_OPT_1", True)
OPT_2 = OPT and _flag("AO_OPT_2", True)
OPT_3 = OPT and _flag("AO_OPT_3", True)
SHAPE_ASSERT = (not OPT) or _flag("AO_SHAPE_ASSERT", False)
NAN_DEBUG = (not OPT) or _flag("AO_NAN_DEBUG", False)
