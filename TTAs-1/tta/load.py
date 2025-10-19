import os
import typing

from utils.load import get_class_in_module
from .base import IStrategy


SRC_DIR = "tta"

BASIC = {
    "none": (f"{SRC_DIR}/suta.py", "NoAdaptStrategy"),
    "suta": (f"{SRC_DIR}/suta.py", "SUTAStrategy"),
    "csuta": (f"{SRC_DIR}/suta.py", "CSUTAStrategy"),
    "cea": (f"{SRC_DIR}/cea.py", "CEAStrategy"),
    "tent": (f"{SRC_DIR}/tent.py", "TNETStrategy"),
}

DSUTA = {
    "dsuta": (f"{SRC_DIR}/dsuta.py", "DSUTAStrategy"),
    "dsuta_reset": (f"{SRC_DIR}/dsuta_reset.py", "DSUTAResetStrategy"),
}

OTHER = {
    "awmc": (f"{SRC_DIR}/awmc.py", "AWMCStrategy"),
    "sgem": (f"{SRC_DIR}/sgem.py", "SGEMStrategy"),
    "t3a-hubert": (f"{SRC_DIR}/t3a-hubert.py", "T3AStrategy"),
    "t3a-wav2vec2": (f"{SRC_DIR}/t3a-wav2vec2.py", "T3AStrategy"),
    "lame": (f"{SRC_DIR}/lame.py", "LAMEStrategy"),
    "foa-hubert": (f"{SRC_DIR}/foa-hubert.py", "FOAStrategy"),
    "foa-wav2vec2": (f"{SRC_DIR}/foa-wav2vec2.py", "FOAStrategy"),
    "eata": (f"{SRC_DIR}/eata.py", "EATAStrategy"),}

STRATEGY_MAPPING = {
    **BASIC,
    **DSUTA,
    **OTHER,
}


def get_tta_cls(name) -> typing.Type[IStrategy]:
    module_path, class_name = STRATEGY_MAPPING[name]
    return get_class_in_module(class_name, module_path)
