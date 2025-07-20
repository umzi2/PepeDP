from enum import Enum
from pepedp.scripts.iqa import (
    HyperThreshold,
    AnIQAThreshold,
    TopIQThreshold,
    BlockinessThreshold,
    IC9600Threshold,
)


class ThresholdAlg(Enum):
    HIPERIQA = HyperThreshold
    ANIIQA = AnIQAThreshold
    TOPIQ = TopIQThreshold
    BLOCKINESS = BlockinessThreshold
    IC9600 = IC9600Threshold
