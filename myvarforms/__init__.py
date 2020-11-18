# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# from qiskit.aqua.components.variational_forms import VariationalForm
# from .ry import RY
# from .ryrz import RYRZ
# from .swaprz import SwapRZ
from .ryrzent1 import RYRZENT1
from .ryrzent2 import RYRZENT2
from .ryrzent3 import RYRZENT3
from .ryrzent1_evolv import RYRZENT1_EVOLV
from .FFFT_varform import FFFT_varform
from .FFFT_varform_2nd_order import FFFT_varform_2nd_order
from .FFFT_varform_2sites import FFFT_varform_2sites
from .VHA import VHA
from .VHA_Hubbard import VHA_Hubbard
from .VHA_Hubbard_1p import VHA_Hubbard_1p
from .VHA_Hubbard_2p import VHA_Hubbard_2p
from .VHA_Hubbard_2p_2 import VHA_Hubbard_2p_2
from .VHA_Hubbard_2p_3 import VHA_Hubbard_2p_3
from .short_QOCA import short_QOCA
from .drive import drive
from .gates import *

__all__ = ['RYRZENT1',
           'RYRZENT2',
           'RYRZENT3',
           'RYRZENT1_EVOLV',
           'FFFT_varform',
           'FFFT_varform_2nd_order',
           'FFFT_varform_2sites',
           'VHA',
           'VHA_Hubbard',
           'VHA_Hubbard_1p',
           'VHA_Hubbard_2p',
           'VHA_Hubbard_2p_2',
           'VHA_Hubbard_2p_3',
           'short_QOCA',
           'drive',
           'gates']