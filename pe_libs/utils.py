# Copyright 2024 SLAPaper
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

import pathlib

import transformers

from modules.shared import cmd_opts

# limitation of np.random.seed(), called from transformers.set_seed()
SEED_LIMIT_NUMPY = 2**32
neg_inf = -8192.0


def set_seed(seed: int) -> None:
    seed = int(seed) % SEED_LIMIT_NUMPY
    transformers.set_seed(seed)

curr_path: pathlib.Path = pathlib.Path(__file__).parent.parent.absolute()
model_path: pathlib.Path = curr_path / "models"

if hasattr(cmd_opts, "pe_model_path") and cmd_opts.pe_model_path:
    model_path = cmd_opts.pe_model_path