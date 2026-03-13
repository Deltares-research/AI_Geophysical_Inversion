# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import HDF5GridDataset

from physicsnemo.sym.utils.io.plotter import GridValidatorPlotter

import sys
import os

pth_utilities = '/home/edwinobando/PINN/physicsnemo-sym/examples/darcy/utilities.py'
if os.path.exists(pth_utilities):
    sys.path.append(os.path.dirname(pth_utilities))
else:
    raise FileNotFoundError(f"File not found: {pth_utilities}")


path_conf = '/home/edwinobando/PINN/01_AI_Geophysics_Inversion/MODELS_2D_Reconstructed/CASE_I/config/'

@physicsnemo.sym.main(config_path=path_conf, config_name="config_FNO")
def run(cfg: PhysicsNeMoConfig) -> None:
    # [keys]
    # load training/ test data

    X_mean= -5.973577188456147e-10 
    X_std = 8.735661685932428e-05
    Y_mean = 188.53571745319738 
    Y_std = 48.1046107577747


    input_keys = [Key("X", scale=(0, 1))]
    output_keys = [Key("Y", scale=(0,  1))]

    train_path = to_absolute_path(
        "/home/edwinobando/PINN/01_AI_Geophysics_Inversion/database/hdf5_files/dataset_CASE_II_source_56m_train.hdf5"
    )
    test_path = to_absolute_path(
        "/home/edwinobando/PINN/01_AI_Geophysics_Inversion/database/hdf5_files/dataset_CASE_II_source_56m_test.hdf5"
    )
    # [keys]

    # [datasets]
    # make datasets
    train_dataset = HDF5GridDataset(
        train_path, invar_keys=["X"], outvar_keys=["Y"], n_examples=2000
    )
    test_dataset = HDF5GridDataset(
        test_path, invar_keys=["X"], outvar_keys=["Y"], n_examples=100
    )
    # [datasets]

    # [init-model]
    # make list of nodes to unroll graph on
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )


    
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=input_keys,
        decoder_net=decoder_net,
        activation_fn = 'tanh',
        decoder_activation_fn = 'tanh',
    )

    

    nodes = [fno.make_node("fno")]

    
    # [init-model]

    # [constraint]
    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
        num_workers=4,  # number of parallel data loaders

    )
    domain.add_constraint(supervised, "supervised")
    # [constraint]

    # [validator]
    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
    )
    domain.add_validator(val, "test")
    # [validator]

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

    

if __name__ == "__main__":
    run()