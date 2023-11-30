"""                                                                      
            dddddddd                                                                
            d::::::d       CCCCCCCCCCCCCFFFFFFFFFFFFFFFFFFFFFFEEEEEEEEEEEEEEEEEEEEEE
            d::::::d    CCC::::::::::::CF::::::::::::::::::::FE::::::::::::::::::::E
            d::::::d  CC:::::::::::::::CF::::::::::::::::::::FE::::::::::::::::::::E
            d:::::d  C:::::CCCCCCCC::::CFF::::::FFFFFFFFF::::FEE::::::EEEEEEEEE::::E
    ddddddddd:::::d C:::::C       CCCCCC  F:::::F       FFFFFF  E:::::E       EEEEEE
  dd::::::::::::::dC:::::C                F:::::F               E:::::E             
 d::::::::::::::::dC:::::C                F::::::FFFFFFFFFF     E::::::EEEEEEEEEE   
d:::::::ddddd:::::dC:::::C                F:::::::::::::::F     E:::::::::::::::E   
d::::::d    d:::::dC:::::C                F:::::::::::::::F     E:::::::::::::::E   
d:::::d     d:::::dC:::::C                F::::::FFFFFFFFFF     E::::::EEEEEEEEEE   
d:::::d     d:::::dC:::::C                F:::::F               E:::::E             
d:::::d     d:::::d C:::::C       CCCCCC  F:::::F               E:::::E       EEEEEE
d::::::ddddd::::::dd C:::::CCCCCCCC::::CFF:::::::FF           EE::::::EEEEEEEE:::::E
 d:::::::::::::::::d  CC:::::::::::::::CF::::::::FF           E::::::::::::::::::::E
  d:::::::::ddd::::d    CCC::::::::::::CF::::::::FF           E::::::::::::::::::::E
   ddddddddd   ddddd       CCCCCCCCCCCCCFFFFFFFFFFF           EEEEEEEEEEEEEEEEEEEEEE
"""

from omegaconf import DictConfig
import logging
import time
from tqdm import tqdm
import torch

torch.set_default_dtype(torch.float64)
from torch import Tensor
import torch.nn as nn
from models.physics.bmi_cfe import BMI_CFE
import pandas as pd
import numpy as np
from utils.transform import normalization, to_physical
from models.MLP import MLP

log = logging.getLogger("models.dCFE")


class dCFE(nn.Module):
    def __init__(
        self, cfg: DictConfig, TrainData=None, ValidateData=None, TestData=None
    ) -> None:
        """
        :param cfg:
        """
        super(dCFE, self).__init__()
        self.cfg = cfg

        # Set up MLP instance
        self.MLP = MLP(self.cfg)

        if (cfg.run_type == "ML") | (cfg.run_type == "ML_synthetic"):
            # Get c
            self.normalized_c_train = normalization(
                TrainData.c, TrainData.min_c, TrainData.max_c
            )
            # TODO: normalized based on training data
            self.normalized_c_validate = normalization(
                ValidateData.c, ValidateData.min_c, ValidateData.max_c
            )

            self.data = TrainData
            self.data_validate = ValidateData

        elif cfg.run_type == "ML_test":
            self.data = TestData

        # Initialize the CFE model
        self.Cgw = torch.ones(self.data.num_basins) * self.cfg.models.initial_params.Cgw
        self.satdk = (
            torch.ones(self.data.num_basins) * self.cfg.models.initial_params.satdk
        )
        self.cfe_instance = BMI_CFE(
            Cgw=self.Cgw,
            satdk=self.satdk,
            cfg=self.cfg,
            cfe_params=self.data.params,
        )
        self.cfe_instance.initialize()

    def initialize(self):
        # Initialize the CFE model with the dynamic parameter
        self.reset_instance_attributes()

        # Reset CFE parameters, states, fluxes, and volume tracking
        self.cfe_instance.load_cfe_params()
        self.cfe_instance.reset_flux_and_states()
        self.cfe_instance.reset_volume_tracking()
        self.cfe_instance.reset_internal_attributes()

    def reset_instance_attributes(self):
        self.cfe_instance.Cgw = self.Cgw.detach()
        self.cfe_instance.satdk = self.satdk.detach()
        if (self.cfg.run_type == "ML") | (self.cfg.run_type == "ML_synthetic"):
            self.normalized_c_train = self.normalized_c_train.detach()

    def forward(self, x, t):  # -> (Tensor, Tensor):
        """
        The forward function to model runoff through CFE model
        :param x: Precip and PET forcings (m/h)
        :return: runoff to be used for validation (mm/h)
        """

        # Read the forcing
        precip = x[:, :, 0]
        pet = x[:, :, 1]

        # Set precip and PET values in CFE
        self.cfe_instance.set_value(
            "atmosphere_water__time_integral_of_precipitation_mass_flux", precip
        )
        self.cfe_instance.set_value("water_potential_evaporation_flux", pet)

        # Update dynamic parameters in CFE
        self.update_params()

        # Run the model with the NN-trained parameters (Cgw and satdk)
        self.cfe_instance.update()

        # Get the runoff output
        runoff = self.cfe_instance.return_runoff() * self.cfg.conversions.m_to_mm

        return runoff

    def update_params(self):
        self.cfe_instance.update_params(self.Cgw, self.satdk)
        if np.random.random() < 0.0005:
            print(f"dCFE line 111 --- Cgw: {self.Cgw}, satdk: {self.satdk}")

    def finalize(self):
        self.cfe_instance.finalize(print_mass_balance=True)

    def print(self):
        None
        # log.info(f"Cgw at timestep 0: {self.Cgw.tolist()[0][0]:.6f}")
        # log.info(f"satdk at timestep 0: {self.satdk.tolist()[0][0]:.6f}")

    def mlp_forward(self, t, period, test_normalized_c=None) -> None:
        """
        A function to run MLP(). It sets the parameter values used within MC
        """

        if period == "train":
            normalized_c = self.normalized_c_train
        elif period == "validate":
            normalized_c = self.normalized_c_validate
        elif period == "test":
            normalized_c = test_normalized_c

        lag_hrs = self.cfg.models.mlp.lag_hrs

        states = self.cfe_instance.return_storage_states().transpose(dim0=0, dim1=1)

        # Normalize states
        normalized_states = torch.zeros_like(states)

        # Normalize soil reservoir
        normalized_states[:, 0] = (
            states[:, 0] / self.cfe_instance.max_gw_storage.detach()
        )

        # Normalize groundwater reservoir
        normalized_states[:, 1] = states[:, 1] / (
            self.cfe_instance.soil_params["D"].detach()
            * self.cfe_instance.soil_params["smcmax"].detach()
        )

        # Concatinate with other attributes
        if t < lag_hrs:
            # when t is up to the lag ours, just repeat the c[t] for lag_hr times as input
            c = torch.cat(
                (
                    normalized_c[:, t, :].unsqueeze(dim=1).repeat(1, lag_hrs, 1),
                    normalized_states.unsqueeze(dim=1).repeat(1, lag_hrs, 1),
                ),
                dim=2,
            )
        else:
            # when t exceed the lag ours, take the c[t-lag_hr,t] as input
            c = torch.cat(
                (
                    normalized_c[:, (t - lag_hrs) : t, :],
                    normalized_states.unsqueeze(dim=1).repeat(1, lag_hrs, 1),
                ),
                dim=2,
            )

        # Run MLP
        _Cgw, _satdk = self.MLP(c)
        self.Cgw = _Cgw.clone()
        self.satdk = _satdk.clone()
