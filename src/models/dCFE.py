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
    def __init__(self, cfg: DictConfig, Data) -> None:
        """
        :param cfg:
        """
        super(dCFE, self).__init__()
        self.cfg = cfg

        # Set up MLP instance
        self.normalized_c = normalization(Data.c)
        self.MLP = MLP(self.cfg, Data)

        self.data = Data

        self.Cgw = torch.zeros(self.data.c.shape[:-1])
        self.satdk = torch.zeros(self.data.c.shape[:-1])

        # Initialize the CFE model
        self.ini_Cgw = (
            torch.ones(1, self.data.num_basins) * self.cfg.models.initial_params.Cgw
        )
        self.ini_satdk = (
            torch.ones(1, self.data.num_basins) * self.cfg.models.initial_params.satdk
        )
        self.cfe_instance = BMI_CFE(
            Cgw=self.ini_Cgw,
            satdk=self.ini_satdk,
            cfg=self.cfg,
            cfe_params=self.data.params,
        )
        self.cfe_instance.initialize()

        self.Cgw[:, 0] = self.ini_Cgw
        self.satdk[:, 0] = self.ini_satdk

    def initialize(self):
        # Initialize the CFE model with the dynamic parameter

        # Reset dCFE attributes
        self.reset_instance_attributes()

        # Reset CFE parameters, states, fluxes, and volume tracking
        self.cfe_instance.load_cfe_params()
        self.cfe_instance.reset_flux_and_states()
        self.cfe_instance.reset_volume_tracking()
        self.cfe_instance.remove_grad()

    def reset_instance_attributes(self):
        self.cfe_instance.Cgw = self.ini_Cgw.detach()
        self.cfe_instance.satdk = self.ini_satdk.detach()

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
        self.update_params(t)

        # Run the model with the NN-trained parameters (Cgw and satdk)
        self.cfe_instance.update()

        # Get the runoff output
        self.runoff = self.cfe_instance.return_runoff() * self.cfg.conversions.m_to_mm
        storage_states = self.cfe_instance.return_storage_states().transpose(
            dim0=0, dim1=1
        )
        return self.runoff, storage_states

    def update_params(self, t):
        self.cfe_instance.update_params(self.Cgw[:, t], self.satdk[:, t])

    def finalize(self):
        self.cfe_instance.finalize(print_mass_balance=True)

    def print(self):
        log.info(f"Cgw at timestep 0: {self.Cgw.tolist()[0][0]:.6f}")
        log.info(f"satdk at timestep 0: {self.satdk.tolist()[0][0]:.6f}")

    def mlp_forward(self, states, t) -> None:
        """
        A function to run MLP(). It sets the parameter values used within MC
        """

        lag_hrs = self.cfg.models.mlp.lag_hrs

        # Normalize states
        normalized_states = torch.zeros_like(states)

        # Normalize soil reservoir
        normalized_states[:, :, 0] = states[
            :, :, 0
        ] / self.cfe_instance.max_gw_storage.detach().transpose(dim0=0, dim1=1)

        # Normalize groundwater reservoir
        normalized_states[:, :, 1] = states[:, :, 1] / (
            self.cfe_instance.soil_params["D"].detach()
            * self.cfe_instance.soil_params["smcmax"].detach()
        ).transpose(dim0=0, dim1=1)

        # Concatinate with other attributes
        if t < lag_hrs:
            # when t is up to the lag ours, just repeat the c[t] for lag_hr times as input
            c = torch.cat(
                (
                    self.normalized_c[:, t, :].unsqueeze(dim=1).repeat(1, lag_hrs, 1),
                    normalized_states[:, t, :].unsqueeze(dim=1).repeat(1, lag_hrs, 1),
                ),
                dim=2,
            )
        else:
            # when t exceed the lag ours, take the c[t-lag_hr,t] as input
            c = torch.cat(
                (
                    self.normalized_c[:, (t - lag_hrs) : t, :],
                    normalized_states[:, t, :].unsqueeze(dim=1).repeat(1, lag_hrs, 1),
                ),
                dim=2,
            )

        # Run MLP
        self.Cgw[:, t], self.satdk[:, t] = self.MLP(c)
