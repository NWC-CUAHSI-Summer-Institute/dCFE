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

        # Initialize the CFE model
        self.Cgw = torch.zeros([self.normalized_c.shape[0]])
        self.satdk = torch.zeros([self.normalized_c.shape[0]])
        self.cfe_instance = BMI_CFE(
            Cgw=self.Cgw,
            satdk=self.satdk,
            cfg=self.cfg,
            cfe_params=Data.params,
        )
        self.cfe_instance.initialize()

        ## Set initial paramesters for the prediction of 1st epoch
        self.Cgw = (
            torch.ones(self.data.c.shape[:-1]) * self.cfg.models.initial_params.Cgw
        )
        self.satdk = (
            torch.ones(self.data.c.shape[:-1]) * self.cfg.models.initial_params.satdk
        )

    def initialize(self):
        # Initialize the CFE model with the dynamic parameter

        # Reset dCFE attributes
        self.reset_instance_attributes()

        # Reset CFE parameters, states, fluxes, and volume tracking
        self.cfe_instance.load_cfe_params()
        self.cfe_instance.reset_flux_and_states()
        self.cfe_instance.reset_volume_tracking()

        # Update parameters
        self.cfe_instance.update_params(self.Cgw[:, 0], self.satdk[:, 0])

    def reset_instance_attributes(self):
        self.cfe_instance.Cgw = torch.zeros_like(self.cfe_instance.Cgw)
        self.cfe_instance.satdk = torch.zeros_like(self.cfe_instance.satdk)

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
        self.cfe_instance.update_params(self.Cgw[:, t], self.satdk[:, t])

        # Run the model with the NN-trained parameters (Cgw and satdk)
        self.cfe_instance.update()

        # Get the runoff output
        self.runoff = self.cfe_instance.return_runoff() * self.cfg.conversions.m_to_mm
        storage_states = self.cfe_instance.return_storage_states().transpose(
            dim0=0, dim1=1
        )
        return self.runoff, storage_states

    def finalize(self):
        self.cfe_instance.finalize(print_mass_balance=True)

    def print(self):
        log.info(f"Cgw at timestep 0: {self.Cgw.tolist()[0][0]:.6f}")
        log.info(f"satdk at timestep 0: {self.satdk.tolist()[0][0]:.6f}")

    def mlp_forward(self, states) -> None:
        """
        A function to run MLP(). It sets the parameter values used within MC
        """

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
        c = torch.cat((self.normalized_c, normalized_states), dim=2)

        # Run MLP
        self.Cgw, self.satdk = self.MLP(c)
