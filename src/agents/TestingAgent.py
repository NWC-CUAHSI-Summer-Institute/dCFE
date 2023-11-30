import logging
from omegaconf import DictConfig
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import hydroeval as he
import os
import pandas as pd

import torch

torch.set_default_dtype(torch.float64)
from torch.utils.data import DataLoader

from data.Data import Data
from models.dCFE import dCFE
from utils.transform import normalization

log = logging.getLogger("agents.DifferentiableCFE")

# Set the RANK environment variable manually

# Refer to https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/agents/graph_network.py#L98
# self.model is https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/graph/models/GNN_baseline.py#L25


class TestingAgent:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.output_dir = self.create_output_dir()

        # Read testing data
        self.test_data = Data(self.cfg, "test")
        self.test_data_loader = DataLoader(self.test_data, batch_size=1, shuffle=False)

        # Read model states
        model_path = os.path.join(
            cfg.test.input_dir, f"model_epoch{self.test.epoch_to_use:03d}.pt"
        )
        model_state_dict = torch.load(model_path)

        # Instantiate the dCFE model
        self.trained_model = dCFE(self.cfg, TrainData=None, ValidateData=None)
        self.trained_model.load_state_dict(model_state_dict)
        self.trained_model.eval()

    def run(self):
        self.trained_model.initialize()
        self.forward(self.test_data, self.trained_model, run_mlp=False)
        y_hat = self.forward(self.test_data, self.trained_model, run_mlp=True)
        self.plot_hydrograph(y_hat)

    def forward(self, data, model, run_mlp=False):
        n_timesteps = data.n_timesteps
        num_basins = data.num_basins
        dataloader = self.test_data_loader

        # initialize
        y_hat = torch.empty(
            [num_basins, n_timesteps],
            device=self.cfg.device,
        )
        y_hat.fill_(float("nan"))

        test_normalized_c = normalization(data.c, data.min_c, data.max_c)
        # Run CFE at each timestep
        for t, (x, _) in enumerate(tqdm(dataloader, desc="test")):
            if run_mlp:
                self.model.mlp_forward(t, "test", test_normalized_c)
                self.Cgw_validate[:, t] = self.model.Cgw.detach().numpy()
                self.satdk_validate[:, t] = self.model.satdk.detach().numpy()
            runoff = self.model(x, t)
            y_hat[:, t] = runoff

        return y_hat

    def finalize(self):
        None

    def plot_hydrograph(self):
        None


"""
import torch
import numpy as np
from omegaconf import OmegaConf
from models.dCFE import dCFE

# Load the saved model
model_path = "path_to_your_saved_model.pt"  # Replace with the actual path
model_state_dict = torch.load(model_path)

# Create a configuration dictionary
config_path = "path_to_your_config_file.yaml"  # Replace with the actual path
config = OmegaConf.load(config_path)

# Instantiate the dCFE model
model = dCFE(config, TrainData=None, ValidateData=None)
model.load_state_dict(model_state_dict)
model.eval()

# Prepare input data for testing
# You should prepare your own test input data (precipitation and PET)
# For example, create a tensor 'input_data' of shape (batch_size, sequence_length, 2)
# where the last dimension contains precipitation and PET values.

# Convert input data to PyTorch tensor if it's not already
input_data = torch.tensor(input_data)

# Run the model on the input data
with torch.no_grad():
    output = model(input_data)

# Post-process the output as needed
# 'output' contains the model's predictions for runoff

# Example post-processing:
# Convert the output from mm/h to your desired units
output = output.numpy()  # Convert to NumPy array if needed
output = (
    output * conversion_factor
)  # Replace 'conversion_factor' with your desired conversion

# Now 'output' contains the model's predictions for runoff in your desired units
"""
