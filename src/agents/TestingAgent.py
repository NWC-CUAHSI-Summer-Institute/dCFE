import logging
from omegaconf import DictConfig
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

        # Output testing results to the same directory
        self.output_dir = cfg.test.input_dir

        # Read testing data
        self.test_data = Data(self.cfg, "test")
        self.test_data_loader = DataLoader(self.test_data, batch_size=1, shuffle=False)

        # Read model states
        self.epoch = self.cfg.test.epoch_to_use
        model_path = os.path.join(cfg.test.input_dir, f"model_epoch{self.epoch:03d}.pt")
        model_state_dict = torch.load(model_path)

        # Instantiate the dCFE model with the saved states
        self.trained_model = dCFE(self.cfg, TestData=self.test_data)
        self.trained_model.load_state_dict(model_state_dict)
        self.trained_model.eval()

    def run(self):
        print("Initializing CFE state")
        self.initialize()
        # If the parameter needs to be updated (the testing run is done on different basin), overwrite the train_model here?

        print("Testing run")
        y_hat_, Cgw_test, satdk_test = self.forward(
            self.test_data, self.trained_model, run_mlp=True
        )

        # Save results
        y_hat = y_hat_.detach().numpy()

        self.save_result(
            y_hat,
            self.test_data.y,
            Cgw_test,
            satdk_test,
            f"final_result_test_epoch{self.epoch}",
            True,
        )

    def initialize(self):
        self.trained_model.initialize()
        # Run CFE once to get state variables
        self.forward(self.test_data, self.trained_model, run_mlp=False)

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
        Cgw_test = np.empty([num_basins, n_timesteps])
        satdk_test = np.empty([num_basins, n_timesteps])

        # Run CFE at each timestep
        for t, (x, _) in enumerate(tqdm(dataloader, desc="test")):
            if run_mlp:
                with torch.no_grad():
                    model.mlp_forward(t, "test")
                    Cgw_test[:, t] = model.Cgw.detach().numpy()
                    satdk_test[:, t] = model.satdk.detach().numpy()
            runoff = model(x)
            y_hat[:, t] = runoff

        return y_hat, Cgw_test, satdk_test

    def finalize(self):
        None

    def save_result(
        self, y_hat, y_t, Cgw_record, satdk_record, out_filename, plot_figure=False
    ):
        warmup = self.cfg.models.hyperparameters.warmup

        for i, basin_id in enumerate(self.test_data.basin_ids):
            # Save the timeseries of runoff and the best dynamic parametersers
            data = {
                "y_hat": y_hat[i, warmup:].squeeze(),
                "y_t": y_t[i, warmup:].squeeze(),
                "Cgw": Cgw_record[i, warmup:],
                "satdk": satdk_record[i, warmup:],
            }
            df = pd.DataFrame(data)
            df.to_csv(
                os.path.join(self.output_dir, f"{out_filename}_{basin_id}.csv"),
                index=False,
            )

            if plot_figure:
                time_range = pd.date_range(
                    self.test_data.start_time, self.test_data.end_time, freq="H"
                )

                eval_metrics = he.evaluator(he.kge, y_hat[i], y_t[i])[0]

                fig, axes = plt.subplots(figsize=(5, 5))
                if self.cfg.run_type == "ML_synthetic_test":
                    eval_label = "evaluation (synthetic)"
                else:
                    eval_label = "observed"

                axes.plot(
                    time_range[warmup:],
                    y_t[i, warmup:],
                    "-",
                    label=eval_label,
                    alpha=0.5,
                )
                axes.plot(
                    time_range[warmup:],
                    y_hat[i, warmup:],
                    "--",
                    label="predicted",
                    alpha=0.5,
                )
                axes.set_xlabel("Time")  # Replace with your actual x-axis label
                axes.set_ylabel("Flow [mm/hr]")  # Replace with your actual y-axis label
                axes.set_title(
                    f"Test period - {self.cfg.soil_scheme} soil scheme \n (KGE={float(eval_metrics):.2})"
                )

                # Rotate date labels for better readability
                plt.setp(axes.xaxis.get_majorticklabels(), rotation=45, ha="right")

                # Improve spacing between date labels
                axes.xaxis.set_major_locator(mdates.AutoDateLocator())

                # Set formatter for date labels
                axes.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_dir, f"{out_filename}_{basin_id}.png")
                )
                plt.close()
