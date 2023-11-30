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
# torch.autograd.set_detect_anomaly(True)
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from agents.base import BaseAgent
from data.Data import Data
from data.metrics import calculate_nse
from models.dCFE import dCFE
from utils.ddp_setup import find_free_port, cleanup
import shutil

log = logging.getLogger("agents.DifferentiableCFE")

# Set the RANK environment variable manually

# Refer to https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/agents/graph_network.py#L98
# self.model is https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/graph/models/GNN_baseline.py#L25


class DifferentiableCFE(BaseAgent):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the Differentiable LGAR code

        Sets up the initial state of the agent
        :param cfg:
        """
        super().__init__()

        self.cfg = cfg
        self.output_dir = self.create_output_dir()

        # Setting the cfg object and manual seed for reproducibility
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)

        # Defining the torch Dataset and Dataloader
        self.train_data = Data(self.cfg, "train")
        self.train_data_loader = DataLoader(
            self.train_data, batch_size=1, shuffle=False
        )
        self.validate_data = Data(self.cfg, "validate")
        self.validate_data_loader = DataLoader(
            self.validate_data, batch_size=1, shuffle=False
        )

        # Defining the model
        self.model = dCFE(
            cfg=self.cfg, TrainData=self.train_data, ValidateData=self.validate_data
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.models.hyperparameters.learning_rate
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=cfg.models.hyperparameters.step_size,
            gamma=cfg.models.hyperparameters.gamma,
        )

        self.current_epoch = 0

        # # Prepare for the DDP
        # free_port = find_free_port()
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = free_port

    def create_output_dir(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        dir_name = f"{current_date}_output"
        output_dir = os.path.join(self.cfg.output_dir, dir_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        return output_dir

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            log.info("You have entered CTRL+C.. Wait to finalize")

    def train(self) -> None:
        """
        Execute the training process.

        Sets the model to train mode, sets up DistributedDataParallel, and initiates training for the number of epochs
        specified in the configuration.
        """
        self.model.train()  # this .train() is a function from nn.Module
        # dist.init_process_group(
        #     backend="gloo",
        #     world_size=0,
        #     rank=self.cfg.num_processes,
        # )

        # Create the DDP object with the GLOO backend
        # self.net = DDP(self.model.to(self.cfg.device), device_ids=None)

        # Run process model once to get the internal states
        log.info("Initializing the model")
        self.model.initialize()
        self.run_model(run_mlp=False, period="train")

        for epoch in range(1, self.cfg.models.hyperparameters.epochs + 1):
            # TODO: Loop through basins
            log.info(f"Epoch #: {epoch}/{self.cfg.models.hyperparameters.epochs}")
            self.initialize_record()
            self.loss_record[epoch - 1] = self.train_one_epoch()
            self.save_weights_and_optimizer(epoch)
            if epoch % self.cfg.models.hyperparameters.validate_every == 0:
                self.loss_record_validate[epoch - 1], _ = self.validate()
            self.current_epoch += 1

    def initialize_record(self):
        self.loss_record = np.zeros(self.cfg.models.hyperparameters.epochs)
        self.loss_record_validate = np.zeros(self.cfg.models.hyperparameters.epochs)
        self.Cgw_train = np.empty(
            [self.train_data.num_basins, self.train_data.n_timesteps]
        )
        self.satdk_train = np.empty(
            [self.train_data.num_basins, self.train_data.n_timesteps]
        )
        self.Cgw_validate = np.empty(
            [self.train_data.num_basins, self.validate_data.n_timesteps]
        )
        self.satdk_validate = np.empty(
            [self.train_data.num_basins, self.validate_data.n_timesteps]
        )

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        # Reset
        self.optimizer.zero_grad()
        # Reset the model states and parameters, and gradients
        self.model.initialize()

        # Run model
        y_hat = self.run_model(run_mlp=True, period="train")

        # Run the following to get a visual image of tesnors
        #######
        # from torchviz import make_dot
        # a = make_dot(loss, params=self.model.c)
        # a.render("backward_computation_graph")
        #######

        # Calculate the loss
        loss = self.evaluate(
            y_hat, self.train_data.y, self.Cgw_train, self.satdk_train, "train"
        )

        return loss

    def run_model(self, run_mlp=False, period="train"):
        if period == "train":
            n_timesteps = self.train_data.n_timesteps
            num_basins = self.train_data.num_basins
            dataloader = self.train_data_loader
        elif period == "validate":
            n_timesteps = self.validate_data.n_timesteps
            num_basins = self.validate_data.num_basins
            dataloader = self.validate_data_loader

        # initialize
        y_hat = torch.empty(
            [num_basins, n_timesteps],
            device=self.cfg.device,
        )
        y_hat.fill_(float("nan"))

        # Run CFE at each timestep
        for t, (x, y_t) in enumerate(tqdm(dataloader, desc=period)):
            if run_mlp:
                self.model.mlp_forward(t, period)  # Instead
                if period == "train":
                    self.Cgw_train[:, t] = self.model.Cgw.detach().numpy()
                    self.satdk_train[:, t] = self.model.satdk.detach().numpy()
                elif period == "validate":
                    self.Cgw_validate[:, t] = self.model.Cgw.detach().numpy()
                    self.satdk_validate[:, t] = self.model.satdk.detach().numpy()
            runoff = self.model(x, t)
            y_hat[:, t] = runoff

        return y_hat

    def evaluate(
        self, y_hat_: Tensor, y_t_: Tensor, Cgw_record, satdk_record, period
    ) -> None:
        """
        One cycle of model validation
        This function calculates the loss for the given predicted and actual values,
        backpropagates the error, and updates the model parameters.

        Parameters:
        - y_hat_ : The tensor containing predicted values
        - y_t_ : The tensor containing actual values.
        """

        # Transform validation/output data for validation
        y_t_ = y_t_.squeeze(dim=2)
        warmup = self.cfg.models.hyperparameters.warmup
        y_hat = y_hat_[:, warmup:]
        y_t = y_t_[:, warmup:]

        y_hat_np = y_hat_.detach().numpy()
        y_t_np = y_t_.detach().numpy()

        # Save results
        # Evaluate
        kge = he.evaluator(he.kge, y_hat_np[0], y_t_np[0])
        log.info(
            f"trained KGE for the basin {self.train_data.basin_ids[0]}: {float(kge[0]):.8}"
        )

        self.save_result(
            y_hat=y_hat_np,
            y_t=y_t_np,
            Cgw_record=Cgw_record,
            satdk_record=satdk_record,
            out_filename=f"epoch{self.current_epoch+1}",
            period=period,
            plot_figure=False,
        )

        # Compute the overall loss
        mask = torch.isnan(y_t)
        y_t_dropped = y_t[~mask]
        y_hat_dropped = y_hat[~mask]
        if y_hat_dropped.shape != y_t_dropped.shape:
            print("y_t and y_hat shape not matching")

        print("calculate loss")
        # TODO: try different loss for the validation
        loss = self.criterion(y_hat_dropped, y_t_dropped)
        log.info(f"loss at epoch {self.current_epoch+1} ({period}): {loss:.8f}")

        if period == "train":
            # Backpropagate the error
            start = time.perf_counter()
            print("Loss backward starts")
            loss.backward()
            print("Loss backward ends")
            end = time.perf_counter()

            # Log the time taken for backpropagation and the calculated loss
            log.debug(f"Back prop took : {(end - start):.6f} seconds")
            log.debug(f"Loss: {loss}")

            # Update the model parameters
            self.model.print()
            print("Start optimizer")
            self.optimizer.step()
            print("End optimizer")
            self.scheduler.step()
            print("Current Learning Rate:", self.optimizer.param_groups[0]["lr"])

        return loss

    def save_weights_and_optimizer(self, epoch: int):
        weight_path = os.path.join(self.output_dir, f"model_epoch{epoch:03d}.pt")
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = os.path.join(
            self.output_dir, f"optimizer_state_epoch{epoch:03d}.pt"
        )
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    def validate(self):
        with torch.no_grad():
            y_hat = self.run_model(run_mlp=True, period="validate")
        loss = self.evaluate(
            y_hat,
            self.validate_data.y,
            self.Cgw_validate,
            self.satdk_validate,
            "validate",
        )

        return loss, y_hat

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        try:
            self._finalize_process("train")
            self._finalize_process("validate")
            self._save_loss()
        except:
            raise NotImplementedError

    def _finalize_process(self, period):
        # __________________________________________________
        # Get the final training
        self.model.cfe_instance.reset_volume_tracking()
        self.model.cfe_instance.reset_flux_and_states()

        # Run one last time
        if period == "train":
            y_hat_ = self.run_model(run_mlp=False, period="train")
            y_t_ = self.train_data.y
            Cgw_record = self.Cgw_train
            satdk_record = self.satdk_train
        elif period == "validate":
            _, y_hat_ = self.validate()
            y_t_ = self.validate_data.y
            Cgw_record = self.Cgw_validate
            satdk_record = self.satdk_validate

        y_hat = y_hat_.detach().numpy()
        y_t = y_t_.detach().numpy()

        self.save_result(
            y_hat=y_hat,
            y_t=y_t,
            Cgw_record=Cgw_record,
            satdk_record=satdk_record,
            out_filename=f"final_result_{period}",
            plot_figure=True,
        )

        if period == "validate":
            print(self.model.finalize())

    def _save_loss(self):
        for period, loss_record in [
            ("train", self.loss_record),
            ("validate", self.loss_record_validate),
        ]:
            df = pd.DataFrame(loss_record)
            file_path = os.path.join(self.output_dir, f"final_result_loss_{period}.csv")
            df.to_csv(file_path)

        fig, axes = plt.subplots()
        # Create the x-axis values
        epoch_list = list(range(1, len(self.loss_record) + 1))

        # Plotting
        fig, axes = plt.subplots()
        axes.plot(epoch_list, self.loss_record, "-", label="training (MSE)")
        axes.plot(epoch_list, self.loss_record_validate, "--", label="validation (MSE)")
        axes.set_title(
            f"Initial learning rate: {self.cfg.models.hyperparameters.learning_rate}"
        )
        axes.set_ylabel("loss")
        axes.set_xlabel("epoch")
        fig.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"final_result_loss.png"))
        plt.close()

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def save_result(
        self,
        y_hat,
        y_t,
        Cgw_record,
        satdk_record,
        out_filename,
        period,
        plot_figure=False,
    ):
        # Save all basin runs

        warmup = self.cfg.models.hyperparameters.warmup

        for i, basin_id in enumerate(self.train_data.basin_ids):
            # Save the timeseries of runoff and the best dynamic parametersers
            data = {
                "y_hat": y_hat[i, warmup:].squeeze(),
                "y_t": y_t[i, warmup:].squeeze(),
                "Cgw": Cgw_record[i, warmup:],
                "satdk": satdk_record[i, warmup:],
            }
            df = pd.DataFrame(data)
            df.to_csv(
                os.path.join(
                    self.output_dir, f"{out_filename}_{period}_{basin_id}.csv"
                ),
                index=False,
            )

            if period == "train":
                start_time = self.train_data.start_time
                end_time = self.train_data.end_time
            elif period == "validate":
                start_time = self.validate_data.start_time
                end_time = self.validate_data.end_time
            time_range = pd.date_range(start_time, end_time, freq="H")

            if plot_figure:
                # Plot
                eval_metrics = he.evaluator(he.kge, y_hat[i], y_t[i])[0]
                fig, axes = plt.subplots(figsize=(5, 5))
                if self.cfg.run_type == "ML_synthetic_test":
                    eval_label = "evaluation (synthetic)"
                elif self.cfg.run_type == "ML":
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
                axes.set_title(
                    f"{period} period - {self.cfg.soil_scheme} soil scheme \n (KGE={float(eval_metrics):.2})"
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
                    os.path.join(
                        self.output_dir, f"{out_filename}_{period}_{basin_id}.png"
                    )
                )
                plt.close()
