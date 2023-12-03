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
from data.Data import Data, BatchData
from data.metrics import calculate_nse
from models.dCFE import dCFE
from utils.ddp_setup import find_free_port, cleanup
import shutil

log = logging.getLogger("agents.DifferentiableCFE")

# Set the RANK environment variable manually

# Refer to https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/agents/graph_network.py#L98
# self.model is https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/graph/models/GNN_baseline.py#L25


# def custom_collate_fn(batch):
#     # Extract inputs and targets from the batch
#     inputs, targets = zip(*batch)
#     # Convert to PyTorch tensors
#     inputs = torch.tensor(inputs)
#     targets = torch.tensor(targets)
#     return inputs, targets


class DifferentiableCFE(BaseAgent):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the Differentiable LGAR code

        Sets up the initial state of the agent
        :param cfg:
        """
        super().__init__()

        # _________________________________________________________________________
        self.cfg = cfg
        self.output_dir = self.create_output_dir()

        # Setting the cfg object and manual seed for reproducibility
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)
        self.current_epoch = 0

        # Initialize the record
        self.loss_record = np.zeros(self.cfg.models.hyperparameters.epochs)
        self.loss_record_validate = np.zeros(self.cfg.models.hyperparameters.epochs)

        # _________________________________________________________________________
        # Defining the torch Dataset and Dataloader
        self.train_data = Data(self.cfg, "train")
        self.train_data_loader = DataLoader(
            self.train_data, batch_size=1, shuffle=False
        )
        self.validate_data = Data(self.cfg, "validate")
        self.validate_data_loader = DataLoader(
            self.validate_data, batch_size=1, shuffle=False
        )

        # _________________________________________________________________________
        # Prepare to evalute model every batch_train_days
        self.timesteps_per_train_batch = (
            self.cfg.data.timesteps_per_day
            * self.cfg.models.hyperparameters.batch_train_days
        )

        quotient, self.last_batch_remainder = divmod(
            self.train_data.n_timesteps, self.timesteps_per_train_batch
        )
        self.n_trainbatches = quotient + 1

        # ___________________________________________________________________________
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
            step_size=(cfg.models.hyperparameters.step_size) * self.n_trainbatches,
            gamma=cfg.models.hyperparameters.gamma,
        )

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

        # _________________________________________________________________
        # Initialization

        self.model.train()  # this .train() is a function from nn.Module
        # dist.init_process_group(
        #     backend="gloo",
        #     world_size=0,
        #     rank=self.cfg.num_processes,
        # )

        # Create the DDP object with the GLOO backend
        # self.net = DDP(self.model.to(self.cfg.device), device_ids=None)

        log.info("Initializing the model")
        self.model.initialize()
        self.run_model(
            period="initialize", run_mlp=False
        )  # Run process model once to get the internal states

        # _________________________________________________________________
        # Training through epoch
        for epoch in range(1, self.cfg.models.hyperparameters.epochs + 1):
            # TODO: Loop through basins
            self.initialize_record()
            self.loss_record[epoch - 1] = self.train_one_epoch()
            self.save_weights_and_optimizer(epoch)
            if epoch % self.cfg.models.hyperparameters.validate_every == 0:
                self.loss_record_validate[epoch - 1], _ = self.validate()
            self.current_epoch += 1

    def initialize_record(self):
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
        self.x_check = np.empty(
            [self.train_data.num_basins, self.train_data.n_timesteps, 2]
        )

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        # _________________________________________________________________________
        # Initialization
        self.optimizer.zero_grad()
        # Reset the model states and parameters, and gradients
        self.model.initialize()

        y_hat = torch.empty([self.train_data.num_basins, self.train_data.n_timesteps])

        # _________________________________________________________________________
        # Evaluate model every batch_train_days
        for train_batch in range(1, self.n_trainbatches + 1):
            # _________________________________
            # Initialization for the next batch
            self.optimizer.zero_grad()
            # Deatch the model states and parameters, and gradients (but not resetting the values as model run continues)
            self.model.detach_gradients()

            # _________________________________
            # Get start and end index
            start_idx = (train_batch - 1) * self.timesteps_per_train_batch
            if train_batch == self.n_trainbatches:
                end_idx = (
                    train_batch - 1
                ) * self.timesteps_per_train_batch + self.last_batch_remainder
            else:
                end_idx = train_batch * self.timesteps_per_train_batch

            # _________________________________
            # Run model
            log.info(
                f"Epoch #: {self.current_epoch+1}/{self.cfg.models.hyperparameters.epochs} --- Batch # {train_batch}/{self.n_trainbatches}"
            )
            _y_hat = self.run_one_batch(start_idx, end_idx)

            # _________________________________
            # Calculate the loss
            if train_batch == 1:
                consider_warmup = True
            else:
                consider_warmup = False

            y_hat[:, start_idx:end_idx] = _y_hat

            log.info(
                f"At epoch {self.current_epoch+1}/{self.cfg.models.hyperparameters.epochs} --- batch {train_batch}/{self.n_trainbatches} (train)"
            )
            loss = self.evaluate(
                _y_hat,
                self.train_data.y[:, start_idx:end_idx, :],
                "train",
                consider_warmup,
            )

            # _________________________________
            # Loss backward and optimizer
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

        # Calculate loss for the entire record
        log.info(
            f"At epoch {self.current_epoch+1}/{self.cfg.models.hyperparameters.epochs} --- throughout the batches (train)"
        )
        loss = self.evaluate(y_hat, self.train_data.y, "train", True)

        self.save_result(
            y_hat=y_hat.detach().numpy(),
            y_t=self.train_data.y.detach().numpy(),
            Cgw_record=self.Cgw_train,
            satdk_record=self.satdk_train,
            out_filename=f"epoch{self.current_epoch+1}",
            period="train",
            plot_figure=False,
        )

        return loss

    def run_one_batch(self, start_idx, end_idx):
        """Run process model with MLP from start_idx to end_idx"""
        # _________________________________
        # initialize
        y_hat = torch.empty(
            [self.train_data.num_basins, (end_idx - start_idx)],
            device=self.cfg.device,
        )
        y_hat.fill_(float("nan"))

        # __________________________________
        # Get limited dataset
        batch_data = BatchData(self.train_data, start_idx, end_idx)
        batch_data_loader = DataLoader(
            batch_data, batch_size=1, shuffle=False
        )  # Specify the batch size and other parameters

        # _________________________________
        # Run model
        for t, x in enumerate(
            tqdm(batch_data_loader, desc=f"test({start_idx}:{end_idx})")
        ):
            absolute_timestep = start_idx + t
            if absolute_timestep >= end_idx:
                break

            # _________________________________
            # Run MLP forward
            self.model.mlp_forward(absolute_timestep, "train")
            self.Cgw_train[:, absolute_timestep] = self.model.Cgw.detach().numpy()
            self.satdk_train[:, absolute_timestep] = self.model.satdk.detach().numpy()
            self.x_check[:, absolute_timestep] = x

            # _________________________________
            # Run process model
            runoff = self.model(x)
            y_hat[:, t] = runoff

        return y_hat

    def run_model(self, period="initialize", run_mlp=False):
        """Run process model with or without MLP"""
        if (period == "initialize") or (period == "train"):
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
        for t, (x, _) in enumerate(tqdm(dataloader, desc=period)):
            if run_mlp:
                self.model.mlp_forward(t, "validate")
            runoff = self.model(x)
            y_hat[:, t] = runoff.detach()

        return y_hat

    def evaluate(
        self,
        y_hat_: Tensor,
        y_t_: Tensor,
        period,
        consider_warmup=True,
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

        if consider_warmup:
            warmup = self.cfg.models.hyperparameters.warmup
        else:
            warmup = 0

        y_hat = y_hat_[:, warmup:]
        y_t = y_t_[:, warmup:]

        y_hat_np = y_hat_.detach().numpy()
        y_t_np = y_t_.detach().numpy()

        # Save results
        # Evaluate
        kge = he.evaluator(he.kge, y_hat_np[0], y_t_np[0])

        # Compute the overall loss
        mask = torch.isnan(y_t)
        y_t_dropped = y_t[~mask]
        y_hat_dropped = y_hat[~mask]
        if y_hat_dropped.shape != y_t_dropped.shape:
            print("y_t and y_hat shape not matching")

        # TODO: try different loss for the validation
        loss = self.criterion(y_hat_dropped, y_t_dropped)

        log.info(f"Loss: {loss:.8f}")
        log.info(
            f"KGE for the basin {self.train_data.basin_ids[0]}: {float(kge[0]):.8}"
        )

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
            y_hat = self.run_model(period="validate", run_mlp=True)
        log.info(
            f"At epoch {self.current_epoch+1}/{self.cfg.models.hyperparameters.epochs} (validate)"
        )
        loss = self.evaluate(
            y_hat,
            self.validate_data.y,
            "validate",
            True,
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
            y_hat_ = self.run_model(period="train", run_mlp=True)
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
            period=period,
            out_filename=f"final_result",
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
            else:
                log.debug(
                    "invalid parameter for period -- choose from 'train' or 'validate'"
                )
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
