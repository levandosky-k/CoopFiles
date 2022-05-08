""" Contains the train module that governs training Deepymod """
import torch
from ..utils.logger import Logger
from .convergence import Convergence
from ..model.deepmod import DeepMoD
from torch.utils.data import DataLoader

import numpy as np #my addition
from matplotlib import pyplot #my addition


def train(
    model: DeepMoD,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer,
    sparsity_scheduler,
    split: float = 0.8,
    exp_ID: str = None,
    log_dir: str = None,
    max_iterations: int = 10000,
    write_iterations: int = 25,
    **convergence_kwargs
) -> None:
    """Trains the DeepMoD model. This function automatically splits the data set in a train and test set.

    Args:
        model (DeepMoD):  A DeepMoD object.
        data (torch.Tensor):  Tensor of shape (n_samples x (n_spatial + 1)) containing the coordinates, first column should be the time coordinate.
        target (torch.Tensor): Tensor of shape (n_samples x n_features) containing the target data.
        optimizer ([type]):  Pytorch optimizer.
        sparsity_scheduler ([type]):  Decides when to update the sparsity mask.
        split (float, optional):  Fraction of the train set, by default 0.8.
        exp_ID (str, optional): Unique ID to identify tensorboard file. Not used if log_dir is given, see pytorch documentation.
        log_dir (str, optional): Directory where tensorboard file is written, by default None.
        max_iterations (int, optional): [description]. Max number of epochs , by default 10000.
        write_iterations (int, optional): [description]. Sets how often data is written to tensorboard and checks train loss , by default 25.
    """

    ####### my additions #######
    #f = open("loss_data_file_mse.txt", "w")
    #f.write("")
    #f.close()
    #f = open("loss_data_file_iters.txt", "w")
    #f.write("")
    #f.close()
    iter_list = np.array([])
    #mse_list = np.array([])
    coeffs_list = np.array(np.array([]))
    print("here1")
    ############################

    
    logger = Logger(exp_ID, log_dir)
    sparsity_scheduler.path = (
        logger.log_dir
    )  # write checkpoint to same folder as tb output.
    n_features = train_dataloader[0][1].shape[-1]
    # n_features = model.func_approx.modules()
    # Training
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # Training variables defined as: loss, mse, regularisation
        batch_losses = torch.zeros(
            (3, n_features, len(train_dataloader)),
            device=train_dataloader.device,
        )
        for batch_idx, train_sample in enumerate(train_dataloader):
            data_train, target_train = train_sample
            # ================== Training Model ============================
            prediction, time_derivs, thetas = model(data_train)
            batch_losses[1, :, batch_idx] = torch.mean(
                (prediction - target_train) ** 2, dim=-2
            )  # loss per output
            batch_losses[2, :, batch_idx] = torch.stack(
                [
                    torch.mean((dt - theta @ coeff_vector) ** 2)
                    for dt, theta, coeff_vector in zip(
                        time_derivs,
                        thetas,
                        model.constraint_coeffs(scaled=False, sparse=True), #scaled=False, sparse=True
                    )
                ]
            )
            batch_losses[0, :, batch_idx] = (
                batch_losses[1, :, batch_idx] + batch_losses[2, :, batch_idx]
            )

            # Optimizer step
            optimizer.zero_grad()
            batch_losses[0, :, batch_idx].sum().backward()
            optimizer.step()

        loss, mse, reg = torch.mean(batch_losses.cpu().detach(), axis=-1)
        #print(loss)
        #print(mse+reg)

        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                batch_mse_test = torch.zeros(
                    (n_features, len(test_dataloader)), device=test_dataloader.device
                )
                for batch_idx, test_sample in enumerate(test_dataloader):
                    data_test, target_test = test_sample
                    prediction_test = model.func_approx(data_test)[0]
                    batch_mse_test[:, batch_idx] = torch.mean(
                        (prediction_test - target_test) ** 2, dim=-2
                    )  # loss per output
            mse_test = batch_mse_test.cpu().detach().mean(dim=-1)
            # ====================== Logging =======================
            _ = model.sparse_estimator(
                thetas, time_derivs
            )  # calculating estimator coeffs but not setting mask
            logger(
                iteration,
                loss.view(-1).mean(),
                mse.view(-1),
                reg.view(-1),
                model.constraint_coeffs(sparse=True, scaled=True), #sparse=True, scaled=True
                model.constraint_coeffs(sparse=True, scaled=False), #sparse=True, scaled=False
                model.estimator_coeffs(),
                MSE_test=mse_test,
            )
            print(model.constraint_coeffs(sparse=True, scaled=False))

            ######### my additions ########
            #f = open("loss_data_file_iters.txt", "a")
            #f.write("%i\n" %iteration)
            #f.close()
            #f = open("loss_data_file_mse.txt", "a")
            #f.write("%f\n" %mse_test)
            #f.close()
            iter_list = np.append(iter_list, iteration)
            #mse_list = np.append(mse_list, mse_test)
            #print(model.estimator_coeffs()[0][0])
            if len(coeffs_list) == 0:
                coeffs_list = [[]] * len(model.estimator_coeffs()[0])
            i=0
            while i < len(model.estimator_coeffs()[0]):
                #print(len(coeffs_list))
                coeffs_list[i] = np.append(coeffs_list[i], model.estimator_coeffs()[0][i])
                i=i+1
            #print(model.estimator_coeffs())
            ###############################

            # ================== Sparsity update =============
            # Updating sparsity
            update_sparsity = sparsity_scheduler(
                iteration, torch.sum(mse_test), model, optimizer
            )
            if update_sparsity:
                model.constraint.sparsity_masks = model.sparse_estimator(
                    thetas, time_derivs
                )

            # ================= Checking convergence
            l1_norm = torch.sum(
                torch.abs(
                    torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1) #sparse=True, scaled=True
                )
            )
            #print(l1_norm)
            #print(model.constraint_coeffs(scaled=False, sparse=True))
            converged = convergence(iteration, l1_norm)
            #if iteration >= max_iterations:
            #    break
            if converged:
                break
    logger.close(model)


    #my additions####
    #print(iter_list)
    #print(mse_list)
    #pyplot.plot(iter_list, mse_list)
    #pyplot.show()
    #i=0
    #while i < len(model.estimator_coeffs()[0]):
    #            pyplot.plot(iter_list, coeffs_list[i])
    #            i=i+1
    #pyplot.savefig("coeffgraph.png")
    #pyplot.show()

    #print(iter_list)
    #print(coeffs_list)

    np.savetxt("iter_list.txt", iter_list)
    np.savetxt("coeffs_list.txt", coeffs_list)
    #################




    
