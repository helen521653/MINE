import click
import os
import logging

import numpy as np
import torch
#====================================================
import numpy as np

from pytorch_lightning import Trainer, seed_everything

from utils import (
    MultivariateNormalDataset,
    plot_mi_results,
    load_config,
)
from mine import MutualInformationEstimator
#====================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




@click.command()
@click.option("--config_path", default="mine_mi_estimator/train_config.yaml")


def main(config_path: str):
    print(config_path)

    config = load_config(config_path)
    print(config)

    # -- Parameters from config ---
    N = config.data_generation.get('num_samples', 20000)
    dim = config.data_generation.get('dim', 20)
    rhos_start = config.data_generation.get('rhos_start', -0.9)
    rhos_end = config.data_generation.get('rhos_end', 0.9)
    rhos_num = config.data_generation.get('rhos_num', 5)

    model_hidden_dim = config.model.get('hidden_dim', 128)
    loss_type_str = config.model.get('loss_type', 'mine')
    # Преобразуем строку в список (поддерживаем оба формата: строку и список)
    if isinstance(loss_type_str, str):
        loss_type = [lt.strip() for lt in loss_type_str.split(',')]
    else:  # если уже список
        loss_type = loss_type_str

    epochs = config.training.get('n_epochs', 50)
    batch_size = config.training.get('batch_size', 64)
    lr = config.training.get('learning_rate', 1e-4)
    ema_decay = config.training.get('ema_decay', 0.01)
    biased_loss = config.training.get('biased_loss', False)
    logging_steps = config.training.get('logging_steps', 10)

    plot_results = config.evaluation.get('plot_results', True)

    seed_everything(42, workers=True)


    #seed = config.get('seed', 8888)
    #set_seed(seed)

    # --- Setting device ---
    device_str = config.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Training MINE for different rhos ---
    # rhos = np.linspace(rhos_start, rhos_end, rhos_num)
    #rhos = [-0.99,0.99]

    rhos = [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

    logging.info("Starting MINE training for various correlation coefficients...")
    #===================================================================================
    results_dict = dict()

    for loss in loss_type:
        results = []
        for rho in rhos:
            train_loader = torch.utils.data.DataLoader(
            MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
            MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)

            
            true_mi = train_loader.dataset.true_mi
            print(true_mi)

            kwargs = {
                'lr': lr,
                'batch_size': batch_size,
                'train_loader': train_loader,
                'test_loader': test_loader,
                'alpha': ema_decay,
                'current_rho': rho,      
                'true_mi': true_mi      
            }

            model = MutualInformationEstimator(
                dim, dim, loss=loss, **kwargs).to(device)

            
            trainer = Trainer(
                max_epochs=epochs,
                log_every_n_steps=1,
                enable_progress_bar=True,
                accelerator='auto',  
                devices='auto',
            )


            trainer.fit(model)
            trainer.test(model)
            

            print("True_mi {}".format(true_mi))
            print("MINE {}".format(model.avg_test))
            results.append((rho, model.avg_test, true_mi))

        results = np.array(results)
        results_dict[loss] = results

    logging.info("MINE training completed.")
    


    if plot_results:
        plot_mi_results(results_dict)
    else:
        logging.info("Plotting results skipped.")

    logging.info("Training finished.")


if __name__ == "__main__":
    main()
