from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_loss_function(log, saving_path):

    df_training = pd.read_csv(log, sep=',', engine='python')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(df_training["epoch"], df_training["loss"], label='Training Loss')
    ax.plot(df_training["epoch"], df_training["val_loss"], color='tab:orange', label='Validation Loss')
    ax.set_title("Loss / Validation Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, which='both')
    plt.savefig(os.path.join(saving_path, "Loss_vs_Epoch.png"))
    return 0

def plot_original_vs_reconstructed(self, reco_data, train_data, save_path):
    epochs = ""
    for c in self.model_path:
        if c.isdigit():
            epochs = epochs + c
    title = 'Original vs Reconstructed Signal ' + epochs + " Epochs" 
    save_path = "AE_test_model/Original_VS_Reconstructed_" + epochs + "_Epochs_" + str(WINDOW_HOPTIME) + "_Seocnds" + ".jpg"
    fig, ax = plt.subplots(2, sharex='col', sharey='row' ,figsize=(10,6))
    plt.suptitle(title)
    ax[0].plot(np.arange(0, reco_data.shape[0]), np.concatenate(train_data), label='original')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.arange(0, reco_data.shape[0]), reco_data, label='reconstructed', color='tab:orange')
    ax[1].legend()
    ax[1].grid(True)
    plt.savefig(save_path)
    plt.show()
    return 0

if __name__ == '__main__':

    training_log = "training_LTSM.log"
    model_path = "LTSM_1024_25.model"
    directory = "Figures/LTSM_model_1"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    plot_loss_function(training_log, directory)
        