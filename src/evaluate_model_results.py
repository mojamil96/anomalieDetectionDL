from preprocess_data import *
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import vallenae as vae
import os
import yaml


def plot_loss_function(training_log, model_path, model_type):

    name = model_path.split('/')[-1].split('.')[0]
    df_training = pd.read_csv(training_log, sep=',', engine='c')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(df_training["epoch"], df_training["loss"], label='Training Loss')
    ax.plot(df_training["epoch"], df_training["val_loss"], color='tab:orange', label='Validation Loss')
    ax.set_title("Loss / Validation Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, which='both')
    
    if model_type == 'LSTM':
        if not os.path.isdir("Figures/LSTM/"):
            os.mkdir("figures/LSTM/")
        fig.savefig(os.path.join("figures/LSTM", name))

    elif model_type == 'normal':
        if not os.path.isdir("Figures/VariationalAE/Model_1"):
            os.mkdir("figures/VariationalAE/Model_1")
        fig.savefig(os.path.join("figures/VariationalAE", name))

def define_threshold(training_mae, quantile):
    training_mae_sorted = np.sort(training_mae, axis=0)
    threshold = np.quantile(training_mae_sorted, quantile)
    return threshold

def evaluate_results_mixture_of_experts(train_mae, test_mae, sup_mae, quantile):
    threshold_ts = define_threshold(train_mae[0], quantile)
    threshold_ft = define_threshold(train_mae[1], quantile)
    if np.array_equal(train_mae[0], test_mae[0]):
        print('Train and Test reconstruction errors are equal')
    if np.array_equal(test_mae[0], sup_mae[0]):
        print('Test and Sup errors are equal')

    false_pos = []
    true_pos = []
    test_vs_train_ts = test_mae[0] > threshold_ts # B* vs B, anomalies detected here are classified as false
    test_vs_train_ft = test_mae[1] > threshold_ft

    sup_vs_test_ts = sup_mae[0] > threshold_ts # B* vs B+wb, anomalies detected here are classiefed as true 
    sup_vs_test_ft = sup_mae[1] > threshold_ft

    if np.array_equal(test_vs_train_ts, sup_vs_test_ts):
        print('test and sup checks are equal')
    # Check if a sample is True in both timeseries and dft
    for idx in range(len(test_vs_train_ts)):
        if test_vs_train_ts[idx] and test_vs_train_ft[idx]:
            false_pos.append(test_vs_train_ts[idx])
    for idx in range(len(sup_vs_test_ts)):
        if sup_vs_test_ts[idx] and sup_vs_test_ft[idx]:
            true_pos.append(sup_vs_test_ts[idx])

    false_positives = len(false_pos)
    true_posistives = len(true_pos)
    true_negatives = len(test_vs_train_ts) - false_positives

    #Calculate sensitivity, precision, accuracy and F1 scores
    P = len(train_mae[0])
    N = len(train_mae[0])
    sensitivity = true_posistives / P
    precision = true_posistives / (true_posistives + false_positives)
    accuracy = (true_posistives + true_negatives) / (P + N)
    F1 = (2 * precision * sensitivity) / (precision + sensitivity)

    return threshold_ts, threshold_ft, true_posistives, false_positives, accuracy, F1

def evaluate_model_LSTM(train_mae, test_mae, sup_mae, quantile):
    training_mae_sorted = np.sort(train_mae, axis=0)
    threshold_ts = np.quantile(training_mae_sorted, quantile)
    false_pos = []
    true_pos = []
    test_vs_train_ts = test_mae > threshold_ts
    sup_vs_test_ts = sup_mae > threshold_ts
    for idx in range(len(test_vs_train_ts)):
        if test_vs_train_ts[idx]:
            false_pos.append(test_vs_train_ts[idx])
    for idx in range(len(sup_vs_test_ts)):
        if sup_vs_test_ts[idx]:
            true_pos.append(sup_vs_test_ts[idx])
    false_positives = len(false_pos)
    true_posistives = len(true_pos)
    true_negatives = len(test_vs_train_ts) - false_positives

    #Calculate sensitivity, precision, accuracy and F1 scores
    P = len(train_mae)
    N = len(train_mae)
    sensitivity = true_posistives / P
    precision = true_posistives / (true_posistives + false_positives)
    accuracy = (true_posistives + true_negatives) / (P + N)
    F1 = (2 * precision * sensitivity) / (precision + sensitivity)

    return threshold_ts, true_posistives, false_positives, accuracy, F1

def evaluate_model(train_mae, test_mae, sup_mae, quantile):
    training_mae_sorted = np.sort(train_mae, axis=0)
    threshold_ts = np.quantile(training_mae_sorted, quantile)
    false_pos = []
    true_pos = []
    test_vs_train_ts = test_mae > threshold_ts
    sup_vs_test_ts = sup_mae > threshold_ts
    for idx in range(len(test_vs_train_ts)):
        if test_vs_train_ts[idx]:
            false_pos.append(test_vs_train_ts[idx])
    for idx in range(len(sup_vs_test_ts)):
        if sup_vs_test_ts[idx]:
            true_pos.append(sup_vs_test_ts[idx])
    false_positives = len(false_pos)
    true_posistives = len(true_pos)
    true_negatives = len(test_vs_train_ts) - false_positives

    #Calculate sensitivity, precision, accuracy and F1 scores
    P = len(train_mae)
    N = len(train_mae)
    sensitivity = true_posistives / P
    precision = true_posistives / (true_posistives + false_positives)
    accuracy = (true_posistives + true_negatives) / (P + N)
    F1 = (2 * precision * sensitivity) / (precision + sensitivity)

    return threshold_ts, true_posistives, false_positives, accuracy, F1

def determine_simpleThreshold(signal, threshold_factor=1.0):
    mean_value = np.mean(signal)
    std_dev = np.std(signal)

    threshold = mean_value + threshold_factor * std_dev

    return threshold

def evaluate_model_simpleThreshold(data_processor, signal_processor, noise_path, wirebreak_path):
    data_type = None

    false_pos   = 0
    true_pos    = 0

    wirebreak                            = signal_processor.get_wirebreak(wirebreak_path, 400)
    valid, noise                         = signal_processor.get_noise_signal(noise_path)
    noise                                = noise[:, 0:170000000]
    noise_seq, target_data               = data_processor.create_sequences(noise)

    if data_type == 'dft':
        wirebreak           = data_processor.to_fft(wirebreak)
        noise_seq           = data_processor.to_fft(noise_seq)

    superimposed_data = signal_processor.superposition_noise_wirebreak(wirebreak, noise_seq, samples_in_seq=1024)

    # threshold_onlyNoise_data = determine_simpleThreshold(noise)
    # threshold_supimposed_data = determine_simpleThreshold(superimposed_data)
    clean_signals = []
    burst_signals = []
    for seq in range(len(noise_seq)):
        threshold_onlyNoise_data      = determine_simpleThreshold(noise_seq[seq])
        threshold_supimposed_data     = determine_simpleThreshold(superimposed_data[seq])
        
        if np.any(noise_seq[seq] > threshold_onlyNoise_data):
            clean_signals.append(noise_seq[seq])
            false_pos += 1


        if np.any(superimposed_data[seq] > threshold_supimposed_data):
            burst_signals.append(superimposed_data[seq])
            true_pos += 1
    
    #Calculate sensitivity, precision, accuracy and F1 scores
    P = len(noise_seq)
    N = len(noise_seq)
    sensitivity = true_pos / P
    precision = true_pos / (true_pos + false_pos)
    F1 = (2 * precision * sensitivity) / (precision + sensitivity)

    return F1, clean_signals, burst_signals

def evaluate_auc_AE(train_mae, test_mae, sup_mae):

    threshold_ft = define_threshold(train_mae, 0.9)

    ###############################################################################################
    # Berechene AUC-Wert
    ###############################################################################################
    clean_signals = test_mae[test_mae > threshold_ft]
    burst_signals = sup_mae[sup_mae > threshold_ft]

    np.savetxt('clean_signals.csv', clean_signals, delimiter=',')
    np.savetxt('burst_signals.csv', burst_signals, delimiter=',')

    # y_true: 0 für saubere Signale, 1 für Burst-Signale
    y_true = np.concatenate([np.zeros(len(clean_signals)), np.ones(len(burst_signals))])

    # y_score: Test-MAE-Werte, wobei Werte über dem Schwellenwert als Klasse 1 betrachtet werden
    y_score = np.concatenate([clean_signals, burst_signals])

    # Berechne ROC-Kurve und AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plot_roc_curve(fpr, tpr, roc_auc)

    return 0

def evaluate_auc_simpleThreshold(clean_signals, burst_signals):
   
    clean_signals = np.array(clean_signals)
    burst_signals = np.array(burst_signals)

    # Labels für saubere Sequenzen (0 für sauber)
    y_true_clean = np.zeros(len(clean_signals.flatten()))

    # Labels für beschädigte Sequenzen (1 für beschädigt)
    y_true_augmented = np.ones(len(burst_signals.flatten()))
    
    # Kombiniere Labels und Daten
    y_true = np.concatenate([y_true_clean, y_true_augmented])
    y_score = np.concatenate([clean_signals.flatten(), burst_signals.flatten()])

    # Berechne ROC-Kurve und AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plot_roc_curve(fpr, tpr, roc_auc)

    return 0

def plot_roc_curve(fpr, tpr, roc_auc):

    # Plot ROC-Kurve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.grid(True, which="both")
    plt.show()

    return 0

