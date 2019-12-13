from itertools import permutations as permme
import numpy as np
from basenji import seqnn
import json

TRACTABLE_LIMIT = 10

MODEL_SEQUENCE_LENGTH = 1000
SEQUENCE_LENGTH = 10
TRAIN_DATA_LEN = 300
TEST_DATA_LEN = 100

CELLS = 4

NEG_FACTOR = -1/3

A = "A"
G = "G"
C = "C"
T = "T"

CHAR_2_INDEX = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}


def load_basenji():
    """
    Loads the Basenji model from the file for its architecture and the file for its weights
    :return: the loaded Basenji
    """
    with open('cs273b_params.json') as params_open:
        params = json.load(params_open)

        params_model = params['model']
        _ = params['train']  # params_train

        model = seqnn.SeqNN(params_model)

        model.model.load_weights('cs273b_model.h5')

        return model


def get_activations_for_sequence(model, sequences_x):
    """

    :param model:
    :param sequences_x:
    :return:
    """
    # zero pad
    remainder = np.array([np.array([0, 0, 0, 0]) for _ in range(MODEL_SEQUENCE_LENGTH - SEQUENCE_LENGTH)])
    sequences_x = list(sequences_x)

    for g in range(len(sequences_x)):
        print(g)
        new_seq = np.zeros((MODEL_SEQUENCE_LENGTH, 4))
        new_seq[0:SEQUENCE_LENGTH] = sequences_x[g]
        new_seq[SEQUENCE_LENGTH:] = remainder
        sequences_x[g] = new_seq
    sequences_x = np.array(sequences_x)

    print(sequences_x.shape)
    pred_y = model.predict(sequences_x)
    print(pred_y)
    return pred_y


def get_cell_type_desired_outcome(cell_type):
    """
    Example: The desired outcome for cell type 1 (index 0) is [True, False, False, False]
    :param cell_type: the cell type for which to get the desired outcome boolean array
    :return: the desired outcome boolean array
    """
    return [bool(cell_type == 1), bool(cell_type == 2), bool(cell_type == 3), bool(cell_type == 4)]


def get_cell_type_undesired_outcome(cell_type):
    """
    Example: The undesired outcome for cell type 1 (index 0) is [False, True, True, True]
    :param cell_type: the cell type for which to get the undesired outcome boolean array
    :return: the undesired outcome boolean array
    """
    return [not bool(cell_type == 1), not bool(cell_type == 2), not bool(cell_type == 3), not bool(cell_type == 4)]


def get_err_vec(seq_predictions, desired, undesired):
    """
    Calculates the error vector for the given predictions.
    Implicitly contains the objective function that we are maximizing.
    This does NOT use the maximization (desired cell - max(of other cells' activations)) objective function
    described in the genetic algorithm.

    :param seq_predictions: Predictions from which we calculate the error
    :param desired: an array of booleans, with one Truth value, indicating the cell to maximize
    :param undesired: an array of booleans, with one False value, indicated the cell to maximize
    :return: The error for the given predictions
    """
    pos_mask = seq_predictions * desired
    neg_mask = seq_predictions * undesired * NEG_FACTOR
    curr_err = np.sum(pos_mask + neg_mask, axis=-1)
    return curr_err.reshape((curr_err.shape[0], 1, 1))


def get_best_acts_for_cell(seqs, activations, cell, seqs_to_get=5):
    """
    Gets the best seqs_to_get sequences from the seqs passed, with the given activations, for the given cell

    :param seqs: the sequences to run
    :param activations: the activations calculated for the sequences
    :param cell: the cell being maximized
    :param seqs_to_get: the sequences to get
    :return: the best sequences by straight magnitude for the desired cell, the best sequences by our objective
    """
    stale_seq = np.array([np.array([0, 0, 0, 0]) for _ in range(SEQUENCE_LENGTH)])
    max_five_mag_seqs = np.array([stale_seq for _ in range(seqs_to_get)])
    max_five_objcs_seqs = np.array([stale_seq for _ in range(seqs_to_get)])

    desired = get_cell_type_desired_outcome(cell_type=cell + 1)
    undesired = get_cell_type_undesired_outcome(cell_type=cell + 1)

    objcs = get_err_vec(activations, desired, undesired).ravel()

    lower_bound_mag = -1000.0
    lower_bound_objc = -1000.0

    max_five_magnitudes = np.array([lower_bound_mag for _ in range(seqs_to_get)])
    max_five_objectives = np.array([lower_bound_objc for _ in range(seqs_to_get)])

    seq_number = 0
    for seq in seqs:
        mag = activations[seq_number][0][cell]
        objc = objcs[seq_number]

        curr_min_mag = np.amin(max_five_magnitudes)
        curr_min_obj = np.amin(max_five_objectives)

        curr_min_mag_arg = int(np.argmin(max_five_magnitudes))
        curr_min_obj_arg = int(np.argmin(max_five_objectives))

        if mag > curr_min_mag:
            max_five_mag_seqs[curr_min_mag_arg] = seq
            max_five_magnitudes[curr_min_mag_arg] = mag
        if objc > curr_min_obj:
            max_five_objcs_seqs[curr_min_obj_arg] = seq
            max_five_objectives[curr_min_obj_arg] = objc

        seq_number += 1
    return max_five_mag_seqs, max_five_objcs_seqs


def get_best_seqs(seqs, activations_calculated):
    """
    Gets the best seqs_to_get sequences from the seqs passed, with the given activations

    :param seqs: the sequences to run
    :param activations_calculated: the activations calculated for the sequences
    :return: the best sequences, where the amount of sequences is given by seqs_to_get * CELLS
    """
    seqs_to_get = 5

    total_best_seqs = np.zeros((CELLS, 2, seqs_to_get, SEQUENCE_LENGTH, 4))

    for r in range(CELLS):
        print(r)
        best_acts = get_best_acts_for_cell(seqs, activations_calculated, r, seqs_to_get=seqs_to_get)
        total_best_seqs[r] = best_acts

    return total_best_seqs


def exhaustive():
    """
    :return: all sequence permutations of length SEQUENCE_LENGTH
    """
    exhaustive_seqs = []

    def get_perms_for_counts(a_count, c_count, g_count, t_count):
        selected_seq = []
        for _ in range(a_count):
            selected_seq.append(A)
        for _ in range(c_count):
            selected_seq.append(C)
        for _ in range(g_count):
            selected_seq.append(G)
        for _ in range(t_count):
            selected_seq.append(T)
        return permme(np.array(selected_seq))

    # Cycle through all possible combinations counts for each base
    for a in range(0, SEQUENCE_LENGTH + 1):
        max_c = SEQUENCE_LENGTH - a
        for c in range(0, max_c + 1):
            max_g = SEQUENCE_LENGTH - a - c
            for g in range(0, max_g + 1):
                t = SEQUENCE_LENGTH - a - c - g
                # Get the permutations
                perms = get_perms_for_counts(a, c, g, t)
                perms = list(set(perms))
                for perm in perms:
                    perm_str = ""
                    for p in range(len(perm)):
                        perm_str += perm[p]
                    exhaustive_seqs.append(perm_str)

    return list(set(exhaustive_seqs))


def get_exhaustive():
    """
    :return: nothing, but saves all permutations of sequences of length SEQUENCE_LENGTH to a file
    """
    exhaustive_seqs = exhaustive()
    exhaustive_seqs_np = []
    for seq in exhaustive_seqs:
        converted_seq = []
        for letter in seq:
            converted_seq.append(CHAR_2_INDEX[letter])
        exhaustive_seqs_np.append(np.array(converted_seq))
    exhaustive_seqs_np = np.array(exhaustive_seqs_np)
    exhaustive_seqs_np = exhaustive_seqs_np.reshape((len(exhaustive_seqs), SEQUENCE_LENGTH, 4))
    np.save("exhaustiveSequences.npy", exhaustive_seqs_np)


def get_temp_five_file_name(for_num):
    """
    Returns the name for a temporary file to store the best sequences for the interval used.

    :param for_num: the number with which to name the file
    :return: the temporary file name
    """
    return "exhaustive/bestSeqs" + str(for_num) + ".npy"


def get_best_some_seqs(model, total_seqs, begin_index, end_index):
    """
    Gets the best sequences for a slice of the total sequences, between the begin_index and the end_index

    :param model: the Basenji model passed
    :param total_seqs: all of the sequences (all of the permutations)
    :param begin_index: the beginning index of the slice
    :param end_index: the ending index of the slice
    :return: nothing, but saves the best sequences to a file
    """
    some_seqs = total_seqs[begin_index:end_index][:][:]
    get_best(model, some_seqs, end_index)


def get_best(model, seqs, file_index):
    """
    Gets the best sequences for the sequences passed

    :param model: the Basenji model passed
    :param seqs: all of the sequences (all of the permutations)
    :param file_index: the file index with which to save the best sequences
    :return: nothing, but saves the best sequences to a file
    """
    activations = get_activations_for_sequence(model, seqs)
    best_seqs = get_best_seqs(seqs, activations)
    np.save(get_temp_five_file_name(file_index), best_seqs)


if __name__ == "__main__":
    # Previously, we permuted all sequences of length SEQUENCE_LENGTH and saved them to a file with get_exhaustive()
    file_name_to_load = "exhaustive/exhaustiveSequencesTotal.npy"
    total_sequences = np.load(get_temp_five_file_name(0), allow_pickle=True)
    print(total_sequences.shape)

    # Load the Basenji model
    basenji_model = load_basenji()

    # The RAM of many VMs and computers are overloaded if you try to input all 1e6 sequences at once.
    # Here we split it up into intervals of 5000, get the best 20, and then get the best of those
    interval = 5000
    seq_num = total_sequences.shape[0]
    range_to_use = int(seq_num / interval)

    for i in range(range_to_use):
        # A progress indicator
        print(str(i * interval / seq_num) + "%")
        ending_index = min(interval + i * interval, seq_num)
        get_best_some_seqs(basenji_model, total_sequences, i, ending_index)

    # Get final splits and get the best results of those.
    axis = 5
    best_five_seqs = np.zeros((5, 10, 4))

    for i in range(range_to_use):
        ending_index = min(interval + i * interval, seq_num)
        if ending_index == 1030000:
            break
        loaded_seqs = np.load(get_temp_five_file_name(ending_index), allow_pickle=True)

        for j in range(CELLS):
            for k in range(2):
                best_five_seqs = np.concatenate((best_five_seqs, loaded_seqs[j, k, :, :, :]))

    print("==============FINAL SEQUENCES============")
    print(best_five_seqs.shape)
    print(best_five_seqs)
    get_best(basenji_model, best_five_seqs, 0)
