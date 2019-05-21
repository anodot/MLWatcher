"""
The metrics script implements functions that can calculate a large bunch of metrics dictionnaries based
 on the received predict_proba matrix (mandatory) and optionally input_matrix and label_matrix.
"""

import numpy as np
from collections import Counter


def merge_dicts(*args):
    """
    Merge multiple dictionaries each with unique key
    :param args: dictionaries to merge
    :return: 1 merged dictionary
    """
    result = {k: v for d in args for k, v in d.items()}
    return result


def prediction_first_values_metrix(float_matrix):
    """
    Calculate metrics for the highest probability (prediction of classification)
    """
    first_prediction_proba = [max(x) for x in float_matrix]
    metrics_dict = calc_continuous_metrix(first_prediction_proba, 'pred_proba_first', 'max_prediction_proba', '', '')
    return metrics_dict


def prediction_second_values_metrix(float_matrix):
    """
    Calculate metrics for the spread between the first and second highest probability.
    """
    second_prediction_proba = [max(x) - max(n for n in x if n != max(x)) for x in float_matrix]
    metrics_dict = calc_continuous_metrix(second_prediction_proba, 'pred_proba_spread', 'spread12_prediction_proba', '', '')
    return metrics_dict


def prediction_matrix_metrix(prediction_matrix, threshold = 0.5):
    """
    Calculate classes distribution based on the prediction matrix values
    inputs : prediction_matrix floats of size (n inputs, m classes)
    returns : dict of freq of class i in batch : {i, freq_class_i}
    """

    buffer_size = prediction_matrix.shape[0]
    num_classes = prediction_matrix.shape[1]

    if num_classes == 2:
        # manage threshold for binary classification with threshold
        output_list = [1 if x[1] > threshold else 0 for x in prediction_matrix]
    else:
        output_list = [np.argmax(x) for x in prediction_matrix]

    m1_ = Counter({x: 0 for x in range(num_classes)})
    m1_.update(Counter(output_list).elements())

    count = [m1_[i] for i in m1_]
    class_count_names = ['pred_class_' + str(i) + '_count' for i in range(num_classes)]
    count_dict = {class_count_names[i]: [count[i], 'prediction_count', 'class_' + str(i), '', '', 'counter'] for i in
                  range(len(count))}
    count_dict['buffer_count'] = [buffer_size, 'buffer_count', '', '', '', 'counter']

    try:
        freq = [m1_[i] / buffer_size for i in m1_]
        class_frequencies_names = ['pred_class_'+str(i)+'_freq' for i in range(num_classes)]
        freq_dict = {class_frequencies_names[i]: [freq[i], 'prediction_freq', 'class_' + str(i), '', '', 'gauge'] for i in range(len(freq))}
    except ZeroDivisionError:
        freq_dict = {}

    return merge_dicts(freq_dict, count_dict)


def input_matrix_metrix(input_matrix):
    """
    Calculate features distribution based on the input matrix values.
    A feature is considered categorical if type == int, else numerical
    inputs : input_matrix floats of size (n lines, m columns)
    returns : dict of freq of class i in batch : {i, freq_class_i}
    """

    batch_size = input_matrix.shape[0]
    num_features = input_matrix.shape[1]

    input_dict = {}

    for feat_num in range(num_features):
        if isinstance(input_matrix[0, feat_num], int):
            feature_dict = calc_discrete_metrix(input_matrix[:, feat_num], 'feat_'+str(feat_num), 'feature', '', 'feat_'+str(feat_num))
        else:
            feature_dict = calc_continuous_metrix(input_matrix[:, feat_num], 'feat_' + str(feat_num), 'feature_value', '', 'feat_'+str(feat_num))
        input_dict = merge_dicts(input_dict, feature_dict)

    return input_dict


def calc_continuous_metrix(list_of_floats, id, what, class_name, feat_name):
    """
    Calculate a bunch of metrics for a list of floats
    return : dictionnary of metrics : {i, freq_class_i}
    """
    mean_m = np.mean(list_of_floats)
    median_m = np.median(list_of_floats)
    std_m = np.std(list_of_floats)

    range_m = max(list_of_floats) - min(list_of_floats)

    q75_m, q50_m, q25_m = np.percentile(list_of_floats, [75, 50, 25])
    iqr_m = q75_m - q25_m

    all_metrics = [mean_m, median_m, std_m, range_m, q25_m, q50_m, q75_m, iqr_m]
    all_metrics_names = ['mean', 'median', 'std', 'range', 'q25', 'q50', 'q75', 'iqr']
    all_metrics_names_ = [id + '_' + x for x in all_metrics_names]

    return {all_metrics_names_[i]: [all_metrics[i], what, class_name, feat_name, all_metrics_names[i], 'gauge']
            for i in range(len(all_metrics))}


def calc_discrete_metrix(list_of_ints, id, what, class_name, feat_name):
    """Calculate the distribution of a list of discrete ints
    Returns : a dictionnary of metrics"""
    batch_size = len(list_of_ints)
    m = Counter(list_of_ints)

    count = [m[i] for i in m]
    count_name = [id + '_value_' + str(i) + '_count' for i in m]
    count_dict = {count_name[i]: [count[i], what + '_count', class_name, feat_name, '', 'counter'] for i in
                  range(len(count))}

    try:
        freq = [m[i] / batch_size for i in m]
        freq_name = [id + '_value_' + str(i) + '_freq' for i in m]
        freq_dict = {freq_name[i]: [freq[i], what+'_freq', class_name, feat_name, '', 'gauge'] for i in range(len(freq))}
    except ZeroDivisionError:
        freq_dict = {}

    return merge_dicts(freq_dict, count_dict)


def label_matrix_metrix(predict_proba_matrix, label_matrix, what, class_name, threshold=0.5):
    """
    Calculates accuracy, precision, recall, f1 metrics from predict_proba and label matrices
    """

    label_dict = {}

    batch_size = label_matrix.shape[0]
    num_classes = label_matrix.shape[1]

    if num_classes == 2:
        # manage threshold for binary classification with threshold
        predict_matrix = np.array([[int(line[1] <= threshold), int(line[1] > threshold)] for line in predict_proba_matrix])
    else:
        predict_matrix = np.array([[1 if x == max(line) else 0 for x in line] for line in predict_proba_matrix])

    for class_num in range(num_classes):
        pred_vector = predict_matrix[:, class_num]
        label_vector = label_matrix[:, class_num]

        accuracy = calculate_accuracy(pred_vector, label_vector)
        precision = calculate_precision(pred_vector, label_vector)
        recall = calculate_recall(pred_vector, label_vector)
        if recall + precision > 0.0:
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0.0

        perf_metrics = [accuracy, precision, recall, f1]
        perf_names = ['accuracy', 'precision', 'recall', 'f1']

        perf_key_names = ['class_' + str(class_num) + "_" + x for x in perf_names]

        temp_dict = {perf_key_names[i]: [perf_metrics[i], perf_names[i], class_name + str(class_num), '', '', 'gauge']
                     for i in range(len(perf_key_names))}

        label_dict = merge_dicts(label_dict, temp_dict)

    return label_dict


def predict_proba_to_predict(predict_proba_matrix):
    """
    Converts the predict proba matrix to a binary predict matrix
    """
    return np.array([[1 if x == max(line) else 0 for x in line] for line in predict_proba_matrix])


def calculate_precision(pred_vector, label_vector):
    """
    Calculate the precision of the predicted values.
    """
    n = len(pred_vector)
    TP = sum([(label_vector[i] == 1) and (pred_vector[i] == 1) for i in range(n)])
    FP = sum([(label_vector[i] == 0) and (pred_vector[i] == 1) for i in range(n)])

    if TP + FP > 0.0:
        result = TP / (TP + FP)
    else:
        result = 0.0
    return result


def calculate_recall(pred_vector, label_vector):
    """
    Calculate the recall of the predicted values.
    """
    n = len(pred_vector)
    TP = sum([(label_vector[i] == 1) and (pred_vector[i] == 1) for i in range(n)])
    FN = sum([(label_vector[i] == 1) and (pred_vector[i] == 0) for i in range(n)])

    if TP + FN > 0.0:
        result = TP / (TP + FN)
    else:
        result = 0.0
    return result


def calculate_accuracy(pred_vector, label_vector):
    """
    Calculate the accuracy of the predicted values.
    """
    n = len(pred_vector)
    TP = sum([(label_vector[i] == 1) and (pred_vector[i] == 1) for i in range(n)])
    TN = sum([(label_vector[i] == 0) and (pred_vector[i] == 0) for i in range(n)])
    if n > 0:
        result = (TN + TP) / n
    else:
        result = 0.0
    return result


def calculate_metrix_dicts(predict_proba_matrix, input_matrix, label_matrix, threshold=0.5):
    """
    Converts the data matrix(s) into a dict of metrics.
    :return: 3 dicts of format {'unique_key':[value, what, class_name, feat_name, stat, target_type]} or empty dicts
    """
    input_metrix = {}
    label_metrix = {}
    prediction_1_metrix = {}
    prediction_2_metrix = {}

    if input_matrix is not None:
        input_metrix = input_matrix_metrix(input_matrix)

    if label_matrix is not None:
        label_metrix = label_matrix_metrix(predict_proba_matrix, label_matrix, '', 'class_', threshold)

    prediction_classes_metrix = prediction_matrix_metrix(predict_proba_matrix, threshold)

    if len(predict_proba_matrix) > 0:
        prediction_1_metrix = prediction_first_values_metrix(predict_proba_matrix)
        prediction_2_metrix = prediction_second_values_metrix(predict_proba_matrix)


    return (input_metrix, merge_dicts(prediction_classes_metrix, prediction_1_metrix, prediction_2_metrix), label_metrix)

