import numpy as np
import pandas as pd

def check_output_format(pred_matrix, n_classes):
    """
    Checks the predict proba matrix format of data and shape
    :param pred_matrix: Predict_proba matrix
    :return: True if all floats, shape of predict_proba matrix
    """
    check_test = all([check_all_floats([element for element in line]) for line in pred_matrix])
    check_msg = "PREDICT_PROBA matrix not a float matrix. " if not check_test else ""
    n_lines, n_columns = pred_matrix.shape
    check_msg = check_msg + "PREDICT_PROBA matrix must have at least 2 columns" if n_columns < 2 else check_msg
    check_msg = check_msg + "PREDICT_PROBA num columns is not consistent : {} in data vs {} in agent declaration".format(n_classes, n_columns) if n_classes != n_columns else check_msg
    return check_msg, n_lines, n_columns


def check_all_floats(list_of_elements):
    "Return True if all elements in the list are floats, else False"
    is_float_32 = all(map(lambda i: isinstance(i, np.float32), list_of_elements))
    is_float = all(map(lambda i: isinstance(i, float), list_of_elements))
    return is_float_32 | is_float


def check_all_ints(list_of_elements):
    "Return True if all elements in the list are ints, else False"
    is_int_32 = all(map(lambda i: isinstance(i, np.int32), list_of_elements))
    is_int = all(map(lambda i: isinstance(i, int), list_of_elements))
    return is_int_32 | is_int


def check_input_format(input_matrix, n_lines):
    "Check that the input matrix has the expected number of lines"
    n_lines_, n_columns_ = input_matrix.shape
    check_msg = "INPUT matrix format problem. " if not n_lines_ == n_lines else ""
    return check_msg


def check_label_format(label_matrix, n_lines, n_columns):
    "Check that the label matrix has the expected number of lines and columns"
    n_lines_, n_columns_ = label_matrix.shape
    check_msg = "LABEL matrix format problem. " if not (n_lines_ == n_lines) & (n_columns_ == n_columns) else ""
    return check_msg


def binarize(label_matrix, n_classes):
    """
    If labels are given as one column of classes indexes, binarize to number of classes size with 1 at class index
    :param label_matrix:
    :return: binarized_label_matrix if applicable
    """
    if label_matrix.ndim == 1:
        label_matrix = label_matrix.reshape(label_matrix.shape[0], 1)
    if label_matrix.shape[1] == 1:
        return np.array(list(map(lambda x: [0 if i != x else 1 for i in range(n_classes)], label_matrix)))
    else:
        return label_matrix


def dataframe_to_values(df):
    """
    If data given are of type dataframe, return their values, else do nothing
    :param df:
    :return: values of dataframe if input is a dataframe, else input
    """
    result = df.values if isinstance(df, pd.DataFrame) else df
    return result


def check_format(pred_matrix, input_matrix, label_matrix, n_classes):
    """
    Check format of all matrices, and apply np.array() to all matrices.
    If mandatory predict_proba matrix is None, then make it an empty matrix of shape (0, n_classes)
    :return: check_msg, pred_matrix, input_matrix, label_matrix
    """
    if pred_matrix is None:
        pred_matrix = np.empty(shape=[0, n_classes])
        pred_matrix = np.array(pred_matrix)
    else:
        pred_matrix = dataframe_to_values(pred_matrix)
        pred_matrix = np.array(pred_matrix)
    check_msg, n_lines, n_columns = check_output_format(pred_matrix, n_classes)
    if input_matrix is not None:
        input_matrix = dataframe_to_values(input_matrix)
        input_matrix = np.array(input_matrix)
        check_msg += check_input_format(input_matrix, n_lines)
    if label_matrix is not None:
        label_matrix = dataframe_to_values(label_matrix)
        label_matrix = np.array(label_matrix)
        label_matrix = binarize(label_matrix, n_columns)
        check_msg += check_label_format(label_matrix, n_lines, n_columns)
    return check_msg, pred_matrix, input_matrix, label_matrix
