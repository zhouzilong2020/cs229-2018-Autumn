import collections

import numpy as np

import util
import svm


def get_words(message):
    """
    Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = str(message).lower().split(' ')
    result = []
    for word in words:
        pos = -1
        word_len = -1 * len(word)
        while pos >= word_len and not word[pos].isalpha() and not word[pos].isnumeric():
            pos -= 1
        if pos == word_len:
            continue
        elif pos + 1 == 0:
            result.append(word)
        else:
            result.append(word[:pos + 1])
    return result

    # *** END CODE HERE ***


def create_dictionary(messages):
    """
    Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    word_cnt = dict()
    for mes in messages:
        # 删去在一条message中给重复出现的词
        mes = list(set(get_words(mes)))
        for word in mes:
            if word_cnt.get(word) is not None:
                word_cnt[word] += 1
            else:
                word_cnt[word] = 1
    keys = word_cnt.keys()
    # 使用字典生成list，并删去出现次数小于5的词
    aux = [(word_cnt[k], k) for k in keys if word_cnt[k] > 5]
    aux.sort(key=lambda o: o[0], reverse=True)
    aux = [(a[1], index) for index, a in enumerate(aux)]
    return dict(aux)
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """
    Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    word_vectors = np.zeros((len(messages), len(word_dictionary.keys())))

    for i, mes in enumerate(messages):
        words = get_words(mes)
        for word in words:
            word_id = word_dictionary.get(word)
            if word_id is not None:
                word_vectors[i][word_id] += 1
    return word_vectors
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        每一列是一个样本,每一行是一个特征
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    spam_matrix = matrix[:][labels == 1]
    non_spam_matrix = matrix[:][labels == 0]

    p_spam_word = np.sum(spam_matrix, axis=0)
    p_non_spam_word = np.sum(non_spam_matrix, axis=0)

    # laplacian smoothing
    p_spam_word = (p_spam_word + 1) / (np.sum(p_spam_word) + 2)
    p_non_spam_word = (p_non_spam_word + 1) / (np.sum(p_non_spam_word) + 2)

    return {'word': {'spam': p_spam_word,
                     'non-spam': p_non_spam_word,
                     'total': np.sum(matrix, axis=0) / np.sum(matrix)},
            'text': {
                'spam': np.sum(labels) / labels.shape[0],
                'non-spam': 1 - np.sum(labels) / labels.shape[0]}
            }
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        每一列是一个样本,每一行是一个特征
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    pred = np.zeros(matrix.shape[0])
    p_spam = model['text']['spam']
    pw_spam = model['word']['spam']
    pw_non_spam = model['word']['non-spam']
    pw_total = model['word']['total']

    for i in range(matrix.shape[0]):
        row = matrix[i]
        prob = p_spam
        for word_id, cnt in enumerate(row):
            if cnt == 0:
                continue
            prob = prob * (pw_spam[word_id] / pw_total[word_id]) ** cnt
        pred[i] = int(prob > .5)
    return pred
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    words = list(dictionary.keys())
    pw_spam = model['word']['spam']
    pw_non_spam = model['word']['non-spam']
    freq = []
    for i, word in enumerate(words):
        freq.append((word, np.log(pw_spam[i] / pw_non_spam[i])))
    freq.sort(key=lambda o: o[1], reverse=True)
    result = []
    for i in range(5):
        result.append(freq[i][0])
    return result
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        eval_matrix: The word counts for the validation data
        eval_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    result = [[r, 0] for r in radius_to_consider]
    for i, radius in enumerate(radius_to_consider):
        pred = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        result[i][1] = np.mean(pred == val_labels)
    result.sort(key=lambda o: o[1], reverse=True)
    return result[0][0]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')

    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary_my', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix_my', train_matrix[:100, :])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions_my', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words_my', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius_my', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
