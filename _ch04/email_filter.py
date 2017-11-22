import re
from bayes import *


def text_parse(text):
    splited_words = re.split(r'\W*', text)
    return [word.lower() for word in splited_words if len(word) >2]


def train_email_filter_model(emails, training_set, classes, vocabulary_list):
    train_matrix = []
    train_classes = []
    for doc_index in training_set:
        train_matrix.append(words2vector(vocabulary_list,
            emails[doc_index]))
        train_classes.append(classes[doc_index])
    p0, p1, p_spam = nb_train(train_matrix, train_classes)
    return p0, p1, p_spam

def get_test_set():
    training_set = [i for i in range(50)]
    test_set = []
    for i in range(10):
        random_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[random_index])
        del(training_set[random_index])
    return training_set, test_set


def load_emails():
    emails = []
    classes = []
    for i in range(1, 26):
        with open('./Ch04/email/spam/' + str(i) + '.txt', 'r', errors='ignore') as f:
            words = text_parse(f.read())
            emails.append(words)
            classes.append(1)
        with open('./Ch04/email/ham/' + str(i) + '.txt', 'r', errors='ignore') as f:
            words = text_parse(f.read())
            emails.append(words)
            classes.append(0)
    return emails, classes


def spam_email_test():
    emails, classes = load_emails()
    vocabulary_list = create_vocabulary_list(emails)
    training_set, test_set = get_test_set()
 
    p0, p1, p_spam = train_email_filter_model(emails, training_set, classes, vocabulary_list)

    error_counter = 0
    for test_index in test_set:
        word_vector = words2vector(vocabulary_list, emails[test_index])
        res = naive_bayes_classify(word_vector, p0, p1, p_spam)
        if res != classes[test_index]:
            error_counter += 1

    print('The error rate is ', float(error_counter)/len(test_set))
