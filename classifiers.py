import math

import numpy as np


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """

    def predict(self, X):
        return [0] * len(X)


class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """

    def __init__(self):
        self.log_prior = [0, 0]
        self.loglikelihood = []

    def fit(self, X, Y):
        num_sentences = len(X)
        num_unique_words = len(X[0])
        num_classes = [0, 0]
        total_words = [0, 0]
        # find total words for the classes (0, 1)
        for i in range(len(X)):
            for z in X[i]:
                total_words[Y[i]] += z
            total_words[Y[i]] += 1
        # find total of each classes
        for i in Y:
            if i == 0:
                num_classes[0] += 1
            else:
                num_classes[1] += 1
        self.loglikelihood = [[0] * num_unique_words for i in range(2)]
        bigdoc = [[0] * num_unique_words for i in range(2)]
        for c in range(0, 2):
            self.log_prior[c] = math.log(num_classes[c] / num_sentences)
            for i in range(len(X)):
                if Y[i] == c:
                    for z in range(num_unique_words):
                        bigdoc[c][z] += X[i][z]
            for w in range(num_unique_words):
                self.loglikelihood[c][w] = math.log((bigdoc[c][w] + 1) / (total_words[c]))
        return (X, self.loglikelihood)

    def predict(self, X):
        summ = [0, 0]
        solu = [0] * len(X)
        for i in range(len(X)):
            for c in range(0, 2):
                summ[c] = self.log_prior[c]
                for z in range(len(X[i])):
                    if X[i][z] != 0:
                        summ[c] += self.loglikelihood[c][z]
            if summ[0] > summ[1]:
                solu[i] = 0
            else:
                solu[i] = 1
        return solu

