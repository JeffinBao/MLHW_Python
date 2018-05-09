# coding: utf-8

import pandas as pd
import sys

class MultinomialClf:
    def __init__(self, path):
        self.path = path

    def pre_process(self):
        names = ['sentence', 'label']
        # separator is '\t'
        df = pd.read_csv(self.path, sep='\t', header=None, names=names)
        X = df['sentence']
        y = df['label']

        from sklearn.feature_extraction.text import CountVectorizer
        # tokenizing and filtering stopwords
        count_vect = CountVectorizer()
        X_counts = count_vect.fit_transform(X)
        from sklearn.feature_extraction.text import TfidfTransformer
        # implement tf-idf, which penalizes words that appear in most of the sentences
        tf_tranformer = TfidfTransformer()
        X_tfidf = tf_tranformer.fit_transform(X_counts)

        from sklearn.model_selection import train_test_split as tt_split
        X_train_tfidf, X_test_tfidf, y_train, y_test = tt_split(X_tfidf, y)

        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def train_predict(self, X_train_tfidf, X_test_tfidf, y_train, y_test):
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import GridSearchCV
        # estimator parameters dict
        parameters = {'alpha': (1, 0.8, 0.5, 0.1, 0.01, 0.001), 'fit_prior': (True, False)}
        gs_clf = GridSearchCV(MultinomialNB(), parameters, cv=20, n_jobs=-1)
        gs_clf.fit(X_train_tfidf, y_train)
        print('best parameters shows as below:')
        for param_name in sorted(parameters.keys()):
            print('%s: %r' % (param_name, gs_clf.best_params_[param_name]))

        # make prediction of test data set
        predicted = gs_clf.predict(X_test_tfidf)
        from sklearn import metrics
        print(metrics.classification_report(predicted, y_test))

        y_score = gs_clf.predict_proba(X_test_tfidf)
        self.__plot_roc(y_test, y_score)

    def __plot_roc(self, y_test, y_score):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)

        import matplotlib.pyplot as plt
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0]) # x coordinate starts from 0.0 to 1.0
        plt.ylim([0.0, 1.05]) # y coordiante starts from 0.0 to 1.05
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right') # where the label should be put
        plt.show()


if __name__ == '__main__':
    # path = str(sys.argv[1])

    path = ''
    multinomialNB = MultinomialClf(path)
    X_train_tfidf,X_test_tfidf, y_train, y_test = multinomialNB.pre_process()
    multinomialNB.train_predict(X_train_tfidf,X_test_tfidf, y_train, y_test)