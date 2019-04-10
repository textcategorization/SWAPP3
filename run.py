# Brute Force Approach
# referenced
#  https://realpython.com/introduction-to-flask-part-2-creating-a-login-page/
#  https://www.tutorialspoint.com/flask/flask_sending_form_data_to_template.htm
#  Referenced: https://www.tutorialspoint.com/flask/flask_file_uploading.htm
#  https://flask-excel.readthedocs.io/en/latest/
# To do finish logic for login, need to pull and compare from DB and need to save password using hash
# using Flask Scikit learn, pandas and numpy framework

from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import xlsxwriter
import time
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer

app = Flask(__name__)


# index
@app.route('/')
# just rendering template you have to create templates in templates folder remember this is just the view
def index():
    return render_template('index.html', content_type='application/json')


@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'Test@gmail.com' or request.form['password'] != 'Test123':
            error = 'Invalid Username and Password. Please try again.'
        else:
            return redirect(url_for('results'))
    return render_template('index.html', error=error)


# results can only be seen after login
@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/analyze_text', methods=['POST', 'GET'])
def analyze_text():
    if request.method == 'POST':
        corpus = "20news.xlsx"
        df = pd.read_excel(corpus)
        data_frame = [df.to_html(classes='table table-striped', header="true")]
        df_desc = df.describe()
        return render_template('results.html', data_frame=data_frame, df_desc=df_desc, content_type='application/json')


@app.route('/vectorizer', methods=['POST', 'GET'])
def vectorize_text():
    if request.method == 'POST':
        corpus = "20news.xlsx"
        data = pd.read_excel(corpus)
        data_features = data["news"]
        data_labels = data["category"]

        # Vectorization of Features
        vec = TfidfVectorizer(stop_words='english')
        vec_feature = vec.fit_transform(data_features)
        # Encoding of Labels
        encode = LabelEncoder()
        encode_label = encode.fit_transform(data_labels)
        vec_vocab = vec.vocabulary_

        # Training Data
        X_train, X_test, y_train, y_test = train_test_split(vec_feature, encode_label, shuffle=True)

        # Mini K-Means Model
        start = time.time()
        # defining clusters
        num_clust = data_labels.nunique()
        # Mini-Batch K-means algorithm
        mini_k_means = MiniBatchKMeans(n_clusters=num_clust, compute_labels=True, random_state=None,
                                       batch_size=100).fit(vec_feature)
        # printing Top list of terms by cluster
        data_cluster = []
        centroids = mini_k_means.cluster_centers_.argsort()[:, ::-1]
        terms = vec.get_feature_names()
        filenames = data["filename"]

        # Top list of terms by cluster
        for i in range(num_clust):
            pred_clust = i

            for j in centroids[i, :3]:
                p = Path(filenames.iloc[i])
                actual_label = p.relative_to(
                    'C:\\Users\\Pam\\PycharmProjects\\20NEWSGROUP\\NewsData\\20news-bydate-train')
                pred_top_terms = terms[j]
                data_cluster.append((actual_label, pred_clust, pred_top_terms))
        data_clust_label = ["Actual Label", "Pred Cluster Label", "Top Terms"]
        clust_df = pd.DataFrame.from_records(data_cluster, columns=data_clust_label)
        clust_df.to_excel('20news_clust.xlsx', index='false', encoding='utf-8', engine='xlsxwriter')
        km_data_frame = [clust_df.to_html(classes='table table-striped', header="true")]
        km_df_desc = clust_df.describe()
        km_acc = accuracy_score(encode_label, mini_k_means.predict(vec_feature))
        end = time.time()
        exc_time = end - start
        km_time = exc_time

        # SVM
        # svm
        svm_start = time.time()
        svm_model = svm.LinearSVC(C=1.0)
        svm_model.fit(X_train, y_train)

        pred_svm = svm_model.predict(X_test)
        svm_acc = accuracy_score(pred_svm, y_test)
        svm_end = time.time()
        svm_exec = svm_end - svm_start

        # DL
        trainsize = int(len(data) * .8)
        testsize = int(len(data) * .2)
        trainpost = data["news"][:trainsize]
        traintags = data["category"][:trainsize]
        trainfilenames = data["filename"][:trainsize]

        testpost = data["news"][:testsize]
        testtags = data["category"][:testsize]
        testfilenames = data["filename"][:testsize]

        num_label = data["category"].nunique()
        # verifying there are only 20 unique labels
        vocab_size = data["news"].nunique()
        # verifying unique vocab
        batch_size = 100

        # Tokenizer
        token = Tokenizer(num_words=vocab_size)
        token.fit_on_texts(trainpost)
        token.fit_on_texts(testpost)

        # checking results
        x_train = token.texts_to_matrix(trainpost, mode='tfidf')
        x_test = token.texts_to_matrix(testpost, mode='tfidf')

        # Binarize labels in a one-vs-all fashion
        # Several regression and binary classification algorithms are available in scikit-learn. A simple way to extend these algorithms to the multi-class classification case is to use the so-called one-vs-all scheme.
        # At learning time, this simply consists in learning one regressor or binary classifier per class. In doing so, one needs to convert multi-class labels to binary labels (belong or does not belong to the class). LabelBinarizer makes this process easy with the transform method.
        # At prediction time, one assigns the class for which the corresponding model gave the greatest confidence.
        # Assigning probability to each label
        encode = LabelBinarizer()
        encode.fit(traintags)
        encode.transform(traintags)
        encode.transform(testtags)
        y_train = encode.transform(traintags)
        y_test = encode.transform(testtags)
        # verifying results
        ##### print("print y train\n", Y_train, "\nprint y test", Y_test)
        # added to resolve shape error.

        # https://keras.io/losses/
        # loss function, optimization score function
        # What is the difference between categorical_crossentropy and sparse_categorical_crossentropy?
        # Also why is it better to use crossentropy vs mean square error
        # The Adam optimization algorithm is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing.
        dl_start = time.time()
        model = Sequential()
        model.add(Dense(512, input_shape=(vocab_size,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_label))
        model.add(Activation("softmax"))
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
        summmary = model.summary()
        hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=1, validation_split=0.1)
        dl_end = time.time()
        dl_class = []
        label = encode.classes_
        for i in range(10):
            pred = model.predict(np.array([x_test[i]]))
            pred_label = label[np.argmax(pred[0])]
            file_name = testfilenames.iloc[i]
            # print("file: ", file)
            # print("predicted label: " , pred_label)
            # print("Actual label: " , testtags[i])
            p = Path(file_name)
            file = p.relative_to('C:\\Users\\Pam\\PycharmProjects\\20NEWSGROUP\\NewsData\\20news-bydate-train')
            dl_class.append((file, pred_label, testtags[i]))

        dl_class_label = ["Filename", "Pred Class", "Actual Label"]
        dl_df = pd.DataFrame.from_records(dl_class, columns=dl_class_label)
        dl_df.to_excel('20news_dl_class.xlsx', index='false', encoding='utf-8', engine='xlsxwriter')
        dl_data_frame = [dl_df.to_html(classes='table table-striped', header="true")]
        dl_df_desc = dl_df.describe()

        score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        dl_acc = score[1]
        dl_time = dl_end - dl_start

        return render_template('results.html', vec_data=vec_feature, vec_vocab=vec_vocab, km_data_frame=km_data_frame,
                               km_df_desc=km_df_desc, km_acc=km_acc, km_time=km_time, svm_acc=svm_acc,
                               svm_exec=svm_exec, dl_data_frame=dl_data_frame, dl_df_desc=dl_df_desc, dl_acc=dl_acc,
                               dl_time=dl_time, content_type='application/json')


@app.route('/logout', methods=['POST', 'GET'])
def logout():
    return render_template('index.html')


# running the app
if __name__ == '__main__':
    app.run(debug=True)
