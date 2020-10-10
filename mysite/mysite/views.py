# created namrata

from django.shortcuts import render
from boilerpy3 import extractors
from matplotlib.pyplot import plot
from sklearn import metrics

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np



def index(request):
    return render(request, 'index.html')


def extract(request):
    extractor = extractors.ArticleExtractor()
    httpurl = request.POST.get('Url')
    doc = extractor.get_doc_from_url(httpurl)
    content = doc.content
    print(content)
    file = open("output.txt", "w+")
    file.write(content)
    file.close()
    value = {'httpurl': httpurl, 'content': content}
    context = {

        'value': value,
    }
    return render(request, 'text.html', context)


def nb_news(request):
    myurl = request.POST.get('Url')
    twenty_train = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(twenty_train.data)
    x_train_counts.shape
    print(count_vect.vocabulary_.get(u'algorithm'))
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    print(x_train_tfidf.shape)
    clf = MultinomialNB().fit(x_train_tfidf, twenty_train.target)
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=1.0,
                              fit_prior=True,
                              class_prior= None
                              )),
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)


    twenty_test = fetch_20newsgroups(subset='test',
                                     shuffle=True, random_state=42)
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    np.mean(predicted == twenty_test.target)

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3,
                                               max_iter=5, tol=None)),
                         ])
                           
    text_clf.fit(twenty_train.data, twenty_train.target)
    f = open("output.txt", "r")
    if f.mode == 'r':
        docs_new = [f.read()]

    predicted = text_clf.predict(docs_new)

    for doc, category in zip(docs_new, predicted):
        value = (' %s' % (twenty_train.target_names[category]))
    if 'talk' in value:
        bloc = "you cannot access this url"
    else:
        bloc = myurl
    predicted = text_clf.predict(docs_test)
    score = np.mean(predicted == twenty_test.target)
    print(metrics.classification_report(twenty_test.target, predicted,target_names = twenty_test.target_names))
    plot(metrics.confusion_matrix(twenty_test.target, predicted))
    classified_cat = {'category': value, 'acc': score, 'block': bloc}
    context = {
        'classified_cat': classified_cat,

    }

    return render(request, 'classify.html', context)