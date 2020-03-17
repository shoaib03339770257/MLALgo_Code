# MLALgo_Code
Machine Learning Algorithm for Multi-class classicification


import nltk
import pandas as pd
import numpy as np
from matplotlib.pyplot import plot
from sklearn.metrics import *
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import sklearn
import nltk
import warnings
warnings.filterwarnings('always')
np.random.seed(5000)

pd.set_option('display.max_columns', 3000)  # DISPLAYS MAXIMUM OF THE 300 COLUNS
pd.set_option('display.width', 5000)  # DISPLAY WIDTH
pd.set_option('display.max_rows', 1000)  # Display max rows

Corpus = pd.read_csv(r"separated.csv", encoding='latin-1')

print(Corpus.Label.value_counts())
# Step - a : Remove blank rows if any.

Corpus['comment'].dropna(inplace=True)

# Dataframes = Corpus
# Dataframes =Corpus.loc[(Corpus['Label'] != 2.0)]
# print(Dataframes)
# Dataframes = Dataframes.loc[(Corpus['Label'] != 3.0)]
# print(Dataframes)
# Dataframes = Dataframes.loc[(Corpus['Label'] != 4.0)]
# print(Dataframes)
# Dataframes.to_csv("E:/separated.csv")

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
# Corpus['comment'] = [entry.lower() for entry in Corpus['comment']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['Comm'] = Corpus['comment'].apply(nltk.word_tokenize)
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
Corpus['Comm'] = Corpus['Comm'].astype("str")
df = sklearn.utils.shuffle(Corpus)
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
# How are you saad bhai ????
for index, entry in enumerate(Corpus['Comm']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index, 'text_final'] = str(Final_words)

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['Label'],

                                                                        test_size=0.1)

#Encoder = LabelEncoder()
#Train_Y = Encoder.fit_transform(Train_Y)
#Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=19000)
Tfidf_vect.fit(Corpus['Comm'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# print(Tfidf_vect.vocabulary_)

# print(Train_X_Tfidf)
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_mean=False)
X_train = sc.fit_transform(Train_X_Tfidf)
X_test = sc.transform(Test_X_Tfidf)
SVM = svm.SVC(kernel='rbf', random_state=10, gamma=4, C=1)
SVM.fit(X_train, Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test)
# Use accuracy_score function to get the accuracy

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Test_Y, predictions_SVM)
print("Confusion matrix: /n",cm)
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=SVM, X=X_train, y=Train_Y, cv=10,scoring=None)
print(accuracies.mean())
print("SVm Deployed accuracy: ", accuracies)
print(accuracies.std())
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)


y_score = SVM.decision_function(Test_X_Tfidf)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(Test_Y, y_score)



#from sklearn.metrics import plot_precision_recall_curve
#import matplotlib.pyplot as plt

#disp = plot_precision_recall_curve(SVM,Test_X_Tfidf, Test_Y)
#print(disp.ax_.set_title('2-class Precision-Recall curve: '
 #                  'AP={0:0.2f}'.format(average_precision)))


print("f1 score macro", sklearn.metrics.f1_score(Test_Y, predictions_SVM, average='macro'))
print("f1 score micro", sklearn.metrics.f1_score(Test_Y, predictions_SVM, average='micro'))
print("precision score", sklearn.metrics.precision_score(Test_Y, predictions_SVM, average='macro'))
print("recall score", sklearn.metrics.recall_score(Test_Y, predictions_SVM, average='macro'))

print('Average precision-recall score: {0:0.2f}'.format(average_precision))


