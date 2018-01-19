import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

stopword_list = ["a", "about" ,"above", "after", "again" ,"against" ,"all" ,"am" ,"an" ,"and", "any", "are", "aren't" ,"as" ,"at" ,"be", "because", "been" ,"before" ,"being" ,"below", "between", "both", "but", "by", "can't" ,"cannot" ,"could" ,"couldn't", "did" ,"didn't", "do" ,"does" ,"doesn't" ,"doing" ,"don't" ,"down", "during" ,"each", "few", "for" ,"from", "further", "had", "hadn't" ,"has" ,"hasn't", "have", "haven't", "having" ",he" ,"he'd", "he'll", "he's", "her", "here" ,"here's", "hers", "herself", "him", "himself" ,"his" ,"how", "how's", "i", "i'd" ,"i'll", "i'm", "i've", "if" ,"in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most" ,"mustn't" ,"my" ,"myself" ,"no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours","ourselves", "out", "over", "own", "same", "shan't", "she", "she'd" ,"she'll", "she's", "should", "shouldn't" ,"so", "some", "such", "than", "that", "that's" ,"the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where" ,"where's", "which", "while" ,"who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"];

# Create a dictionary of words with its frequency

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    # Paste code for non-word removal here(code snippet is given below)
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix

# Create a dictionary of words with its frequency

train_dir = 'ling-spam/train-mails'
dictionary = make_Dictionary(train_dir)
# print dictionary
# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier

model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)

# Test the unseen mails for Spam
test_dir = 'ling-spam/test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
print(confusion_matrix(test_labels,result1))
print(confusion_matrix(test_labels,result2))

