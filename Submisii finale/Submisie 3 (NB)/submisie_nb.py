import numpy as np
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# functie pentru citirea datelor din fisiere
def read_data(cale):
    data = []
    iduri = []
    with open(cale, 'r', encoding='utf-8') as fin:
        line = fin.readline()
        while line:
            cuvinte_text = line.split()
            iduri.append(cuvinte_text[0])
            data.append(cuvinte_text[1:])
            line = fin.readline()
    return iduri, data


class BagOfWords:
    def __init__(self):
        self.words = []
        self.vocabulary = {}

    def build_vocabulary(self, sentences):
        for sentence in sentences:
            for word in sentence:
                if word not in self.words:
                    # print(word)
                    self.words.append(word)
                    self.vocabulary[word] = self.words.index(word)

    def get_features(self, sentences):
        features = np.zeros((len(sentences), len(self.words)))
        for index, sentence in enumerate(sentences):
            for word in sentence:
                # print(word)
                if word in self.words:
                    features[index][self.vocabulary[word]] += 1
        return features


# functie pentru normalizarea datelor
def normalize_data(train_data, test_data, norm=None):
    if norm == 'L1':
        train_data /= np.sum(abs(train_data), axis=1, keepdims=True)
        test_data /= np.sum(abs(test_data), axis=1, keepdims=True)

    if norm == 'L2':
        train_data /= np.sqrt(np.sum(train_data ** 2, axis=1, keepdims=True))
        test_data /= np.sqrt(np.sum(test_data ** 2, axis=1, keepdims=True))

    return train_data, test_data


# functie pentru scriere in fisier
def write_submission(nume_fisier, predictii, iduri):
    with open(nume_fisier, 'w') as fout:
        fout.write("id,label\n")
        for id_text, pred in zip(iduri, predictii):
            fout.write(str(id_text) + ',' + str(int(pred)) + '\n')


# citirea datelor
dir_path = './sample_data/'
train_labels_path = os.path.join(dir_path, 'train_labels.txt')
train_data_path = os.path.join(dir_path, 'train_samples.txt')
test_data_path = os.path.join(dir_path, 'test_samples.txt')

validation_labels_path = os.path.join(dir_path, 'validation_labels.txt')
validation_samples_path = os.path.join(dir_path, 'validation_samples.txt')

iduri_train, train_data = read_data(train_data_path)
iduri_train_labels, train_labels = read_data(train_labels_path)
iduri_test, test_data = read_data(test_data_path)

iduri_valid, valid_data = read_data(validation_samples_path)
iduri_valid_labels, valid_labels = read_data(validation_labels_path)


'''
############################ testare model #################################
# antrenare pe train_samples si testare pe validation_samples

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)

bow = BagOfWords()
bow.build_vocabulary(train_data)

train_features = bow.get_features(train_data)
valid_features = bow.get_features(valid_data)

norm_train_features, norm_valid_features = normalize_data(train_features, valid_features, 'L2')

clasificator = MultinomialNB(alpha=.01)
clasificator.fit(norm_train_features, train_labels.ravel())
predictions = clasificator.predict(norm_valid_features)

print("Rezultate classification report: \n")
print(classification_report(valid_labels, predictions, labels=[0, 1]))  
print('\nMatricea de confuzie: \n')
print(confusion_matrix(valid_labels, predictions))
print('\n\n')

'''

################# antrenare pe train_samples + validation_samples ####################

all_data = train_data + valid_data
all_labels = train_labels + valid_labels
all_labels = np.array(all_labels)

bow = BagOfWords()
bow.build_vocabulary(all_data)

all_features = bow.get_features(all_data)
test_features = bow.get_features(test_data)

norm_all_features, norm_test_features = normalize_data(all_features, test_features, 'L2')

clasificator = MultinomialNB(alpha=.01)
clasificator.fit(norm_all_features, all_labels.ravel())
predictions = clasificator.predict(test_features)

write_submission("submisie_nb.csv", predictions, iduri_test)
