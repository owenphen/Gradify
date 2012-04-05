import csv
from pprint import pprint
from sklearn import svm
from nltk.tokenize import sent_tokenize as ntlk_sent_tokenize
from nltk.tokenize import word_tokenize as ntlk_word_tokenize
import nltk
from nltk.corpus import brown
import random
from filecache import filecache

words_file = open('/usr/share/dict/words', 'r')
ALL_ENGLISH_WORDS = set(line.strip() for line in words_file)
words_file.close()

@filecache()
def load_brown_freq_ratios():
    brown_freqdist = nltk.FreqDist([w.lower() for w in brown.words()])
    num_words = len(brown.words())
    ratios = {}
    for word, number in brown_freqdist.iteritems():
        ratios[word] = float(number)/num_words
    return ratios
brown_freq_ratios = load_brown_freq_ratios()

def guess(in_essays, withold=1):
    features = [make_features(in_essay) for in_essay in in_essays]
    scores = [in_essay.normalized_rating() for in_essay in in_essays]
    clf = svm.SVR()
    clf.fit(features[withold:], scores[withold:])
    svm.SVR(C=1.0, coef0=0.0, degree=3, epsilon=0.1, gamma=0.5,
      kernel='rbf', probability=False, shrinking=True,
      tol=0.001)

    predictions = clf.predict(features[:withold])
    return zip(predictions, in_essays[:withold])

class Essay(object):
    max_ratings_per_type = {
        1:12,
        3:3,
        4:3,
        5:4,
        6:4,
        7:30,
        8:60
    }
    def __init__(self, in_row):
        self.essay_id = int(in_row[0] or -1)
        self.set_id = int(in_row[1])
        self.essay = in_row[2]
        self.domain1_score = int(in_row[6])

    def max_rating(self):
        max_rating = Essay.max_ratings_per_type[self.set_id]
        assert self.domain1_score <= max_rating, 'score %s was higher than max %s for type %s' % (self.domain1_score, max_rating, self.set_id)
        return max_rating

    def normalized_rating(self):
        return float(self.domain1_score) / self.max_rating()

    def __repr__(self):
        return "<Essay Normed: %f2 Scored: %s/%s \"%s...\">" % (self.normalized_rating(), self.domain1_score, self.max_rating(), self.essay[:50])

def make_features(in_essay):
    features = []
    features.append(Feature.num_unique_words(in_essay))
    features.append(Feature.ratio_unique_words(in_essay))
    features.append(Feature.average_word_length(in_essay))
    features.append(Feature.period_ratio(in_essay))
    features.append(Feature.comma_ratio(in_essay))
    features.append(Feature.ratio_dict_words(in_essay))
    features.append(Feature.average_sentence_length(in_essay))
    features.append(Feature.num_unique_misspellings(in_essay))
    features.append(Feature.brown_freq_diff(in_essay))
    return features

class Feature:
    @staticmethod
    def num_unique_words(in_essay):
        return len(set(tokenize(in_essay)))

    @staticmethod
    def ratio_unique_words(in_essay):
        return float(len(tokenize(in_essay))) / Feature.num_unique_words(in_essay)

    @staticmethod
    def average_word_length(in_essay):
        lengths = tuple(len(word) for word in tokenize(in_essay) if not word.startswith("@"))
        return float(sum(lengths)) / len(lengths)

    @staticmethod
    def period_ratio(in_essay):
        return float(in_essay.essay.count('.')) / len(in_essay.essay)

    @staticmethod
    def comma_ratio(in_essay):
        return float(in_essay.essay.count(',')) / len(in_essay.essay)

    @staticmethod
    def ratio_dict_words(in_essay):
        words = [w for w in tokenize(in_essay) if not w.startswith("@")]
        words_in_dict = [w for w in words if w in ALL_ENGLISH_WORDS]
        return float(len(words_in_dict))/len(words)

    @staticmethod
    def num_unique_misspellings(in_essay):
        misspelled_words = [w for w in tokenize(in_essay) if not w.startswith("@") and w not in ALL_ENGLISH_WORDS]
        return len(set(misspelled_words))

    @staticmethod
    def average_sentence_length(in_essay):
        lengths = [len(sentence) for sentence in sent_tokenize(in_essay)]
        return float(sum(lengths))/len(lengths)

    @staticmethod
    def brown_freq_diff(in_essay):
        in_dist = nltk.FreqDist([w for w in tokenize(in_essay) if not w.startswith("@")])
        diffs = []
        for word in tokenize(in_essay):
            ratio = float(in_dist[word])/len(tokenize(in_essay))
            diffs.append(abs(ratio - brown_freq_ratios.get(word, 0)))
        avg = sum(diffs)/len(diffs)
        return avg

def tokenize(in_essay):
    if not hasattr(tokenize, 'cache'):
        tokenize.cache = {}
    if in_essay in tokenize.cache:
        return tokenize.cache[in_essay]
    out_tokens = tuple(i.lower() for i in ntlk_word_tokenize(in_essay.essay))
    out_tokens = tuple(i.lower() for i in in_essay.essay.split())
    tokenize.cache[in_essay] = out_tokens
    return out_tokens

def sent_tokenize(in_essay):
    if not hasattr(tokenize, 'cache'):
        tokenize.cache = {}
    if in_essay in tokenize.cache:
        return tokenize.cache[in_essay]
    out_tokens = ntlk_sent_tokenize(in_essay.essay)
    tokenize.cache[in_essay] = out_tokens
    return out_tokens

@filecache()
def parse(in_filename):
    in_file = open(in_filename, 'r')
    reader = csv.reader(in_file, delimiter='\t')
    essays = []
    for i, row in enumerate(reader):
        if i==0: continue
        this_essay = Essay(row)
        if this_essay.set_id != 2:
            essays.append(this_essay)
    in_file.close()
    return essays

if __name__ == '__main__':
    essays = parse("training/training_set_rel3.tsv")
    random.shuffle(essays)
    guesses = guess(essays, withold=50)
    pprint(guesses)
    diffs = tuple(abs(round(guess) - essay.normalized_rating()) for guess, essay in guesses)
    print "Average Difference:", sum(diffs)/len(diffs)
