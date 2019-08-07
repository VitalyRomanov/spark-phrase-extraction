
import sys
from math import log
import argparse
import random
from functools import reduce
from util import init_spark, init_spark_tagger, init_spark_tokenizer, init_spark_sentencizer, process_wiki_json
tokenize = lambda x: x.split()

COUNT_THRESHOLD = 2
NPMI_THRESHOLD = 0.75

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('vocab', type=str,
                    help='Path to vocabulary')
parser.add_argument('corpus',type=str,
                    help='Path to corpus')
parser.add_argument('output',type=str,
                    help='Path to output')
parser.add_argument('ngram',type=int,
                    help='Length of ngram')
parser.add_argument('subsample',type=float,
                    help='Subsample amount')

args = parser.parse_args()


spark, sc = init_spark()
tokenize = init_spark_tokenizer(sc)
tag_rus = init_spark_tagger(sc, 'rus')
sent_tokenize = init_spark_sentencizer(sc, 'rus')

vocabulary_location = args.vocab
corpus_location = args.corpus
ngram_len = args.ngram
subsample_factor = 1. - args.subsample


corpus = sc.textFile(corpus_location)
vocabulary_file = sc.textFile(vocabulary_location)


def valid_n_gram(ngram_count_npmi):

    allowed = {'ADJ', "NOUN", 'S'}

    ngram_count, npmi = ngram_count_npmi
    ngram, count = ngram_count

    if count < COUNT_THRESHOLD:
        return False

    def item_allowed(item):

        item_str, item_pos = item

        if item_pos in allowed :
            return True
        return False
    
    return all(item_allowed(gram) for gram in ngram)# item_allowed(bigram[0]) and item_allowed(bigram[1])    

def npmi(ngram_count, vocabulary, total_words, total_grams):
    voc = vocabulary.value
    # gram_freq = gram_frequency.value

    ngram, gram_freq = ngram_count

    p_ = reduce(lambda x, y: x*y, [voc[gram]/total_words for gram in ngram])

    p_x_y = gram_freq / total_grams

    pmi = log(p_x_y / p_)

    npmi = pmi / (- log(p_x_y) + 1e-18)

    return npmi / (ngram_len - 1)


def split_in_grams(tokens):
    return [tuple(tokens[i:i+ngram_len]) for i in range(len(tokens) - (ngram_len - 1))]


def subsample(smth):
    return random.random() < subsample_factor


ngram_counts_sorted = corpus.map(process_wiki_json)\
    .map(tokenize)\
    .flatMap(split_in_grams)\
    .filter(subsample)\
    .map(lambda ngram: (ngram, 1))\
    .reduceByKey(lambda x,y: x+y)\
    .filter(lambda x: x[1] > COUNT_THRESHOLD)\
    .sortBy(lambda x: x[1], False)


def parse_vocab_entry(line):
    parts = line.split("\t")
    # return ((parts[0], parts[1]), int(parts[2]))
    return (parts[0], int(parts[1]))


vocab = vocabulary_file.map(parse_vocab_entry)
vocabMap_bc = sc.broadcast(vocab.collectAsMap())
total_words = vocab.map(lambda x: x[1]).reduce(lambda x, y: x + y)


total_ngrams = ngram_counts_sorted.map(lambda x: x[1]).reduce(lambda x, y: x + y)
ngram_norm = ngram_counts_sorted.map(lambda x: x[1]).reduce(lambda x, y: max(x,y))

ngram_counts_sorted\
    .map(lambda x: "%s\t%d" % ("_".join(x[0]), x[1]))\
    .saveAsTextFile("%s.%dgram_count" % (args.output, ngram_len))
    #

ngram_counts_sorted\
    .map(lambda x: "%s\t%.4f" % ("_".join(x[0]), x[1] / ngram_norm))\
    .saveAsTextFile("%s.%dgram_count_normalized" % (args.output, ngram_len))

def add_pos_tags(record):
    gram_count, npmi = record
    gram, count = gram_count

    gram_pos = tag_rus(gram)
    return ((gram_pos, count), npmi)


scores = ngram_counts_sorted\
    .map(lambda gram: (gram, npmi(gram, vocabMap_bc, total_words, total_ngrams)))\
    
scores.sortBy(lambda x: x[1], False)\
    .map(lambda x: "%s\t%d\t%f" % ("_".join(x[0][0]), x[0][1], x[1]))\
    .saveAsTextFile("%s.%dgram_scores" % (args.output, ngram_len))
    #.filter(lambda gram: gram[1] > NPMI_THRESHOLD)\

scores.sortBy(lambda x: x[0][1] * x[1], False)\
    .map(lambda x: "%s\t%f" % ("_".join(x[0][0]), x[0][1] / ngram_norm * x[1]))\
    .saveAsTextFile("%s.%dgram_countscore" % (args.output, ngram_len))




