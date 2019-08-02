
import sys
from math import log
from util import init_spark, init_spark_tagger, init_spark_tokenizer, init_spark_sentencizer, process_wiki_json
tokenize = lambda x: x.split()

COUNT_THRESHOLD = 2
NPMI_THRESHOLD = 0.5




spark, sc = init_spark()
tokenize = init_spark_tokenizer(sc)
tag_rus = init_spark_tagger(sc, 'rus')
sent_tokenize = init_spark_sentencizer(sc, 'rus')

# print(tag_rus(("пришел", "поздно")))

# sys.exit()

vocabulary_location = sys.argv[1]
corpus_location = sys.argv[2]

# voc = Vocabulary.load(vocabulary_location)
corpus = spark.sparkContext.textFile(corpus_location)
vocabulary_file = spark.sparkContext.textFile(vocabulary_location)

# broad_voc = spark.sparkContext.broadcast(voc)

# def calculate_npmi(bigram, total_bigrams, vocab):
#     voc = vocab.value
#     ids = voc.tokens2ids(bigram[0])
#     bigram_count = bigram[1]
#     total_words = voc.word_count_sum
#     freq_1 = voc.count[ids[0]]
#     freq_2 = voc.count[ids[1]]
#     pwmi = (log(bigram_count) - log(freq_1) - log(freq_2) + 2 * log(total_words) - log(total_bigrams)) / (log(total_bigrams) - log(bigram_count))
#     return pwmi * bigram_count


def valid_n_gram(bigram_count_npmi):

    allowed = {'ADJ', "NOUN", 'S'}

    bigram_count, npmi = bigram_count_npmi
    bigram, count = bigram_count

    if count < COUNT_THRESHOLD:
        return False

    def item_allowed(item):

        item_str, item_pos = item

        if item_pos in allowed :
            return True
        return False
    
    return item_allowed(bigram[0]) and item_allowed(bigram[1])    

def npmi(bigram_count, vocabulary, total_words, total_grams):
    voc = vocabulary.value
    # gram_freq = gram_frequency.value

    bigram, gram_freq = bigram_count

    p_x = voc[bigram[0]]
    p_y = voc[bigram[1]]
    p_x_y = gram_freq

    pmi = log(p_x_y / (p_x * p_y))

    npmi = pmi / (- log(p_x_y) + 1e-18)

    return npmi

    #     if npmi > threshold:
    #         # Allow only nouns and adjectives as stable phrases
    #         allowed = {'JJ', "NN", 'NNP', 'NNS'}
    #         tagged = pos_tag(bigram)
    #         if tagged[0][1] in allowed and tagged[1][1] in allowed:
    #             candidates[bigram] = npmi
    #             # candidates.add((bigram, npmi))
    #             # candidates.add(bigram)
    #             # print(bigram, npmi)
    # return candidates


# bigram_counts = corpus.map(process_wiki_json)\
#     .flatMap(lambda line: sent_tokenize(line))\
#     .map(lambda sent: tag_rus(tokenize(sent)))\
#     .flatMap(lambda tokens: [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])\
#     .map(lambda bigram: (bigram, 1))\
#     .reduceByKey(lambda x,y: x+y)


bigram_counts = corpus.map(process_wiki_json)\
    .map(tokenize)\
    .flatMap(lambda tokens: [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])\
    .map(lambda bigram: (bigram, 1))\
    .reduceByKey(lambda x,y: x+y)


def parse_vocab_entry(line):
    parts = line.split("\t")
    # return ((parts[0], parts[1]), int(parts[2]))
    return (parts[0], parts[1])


vocab = vocabulary_file.map(parse_vocab_entry)
vocabMap_bc = sc.broadcast(vocab.collectAsMap())
total_words = vocab.map(lambda x: x[1]).reduce(lambda x, y: x + y)


total_bigrams = bigram_counts.map(lambda x: x[1]).reduce(lambda x, y: x + y)
# bigramCounts_bc = sc.broadcast(bigram_counts.collectAsMap())


bigram_counts.sortBy(lambda x: x[1], False)\
    .saveAsTextFile("bigram_count.lenta")

def add_pos_tags(record):
    gram_count, npmi = record
    gram, count = gram_count

    gram_pos = tag_rus(gram)
    return ((gram_pos, count), npmi)


bigram_counts.map(lambda gram: (gram, npmi(gram, vocabMap_bc, total_words, total_bigrams)))\
    .filter(lambda gram: gram[1] > NPMI_THRESHOLD)\
    .map(lambda x: "%s_%s" % x[0])\
    .saveAsTextFile("bigram_filtered.lenta")


# total_bigrams = bigram_counts.map(lambda x: x[1]).reduce(lambda x, y: x + y)

# bigram_score = bigram_counts.map(lambda bg: (calculate_npmi(bg, total_bigrams, broad_voc), bg[0])).sortByKey()

# bigram_score.saveAsTextFile("bs.txt")