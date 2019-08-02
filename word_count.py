import pyspark
import sys
from nltk import RegexpTokenizer
import json
from util import init_spark, init_spark_tagger, init_spark_tokenizer, init_spark_sentencizer, process_wiki_json
from Word import Word

spark, sc = init_spark()
tokenize = init_spark_tokenizer(sc)
tag_rus = init_spark_tagger(sc, 'rus')
sent_tokenize = init_spark_sentencizer(sc, 'rus')


corpus_location = sys.argv[1] 
output_location = sys.argv[2]


corpus = sc.textFile(corpus_location)

# vocabulary = corpus.map(process_wiki_json)\
#     .flatMap(lambda line: sent_tokenize(line))\    
#     # .flatMap(lambda sent: tag_rus(tokenize(sent)))\
#     .map(lambda token: (token, 1))\
#     .reduceByKey(lambda x, y: x + y)\
#     .map(lambda x: (x[1], x[0]))\
#     .sortByKey(False)\
#     .map(lambda x: "%s %s %d" % (x[1][0], x[1][1], x[0]))


vocabulary = corpus.map(process_wiki_json)\
    .flatMap(tokenize)\
    .map(lambda x: (x, 1))\
    .reduceByKey(lambda x, y: x + y)\
    .sortBy(lambda x: x[1], False)\
    .map(lambda x: "%s\t%d" % (x[0], x[1]))
    # .map(lambda x: "%s\t%s\t%s" % (x[0], x[1], tag_rus([x[0]])[0][1]))


vocabulary.saveAsTextFile(output_location)