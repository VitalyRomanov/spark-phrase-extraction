import pyspark
# from Vocabulary import Vocabulary
from nltk import RegexpTokenizer
from nltk.data import load
import json

def init_spark():
    spark = (
            pyspark.sql.SparkSession.builder
            # This doesn't seem to have an impact on YARN.
            # Use `spark-submit --name` instead.
            # .appName('Sample Spark Application')
            .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')

    sc = spark.sparkContext
    return spark, sc


def init_spark_tokenizer(sparkContext):
    bc_tokenizer = sparkContext.broadcast(RegexpTokenizer('[A-Za-zА-Яа-я-]+|[^\w\s]'))

    tokenize = lambda text: bc_tokenizer.value.tokenize(text)
    return tokenize


def init_spark_tagger(sparkContext, lang):
    # valid languages eng rus
    from nltk.tag import _get_tagger, _pos_tag
    tagger = _get_tagger(lang)

    bc_tagger = sparkContext.broadcast(tagger)
    
    # return lambda tokens: _pos_tag(tokens, None, bc_tagger.value, lang) 
    return lambda tokens: _pos_tag(tokens, "universal", bc_tagger.value, lang)


def init_spark_sentencizer(sparkContext, lang):
    if lang == 'rus':
        language = 'russian'
    elif lang == 'eng':
        language = 'english'
    else:
        raise NotImplementedError()

    tokenizer = load("tokenizers/punkt/{0}.pickle".format(language))

    bc_tokenizer = sparkContext.broadcast(tokenizer)

    return lambda line: bc_tokenizer.value.tokenize(line)


def process_wiki_json(line):
    if line and line[0] == '{':
        return json.loads(line)['text']
    else:
        return line