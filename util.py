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
    conf = pyspark.SparkConf()\
        .setAll([('spark.local.dir', './spark_temp')])#, ('spark.driver.memory', '3g'), ('spark.executor.memory', '3g')])
    sc.stop()
    sc = pyspark.SparkContext(conf=conf)
    return spark, sc


def init_spark_tokenizer(sparkContext):
    bc_tokenizer = sparkContext.broadcast(RegexpTokenizer('[A-Za-zА-Яа-яёЁar0-9-]+|[^\w\s]'))

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


def replace_accents_rus(sent_orig):
    sent = sent_orig.replace("о́", "о")
    sent = sent.replace("а́", "а")
    sent = sent.replace("е́", "е")
    sent = sent.replace("у́", "у")
    sent = sent.replace("и́", "и")
    sent = sent.replace("ы́", "ы")
    sent = sent.replace("э́", "э")
    sent = sent.replace("ю́", "ю")
    sent = sent.replace("я́", "я")
    sent = sent.replace("о̀", "о")
    sent = sent.replace("а̀", "а")
    sent = sent.replace("ѐ", "е")
    sent = sent.replace("у̀", "у")
    sent = sent.replace("ѝ", "и")
    sent = sent.replace("ы̀", "ы")
    sent = sent.replace("э̀", "э")
    sent = sent.replace("ю̀", "ю")
    sent = sent.replace("я̀", "я")
    sent = sent.replace(b"\u0301".decode('utf8'), "")
    sent = sent.replace(b"\u00AD".decode('utf8'), "")
    sent = sent.replace(b"\u00A0".decode('utf8'), " ")
    sent = sent.replace(" ", " ")
    return sent


def process_wiki_json(line):
    try:
        return replace_accents_rus(json.loads(line)['text'])
    except:
        return replace_accents_rus(line)

