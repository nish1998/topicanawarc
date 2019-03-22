
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os
import re, spacy, gensim
# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import hashlib
import json
import logging
import os
import datetime


__author__ = "Felix Hamborg"
__copyright__ = "Copyright 2017"
__credits__ = ["Sebastian Nagel"]

############ CONFIG ############
# download dir for warc files
my_local_download_dir_warc = './cc_download_warc/'
# download dir for articles
my_local_download_dir_article = './cc_download_articles/'
# hosts (if None or empty list, any host is OK)
my_filter_valid_hosts = ['nytimes.com', 'www.nytimes.com', 'www.thehindu.com', 'thehindu.com', 'www.bbc.com', 'edition.cnn.com']  # example: ['elrancaguino.cl']
# start date (if None, any date is OK as start date), as datetime
my_filter_start_date = None  # datetime.datetime(2016, 1, 1)
# end date (if None, any date is OK as end date), as datetime
my_filter_end_date = None  # datetime.datetime(2016, 12, 31)
# if date filtering is strict and news-please could not detect the date of an article, the article will be discarded
my_filter_strict_date = True
# if True, the script checks whether a file has been downloaded already and uses that file instead of downloading
# again. Note that there is no check whether the file has been downloaded completely or is valid!
my_reuse_previously_downloaded_files = True
# continue after error
my_continue_after_error = True
# show the progress of downloading the WARC files
my_show_download_progress = True
# log_level
my_log_level = logging.INFO
# json export style
my_json_export_style = 1  # 0 (minimize), 1 (pretty)
# number of extraction processes
my_number_of_extraction_processes = 1
# if True, the WARC file will be deleted after all articles have been extracted from it
my_delete_warc_after_extraction = True
# if True, will continue extraction from the latest fully downloaded but not fully extracted WARC files and then
# crawling new WARC files. This assumes that the filter criteria have not been changed since the previous run!
my_continue_process = True
############ END YOUR CONFIG #########

#from sklearn.model_selection import GridSearchCV
#from pprint import pprint
# Plotting tools
# import pyLDAvis
# import pyLDAvis.sklearn
# import matplotlib.pyplot as plt

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])



def __setup__():
    """
    Setup
    :return:
    """
    if not os.path.exists(my_local_download_dir_article):
        os.makedirs(my_local_download_dir_article)


def __get_pretty_filepath(path, article):
    """
    Pretty might be an euphemism, but this function tries to avoid too long filenames, while keeping some structure.
    :param path:
    :param name:
    :return:
    """
    short_filename = hashlib.sha256(article.filename.encode()).hexdigest()
    sub_dir = article.source_domain
    final_path = path + sub_dir + '/'
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    return final_path + short_filename + '.json'

list_awesome = []

def on_valid_article_extracted(article):
    """
    This function will be invoked for each article that was extracted successfully from the archived data and that
    satisfies the filter criteria.
    :param article:
    :return:
    """
    print(article.text)
    if(article.text!=None):
        list_awesome.append(str(article.text))
    return
    # # do whatever you need to do with the article (e.g., save it to disk, store it in ElasticSearch, etc.)
    # with open(__get_pretty_filepath(my_local_download_dir_article, article), 'w') as outfile:
    #     if my_json_export_style == 0:
    #         json.dump(article.__dict__, outfile, default=str, separators=(',', ':'))
    #     elif my_json_export_style == 1:
    #         json.dump(article.__dict__, outfile, default=str, indent=4, sort_keys=True)
    #     # ...


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# Show top n keywords for each topic
def show_topics(vectorizer, lda_model, n_words=20):
    vectorizer=vectorizer
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        datax = request.get_json()
        
        dateis = str(datax["datepicker"])
        global url_half

        url_half = dateis[6:]+'/'+dateis[0:2]+'/CC-NEWS-'+dateis[6:]+dateis[0:2]+dateis[3:5]

        import commoncrawl_crawler as commoncrawl_crawler
        print(url_half)
        try:
            # df = pd.read_json('https://api.nytimes.com/svc/archive/v1/'+str(datax["news_year"])+'/'+str(datax["news_month"])+'.json?api-key=9LveXbUx48VQniWyMM5AYGaIGY9kgQSG')
            # df2= df['response'][0]
            # data=[]
            # for i in range(0,len(df2)):
            #     if 'lead_paragraph' in df2[i]:
            #         data.append(df2[i]['lead_paragraph'])
            __setup__()
            commoncrawl_crawler.crawl_from_commoncrawl(on_valid_article_extracted,
                                               valid_hosts=my_filter_valid_hosts,
                                               start_date=my_filter_start_date,
                                               end_date=my_filter_end_date,
                                               strict_date=my_filter_strict_date,
                                               reuse_previously_downloaded_files=my_reuse_previously_downloaded_files,
                                               local_download_dir_warc=my_local_download_dir_warc,
                                               continue_after_error=my_continue_after_error,
                                               show_download_progress=my_show_download_progress,
                                               number_of_extraction_processes=my_number_of_extraction_processes,
                                               log_level=my_log_level,
                                               delete_warc_after_extraction=True, 
                                               continue_process=True)
            data = list_awesome
            print(data)
            
            # Remove Emails
            data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

            # Remove new line characters
            data = [re.sub('\s+', ' ', sent) for sent in data]

            # Remove distracting single quotes
            data = [re.sub("\'", "", sent) for sent in data]

            data_words = list(sent_to_words(data))

            # nlp = spacy.load('en', disable=['parser', 'ner'])

            # Do lemmatization keeping only Noun, Adj, Verb, Adverb
            data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

            vectorizer = CountVectorizer(analyzer='word',       
                                        min_df=5,                        # minimum reqd occurences of a word 
                                        stop_words='english',             # remove stop words
                                        lowercase=True,                   # convert all words to lowercase
                                        token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                        # max_features=50000,             # max number of uniq words
                                        )

            data_vectorized = vectorizer.fit_transform(data_lemmatized)
            # Materialize the sparse data
            data_dense = data_vectorized.todense()
            lda_model = LatentDirichletAllocation(n_components=int(datax["news_topics"]),               # Number of topics
                                                max_iter=10,               # Max learning iterations
                                                learning_method='online',   
                                                random_state=100,          # Random state
                                                batch_size=128,            # n docs in each learning iter
                                                evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                                n_jobs = -1,               # Use all available CPUs
                                                )
            lda_output = lda_model.fit_transform(data_vectorized)
            topic_keywords = show_topics(vectorizer, lda_model, n_words=10)        

            # Topic - Keywords Dataframe
            df_topic_keywords = pd.DataFrame(topic_keywords)
            df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
            df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
            return df_topic_keywords.to_json()
        except ValueError:
            return "error"

        return "error"

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port)
    app.run()
