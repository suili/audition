import re
import os
from datetime import datetime
from collections import defaultdict, Counter
import math
import pickle

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from text2vec import SentenceModel, EncoderType
import paddlehub as hub

class TextAnalyzer:
    suggestion_pattern = re.compile('|'.join(["建议", "能不能", "可(?!能)", "更", "最好", "有点", "适当", "应该", "需要", "如果", "若能", "一下"]))
    sentence_delimiter_pattern = re.compile('|'.join([r"\n", r"\\n", r"\d\.", r"\d\)", r"\.", r"\?", "。", "！", "？", "；", "!", ";", "但是", "但", "不过", "另外", "然而", "并", r"(?=建议)"]))

    # do the K-Means clustering with sentiment analysis
    def __init__(self, document, analysis_path, n_clusters=30):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        kmeans_model_file = analysis_path + f"sentences.txt_clustering_{n_clusters}_model.pkl"
        cluster_names_file = analysis_path + f"sentences.txt_clustering_{n_clusters}_names.txt"
        # if K-Means resutls exist, just load it
        if os.path.exists(kmeans_model_file) and os.path.exists(cluster_names_file):
            self.cosent = SentenceModel("shibing624/text2vec-base-chinese", encoder_type=EncoderType.FIRST_LAST_AVG)
            with open(kmeans_model_file, 'rb') as f:
                self.kmeans = pickle.load(f)
            with open(cluster_names_file, 'r') as f:
                self.cluster_names = [line.strip() for line in f]
            self.ernie = hub.Module(name="ernie_skep_sentiment_analysis")
            self.jieba = hub.Module(name="jieba_paddle")
            self.n_clusters = n_clusters
        else:
            # STEP 1: cut a long text into small sentences and write to file (for debugging)
            print(datetime.now(), f"cutting sentences")
            sents = self.cut_sentences(document)
            print(datetime.now(), f"total sentences: {len(sents)}")
            sentence_file = analysis_path + 'sentences.txt'
            print(datetime.now(), f"writing sentences to file {sentence_file}")
            with open(sentence_file, 'w') as f:
                for sent in sents:
                    f.write(f"{sent}\n")

            # STEP 2: Sentence vecter representation algorithm by CoSENT
            print(datetime.now(), f"getting sentence embeddings through cosent")
            cosent = SentenceModel("shibing624/text2vec-base-chinese", encoder_type=EncoderType.FIRST_LAST_AVG)
            X = cosent.encode(sents)
            length = np.sqrt((X**2).sum(axis=1))[:,None]
            X = X / length

            # STEP 3: sentiment ananlysis using Baidu's model
            print(datetime.now(), f"doing sentiment analysis...")
            ernie = hub.Module(name="ernie_skep_sentiment_analysis")
            results = ernie.predict_sentiment(sents, use_gpu=True)
            sentiments = [result['positive_probs'] for result in results]

            # STEP 4: K-Means clustering using sklearn
            print(datetime.now(), f"doing kmeans clustering, k={n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
            X_cluster_ids = kmeans.labels_
            with open(kmeans_model_file, 'wb') as f:
                pickle.dump(kmeans, f)

            # STEP 5: Use top-3 TF-IDF words as cluster labels
            print(datetime.now(), f"preparing word freq data")
            with open('misc/hit_stopwords.txt') as f:
                stop_words = {w.strip() for w in f.readlines()}
            stop_words.update([''])
            jieba = hub.Module(name="jieba_paddle")
            cluster_data = defaultdict(list)
            df = defaultdict(int)
            for i in range(len(sents)):
                words = jieba.cut(sents[i], cut_all=False, HMM=True)
                words = [w for w in words if w not in stop_words]
                cluster_data[X_cluster_ids[i]].extend(words)
                for w in words:
                    df[w] += 1
            cluster_names = []
            for i in range(n_clusters):
                words = cluster_data[i]
                wc_i = Counter(words)
                top3_items = sorted(Counter(words).items(), key=lambda x:-x[1]*math.log(len(sents)/df[x[0]]))[:3]
                cluster_names.append(",".join([x[0] for x in top3_items]))
            with open(cluster_names_file, 'w') as f:
                for name in cluster_names:
                    f.write(name + '\n')

            # STEP 6: Write clustering results    
            clustering_file = analysis_path + f"sentences.txt_clustering_{n_clusters}.txt"
            print(datetime.now(), f"writing kmeans results to file {clustering_file}")
            with open(clustering_file, 'w') as f:
                for cluster_id, sentiment, sent in sorted(zip(X_cluster_ids, sentiments, sents)):
                    f.write(f"{cluster_id}\t{cluster_names[cluster_id]}\t{sentiment}\t{sent}\n")

            # Set object variables (mostly the external libraries)
            self.cosent = cosent
            self.ernie = ernie
            self.jieba = jieba
            self.kmeans = kmeans
            self.n_clusters = n_clusters
            self.cluster_names = cluster_names

    # cut a long sentence into short sentences (manual pattern)
    def cut_sentences(self, text):
        sents = self.sentence_delimiter_pattern.split(text)
        sents = [s.strip() for s in sents]
        sents = list(set([s for s in sents if len(s) > 2]))
        return sents

    # given a paragraph of text, cut into small sentences 
    # and return (cluster_ids, sentiments, sentences)
    def cut_and_analysis(self, text):
        sents = self.cut_sentences(text)
        if len(sents) == 0:
            return np.array([]), np.array([]), []
        X = self.cosent.encode(sents)
        length = np.sqrt((X**2).sum(axis=1))[:,None]
        X = X / length
        cluster_ids = self.kmeans.predict(X)
        results = self.ernie.predict_sentiment(sents, use_gpu=True)
        sentiments = np.array([result['positive_probs'] for result in results])
        return cluster_ids, sentiments, sents

    # for a input form, compute the weighted positive/suggestion sentence length
    def pos_neg_stat(self, form):
        # part 1: highlights/suggestion in steps (weight 0.5)
        highlights = form.highlights()
        suggestions = form.suggestions()
        pos_len, neg_len = len(highlights) // 2, len(suggestions) // 2
        # part 2: comments
        cluster_ids, sentiments, sents = self.cut_and_analysis(form.comment)
        for cluster_id, sentiment, sent in zip(cluster_ids, sentiments, sents):
            if sentiment < 0 or TextAnalyzer.suggestion_pattern.search(sent) is not None:
                neg_len += len(sent)
            else:
                pos_len += len(sent)
        return pos_len, neg_len
            
    #for a long text, return cluster distribution (30 dim) and sentiment in each cluster (30 dim)
    def cluster_and_sentiment_distribution(self, text):
        cluster_dist = np.zeros(self.n_clusters)
        sentiment_dist = np.zeros(self.n_clusters)
        total_len = 0
        cluster_ids, sentiments, sents = self.cut_and_analysis(text)
        for cluster_id, sentiment, sent in zip(cluster_ids, sentiments, sents):
            cluster_dist[cluster_id] += len(sent)
            sentiment_dist[cluster_id] += len(sent) * (2 * sentiment - 1)  # range [0,1] to [-1,1]
            total_len += len(sents)
        if total_len > 0:
            sentiment_dist /= total_len
        return cluster_dist, sentiment_dist

    # direct sentence representation
    def get_vectors(self, texts):
        return self.cosent.encode(texts)

    # direct sentiment score
    def get_sentiment_score(self, text):
        #sents = self.cut_sentences(text)
        sents = [text]
        if len(sents) == 0:
            return 0
        results = self.ernie.predict_sentiment(sents, use_gpu=True)
        sentiments = [result['positive_probs'] for result in results]
        return sum([sentiment * len(sent) for sentiment, sent in zip(sentiments, sents)]) / sum([len(sent) for sent in sents])


    # dimensional reduction into 2d vector
    def visualization(self, X):
        tsne = TSNE(n_components=2, init='random')
        return tsne.fit_transform(X)



