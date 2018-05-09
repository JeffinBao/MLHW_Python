# coding: utf-8
import json
import string
import numpy as np
import sys
import math

class KmeansAlgoTweetCluster:

    def __init__(self, k, id_tweet_dic, id_list, centroids):
        self.k = k
        self.id_tweet_dic = id_tweet_dic
        self.id_list = id_list
        self.centroids = centroids
        self.dist_arr_size = len(id_list)

    def __cal_tweets_jaccard_dist(self, tweet1, tweet2):
        tweet1_word_list = tweet1.split(' ')
        tweet2_word_list = tweet2.split(' ')

        conjunction_size = 0 # number of same words in both tweets
        union_size = 0 # number of words in union
        for word in tweet1_word_list:
            if word in tweet2_word_list:
                conjunction_size += 1

        union_size = len(tweet1_word_list) + len(tweet2_word_list) - conjunction_size

        # for word in tweet2_word_list:
        #     if word not in tweet1_word_list:
        #         union_size += 1

        return 1 - (conjunction_size / union_size)

    def __cal_tweets_dist_arr(self):
        dist_arr = np.zeros((self.dist_arr_size, self.dist_arr_size))
        for i in range(0, self.dist_arr_size):
            tweet_i = self.id_tweet_dic[self.id_list[i]]
            for j in range(i + 1, self.dist_arr_size):
                tweet_j = self.id_tweet_dic[self.id_list[j]]
                jaccard_dist = self.__cal_tweets_jaccard_dist(tweet_i, tweet_j)
                dist_arr[i, j] = jaccard_dist
                dist_arr[j, i] = jaccard_dist

        return dist_arr

    # kmeans algorithm
    def kmeans(self):
        self.dist_arr = self.__cal_tweets_dist_arr()
        clusters = {}

        while True:
            # assign different tweets to different clusters
            for i in range(0, self.dist_arr_size):
                min_dist_to_centroid = sys.maxsize
                cluster_idx = -1
                for centroid in self.centroids:
                    centroid_idx = self.id_list.index(centroid)
                    if self.dist_arr[i, centroid_idx] < min_dist_to_centroid:
                        min_dist_to_centroid = self.dist_arr[i, centroid_idx]
                        cluster_idx = self.centroids.index(centroid)

                cluster = clusters.get(cluster_idx)
                if cluster is None:
                    cluster = []

                cluster.append(self.id_list[i])
                clusters[cluster_idx] = cluster

            # calculate new centroid
            new_centroids = []
            for i in range(0, self.k):
                cluster = clusters[i]
                centroid_id = 0
                min_dist = sys.maxsize
                for tweet_id in cluster:
                    dist_sum = 0
                    tweet_id_idx = self.id_list.index(tweet_id)
                    for tweet_id_other in cluster:
                        tweet_id_other_idx = self.id_list.index(tweet_id_other)
                        dist_sum += self.dist_arr[tweet_id_idx, tweet_id_other_idx]
                    if dist_sum < min_dist:
                        min_dist = dist_sum
                        centroid_id = tweet_id

                new_centroids.append(centroid_id)

            # check whether new_centroids equals to previous centroids
            common_centroid_size = 0
            for new_centroid in new_centroids:
                if new_centroid not in self.centroids:
                    self.centroids = new_centroids
                    break
                else:
                    common_centroid_size += 1

            # if centroids don't change anymore jump out of the while true loop
            if common_centroid_size == self.k:
                break

        return clusters

    # calculate sse(sum of squared error)
    def __cal_sse(self, clusters):
        sse = 0.0
        for i in range(0, self.k):
            centroid = self.centroids[i]
            cluster = clusters[i]
            for tweet_id in cluster:
                centroid_idx = self.id_list.index(centroid)
                tweet_id_idx = self.id_list.index(tweet_id)
                sse += math.pow(self.dist_arr[centroid_idx, tweet_id_idx], 2)

        return sse


    def write_into_file(self, clusters):
        sse = self.__cal_sse(clusters)
        with open('tweets-k-means-output.txt', 'a') as output_file:
            for i in range(0, self.k):
                output_file.write(str(i) + ' ')
                cluster = clusters[i]
                for tweet_id in cluster:
                    output_file.write(str(tweet_id) + ', ')
                output_file.write('\n')
            output_file.write('sse is:' + str(sse))

if __name__ == '__main__':
    data_file = open('Tweets.json', 'r')
    tweets = []
    for line in data_file:
        tweets.append(json.loads(line))

    id_tweet_dic = {}
    id_list = []
    table = str.maketrans({key: None for key in string.punctuation})
    for tweet in tweets:
        tweet['text'] = tweet['text'].translate(table)  # remove punctuation
        tweet['text'] = tweet['text'].replace('\n', ' ')  # remove \n
        id_tweet_dic[tweet['id']] = tweet['text']
        id_list.append(tweet['id'])

    k = 25  # default k value is 25
    centroids = []
    centroid_file = open('InitialSeeds.txt', 'r')
    for line in centroid_file:
        line = line.replace(',', '')
        line = line.replace('\n', '')
        centroids.append(int(line))

    tweet_cluster_kmeans = KmeansAlgoTweetCluster(k, id_tweet_dic, id_list, centroids)
    clusters = tweet_cluster_kmeans.kmeans()
    tweet_cluster_kmeans.write_into_file(clusters)