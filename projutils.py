from tweetreader import TweetReader
import numpy as np

def load_project_data():
    data_set = _normalize_tweet_data([i for i in TweetReader("data/tweet_data.txt")][:10000])
    data_set = np.array(data_set)

    trX = data_set[:5000, :-1]
    trY = data_set[:5000, -1].reshape(5000,1)
    teX = data_set[5000:, :-1]
    teY = data_set[5000:, -1].reshape(5000,1)

    return (trX, trY, teX, teY)

def _normalize_tweet_data(data_set):
    high_time = float(max([i[0] for i in data_set]))
    high_hash = float(max([i[2] for i in data_set]))
    high_retw = float(max([i[4] for i in data_set]))

    for i in range(len(data_set)):
            data_set[i][0] /= high_time
            data_set[i][1] = (data_set[i][1] + 1)/2.0
            data_set[i][2] /= high_hash
            data_set[i][3] = float(data_set[i][3])
            data_set[i][4] /= high_retw
            data_set[i][5] /= 140.0 #max tweet len
            data_set[i][6] = float(data_set[i][6])

    return data_set
