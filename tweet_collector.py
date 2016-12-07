from twitutils import get_twitter_api
import os
import random
import tweepy
import sys
import json
from datetime import datetime
from textblob import TextBlob
import pytz
import numpy
import math

if len(sys.argv) < 3:
    print "Not enough arguments supplied. Format is \n$ python [script] [num] [outfile]"
    exit(1)

class StreamListenerImpl(tweepy.StreamListener):
    def __init__(self, collect_num, output):
        super(tweepy.StreamListener, self).__init__()
        self.collect_num = collect_num
        self.output = output

    def _process(self, data):
        """Gathers live time, connotation, number of hashtags, link presence,
        and tweet length from data"""

        # only get tweet if it's a retweet
        if "retweeted_status" not in data:
            return None

        data = data["retweeted_status"]

        if data["user"]["followers_count"] == 0:
            return None

        output = []
        text = data["text"]

        # append live time of tweet
        output.append(math.fabs((datetime.now() - datetime.strptime(data["created_at"],'%a %b %d %H:%M:%S +0000 %Y')).total_seconds()))

        # append polarity of text
        output.append(numpy.mean([s.sentiment.polarity for s in TextBlob(text).sentences]))

        # append # of hashtags
        output.append(len(data["entities"]["hashtags"]))

        # append whether or not link exists
        output.append(True if data["entities"]["urls"] else False)

        # append favorite count
        output.append(data["retweet_count"])

        # append tweet length
        output.append(len(text))

        # append user is verified
        output.append(data["user"]["verified"])

        # append RT/Follower ration
        output.append(float(data["favorite_count"])/float(data["user"]["followers_count"]))

        return output

    def on_data(self, data):
        marshalled = self._process(json.loads(data.replace('\n', '').strip()))

        if not marshalled:
            return True

        self.collect_num -= 1
        if self.collect_num < 0:
            return False
        self.output.write(json.dumps(marshalled) + '\n')
        return True

    def on_error(self, status):
        print status

tweets = []
output = open(sys.argv[2], 'w')
l = StreamListenerImpl(int(sys.argv[1]), output)
o = tweepy.OAuthHandler(os.environ.get("CONSUMER_KEY"),
                os.environ.get("CONSUMER_SEC_KEY"))
o.set_access_token(os.environ.get("ACCESS_TOKEN"),
                os.environ.get("ACCESS_TOKEN_SEC"))
s = tweepy.Stream(o, l)
s.sample(languages=['en'])

output.close()
