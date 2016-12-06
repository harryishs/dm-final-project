from twitutils import get_twitter_api
import os
import random
import tweepy
import sys

if len(sys.argv) < 3:
    print "Not enough arguments supplied. Format is \n$ python [script] [num] [outfile]"
    exit(1)

class StreamListenerImpl(tweepy.StreamListener):
    def __init__(self, collect_num, output):
        super(tweepy.StreamListener, self).__init__()
        self.collect_num = collect_num
        self.output = output

    def on_data(self, data):
        self.collect_num -= 1
        if self.collect_num < 0:
            return False
        self.output.write(data.replace('\n', '').strip())
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
