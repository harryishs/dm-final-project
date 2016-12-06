import tweepy
import operator

def get_twitter_api(cons_key, cons_sec_key, acc_token, acc_sec_token):
    auth = tweepy.OAuthHandler(cons_key, cons_sec_key)
    auth.set_access_token(acc_token, acc_sec_token)
    return tweepy.API(auth)

def get_all_tweets(screen_name, api):
    alltweets = []
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    while len(new_tweets) > 0:
	new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
	alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
    return alltweets

def twit_top_words(screen_name, api, n=20, l=5):
    tweets = get_all_tweets(screen_name, api)

    word_count = {}
    hashtags = {}
    for text in [tweet.text for tweet in tweets]:
        processed = text.lower().strip().split()
        for word in processed:
            if len(word) < l: continue
            if word.startswith(u'@'):continue
            if word.startswith(u'#'):continue
            word_count[word] = word_count.get(word, 0) + 1

    return sorted(word_count.items(),
            key=operator.itemgetter(1),
            reverse=True)[:n]
