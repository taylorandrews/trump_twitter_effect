import tweepy
from tweepy.auth import OAuthHandler
# GET API KEYS FROM PARENT DIRECTORY
import sys, os
sys.path.append(os.path.abspath('../../'))
from capstone import secrets


auth = tweepy.OAuthHandler(secrets.twitter_api, secrets.twitter_api_secret)
auth.set_access_token(secrets.twitter_access_token, secrets.twitter_access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text
