import re                                  # library for regular expression operations
import string                              # for string operations
import numpy as np
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings


class Sentiment_Analysis:
    def __init__(self):
        self.freq={}
        self.ann=None

    def process_tweet(self,tweet):
        '''
        input:
            tweet: a tweet to be processed

        output: 
            cleaned_tweet: a list of tokenized words from the tweet after removing punctuations,stopword,applying stemming and tokenizing etc
        '''

        # instantiate tokenizer class
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)

        #Import the english stop words list from NLTK
        english_stopwords = stopwords.words('english') 
        # remove hyperlinks
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)

        # tokenize tweets
        tweet = tokenizer.tokenize(tweet)

        cleaned_tweet=[]
        for word in tweet:
            if word not in english_stopwords and word not in string.punctuation:
                cleaned_tweet.append(word)

        return cleaned_tweet


    def frequency_dic(self,tweets,label):
        '''
        input:
            tweets: a list of tweets
            label : a list of labels(0,1) of the corresponding tweets

        output:
            freq: a dictionary{key:word, value:np.array([positive_count,negative_count])}
        '''
        freq={}
        
        for tweet,y in zip(tweets,np.squeeze(label)):
            for word in self.process_tweet(tweet):
                
                freq[word]= freq.get(word,0)+np.array([1,0]) if y==1 else freq.get(word,0)+np.array([0,1])

        return freq


    def extract_feature(self,tweet,freq):
        '''
            input:
                tweet: a tweet
                freq: frequency matrix

            output:
                X: an array of shape(3,1), X=[bias=0,positive_count,negative_count]
        '''
        X=[0]*3
        for word in self.process_tweet(tweet):
            X[1] += freq.get(word,[0,0])[0]
            X[2] += freq.get(word,[0,0])[1]

        return np.array(X).reshape(3,1)
    
    def create_input_feature(self,tweets):
        '''
            input:
                tweets: a list of tweets
            output:
                return the input features of the tweets (x=[bias=0,pos_freq,neg_freq])
                 
        '''
        # add a column of zeros
        input_features=np.zeros((3,1))
        
        for tweet in tweets:
            input_features= np.append(input_features,self.extract_feature(tweet,self.freq),axis=1)
    
        #remove the column of zeros and return it
        return input_features[:,1:]
    
    def train(self,train_tweets,train_labels,test_tweets,test_label,batch_size,lr,iterations,ann):
        '''
            input:
                tweets: a list of tweets
                labels: label of the corresponding tweet (0/1)
                ann:      ann model
        '''
        self.ann=ann
        self.freq=self.frequency_dic(train_tweets,train_labels)
        

        train_features= self.create_input_feature(train_tweets)*1e-3
        test_features= self.create_input_feature(test_tweets)*1e-3
        
 
        self.ann.train(train_features,train_labels,batch_size,test_features,test_label,lr,iterations,[True,True])
        

    def predict(self,tweets,labels=0):
        '''
            input:
                tweets: a list of tweet
                labels: labels of the tweets (0/1)
                NOTE: The label parameter is not necessary , while predicting custom tweets. It is only usefull when we have the label
                for the tweets and we want to know the accuracy of the prediction. By default it is set to 0
                
            output:
                returns  the accuracy and actual output
        '''
        if labels==0:
            labels= np.zeros((1,len(tweets)))
        return self.ann.predict(self.create_input_feature(tweets)*1e-3,labels,batch_size=100,train=False)
        
        
        
        
        
        


