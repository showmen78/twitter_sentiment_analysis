import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt            # library for visualization
import random  
import numpy as np

# downloads sample twitter dataset.
nltk.download('twitter_samples')

# download the stopwords from NLTK
nltk.download('stopwords')

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


#creating training and test set
tweet= all_positive_tweets+all_negative_tweets
# make a numpy array representing labels of the tweets
labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

train_tweets= tweet[:4000]+tweet[5000:9000]
test_tweets= tweet[4000:5000]+tweet[9000:10000]

train_label= np.array(list(labels[:4000])+list(labels[5000:9000]))
test_label=np.array(list(labels[4000:5000])+list(labels[9000:10000]))

train_label= train_label.reshape(1,train_label.size)
test_label= test_label.reshape(1,test_label.size)


from Ann import Ann
from sentiment_analysis import Sentiment_Analysis
#setting the seed , to reproduce the result
random.seed(42) 
ann = Ann([10,1],['sigmoid','sigmoid'],_type='binary_classification')
s=Sentiment_Analysis()
#training the model 
s.train(train_tweets,train_label,test_tweets,test_label,batch_size=20,lr=.1,iterations=2,ann=ann)


#prediction for custom tweets. If the vlaue is more that 0.5 then the tweet is positive , otherwise negative
# p= s.predict(['god help him '])
# print(p[1])