
# Sentiment Analysis using Ann
Sentiment analysis, also known as opinion mining, is a powerful technique that allows us to gain insights into public opinion and emotions expressed in text data. With the rise of social media platforms, such as Twitter, sentiment analysis has become an essential tool for understanding the sentiments and attitudes of individuals towards various topics.

In this project, I tried to leverage the capabilities of the Natural Language Toolkit (NLTK) library and Artificial Neural Networks (ANN) to perform sentiment analysis on a Twitter dataset. Twitter, being a popular microblogging platform, provides a rich source of real-time data containing diverse opinions and sentiments expressed by users across the globe.

The primary objective of this project is to develop a robust sentiment analysis model capable of accurately classifying tweets into positive and negative sentiment categories. By utilizing the power of ANN, we aim to build a predictive model that can effectively learn patterns and relationships in the textual data, enabling it to make accurate predictions on unseen tweets.

To achieve this goal, we first preprocess the Twitter dataset by removing noise, such as special characters, hashtags, and URLs. We then employ NLTK's extensive collection of text processing functionalities, including tokenization, stemming, and stop-word removal, to transform the raw textual data into a structured format suitable for analysis.

Next, we design and train an ANN model using the preprocessed dataset. Artificial Neural Networks are widely recognized for their ability to learn complex patterns in data and are well-suited for sentiment analysis tasks. By leveraging a combination of input, hidden, and output layers, the ANN model can effectively capture the underlying sentiment dynamics present in the tweet dataset.

Once the model is trained, we evaluate its performance using accuracy metric. This evaluation allows us to gauge the effectiveness of our sentiment analysis model in correctly predicting sentiment labels for tweets.


Overall, this project demonstrates the power of combining natural language processing techniques from NLTK with the learning capabilities of ANN to perform sentiment analysis on Twitter data. 


# Table of Content
- [Importing Libraries]()
- [Downloading twitter data]()
- [Visualizing the Data]()
- [Sentiment Analysis Class]()
- [Train and Test data preparation]()
- [Training the Ann model and checking Accuracy]()
- [Making Prediction]()


# Importing Libraries
#### Library Used 
#### ``` nltk , matplotlib , random , re , string , numpy ```
```python
import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt            # library for visualization
import random  
import re                                  # library for regular expression operations
import string                              # for string operations
import numpy as np
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

```

# Downloading Twitter Data
#### A coupus of tweets were downloaded from NLTK. There are 10000 samples of tweets. Among them 5000 are positive tweets and 5000 are negative tweets. After downloaded tweets were divided into positive and negative tweets

# Visualizing the Data
Then a pie chart was drawn . The pie diagram indicates that 50% of the tweets are positive and other 50% are negative.


# Sentiment Analysis Class

The `Sentiment_Analysis` class is a Python class that provides methods for sentiment analysis using a Twitter dataset. It leverages the Natural Language Toolkit (NLTK) library and an Artificial Neural Network (ANN) model to perform sentiment analysis and predict whether a given tweet expresses positive or negative sentiment. To know more about the Ann model visit [github project](https://github.com/showmen78/Artificial_Neural_Network_from_Scratch/blob/main/Ann.py) or [kaggle notebook](https://www.kaggle.com/code/smndey/artificial-neural-network)

## Methods

### 1. `process_tweet(tweet)`

This method takes a single tweet as input and performs various preprocessing steps to remove noise from the text. It removes punctuation, stop words, hyperlinks, and other unnecessary elements. Then, it tokenizes the tweet by splitting it into individual words and returns a cleaned tweet.

### 2. `frequency_dic(tweet, label)`

This method takes a list of tweets as input along with their corresponding labels (0 for negative sentiment, 1 for positive sentiment). It creates a dictionary called `freq`, where each word is a key, and the value is a NumPy array representing the occurrence of that word in positive and negative tweets. The array has two elements: the count of occurrences in positive tweets and the count of occurrences in negative tweets. This method returns the `freq` dictionary.

### 3. `extract_feature(tweet, freq)`

Given a single tweet and the `freq` dictionary as input, this method preprocesses the tweet using the `process_tweet()` method. It then utilizes the `freq` dictionary to create a feature matrix for the tweet of shape (1, 3). The elements in this matrix are as follows: [bias = 0, total positive occurrences, total negative occurrences]. The method returns this feature matrix.

### 4. `create_input_feature(tweets)`

This method takes a list of tweets as input and uses the `extract_feature()` method to create input features for each tweet. It generates the input feature matrix for both training and test samples and returns the resulting input features.

### 5. `train(train_tweets, train_labels, test_tweets, test_labels, batch_size, lr, iterations, ann)`

The `train()` method is responsible for training the sentiment analysis model. It takes the training tweets, training labels, test tweets, test labels, batch size, learning rate (lr), iterations, and the ANN model as inputs. It calls the `train()` method of the ANN model and trains the actual sentiment analysis model using the provided parameters.

### 6. `predict(tweets)`

After training the model, the `predict()` method is used to predict the sentiment (positive or negative) of a list of tweets. It takes the list of tweets as input and returns a list of predictions, indicating whether each tweet expresses positive or negative sentiment.

# Train and Test data preparation
8000 tweets are taken as input data , 4000 tweets from each positive and negative tweets. And 2000 tweets are taken as test data , 1000 from each.


# Training The Ann model and Checking The Accuracy
The ann model is trained using the train data and the test data are also given to get the accuracy on the test data.

```python
from Ann import Ann
from sentiment_analysis import Sentiment_Analysis

#setting the seed , to reproduce the result
random.seed(42) 

ann = Ann([10,1],['sigmoid','sigmoid'],_type='binary_classification')
s=Sentiment_Analysis()
s.train(train_tweets,train_label,test_tweets,test_label,batch_size=20,lr=.1,iterations=20,ann=ann)
```
```
Parameter:

lr= .1
batch_size=20
iterations=20
neurons: [10,1] -> 10 in the first layer and 1 in the second.
activation: sigmoid in both layer
After training :
training_accuracy:99.25% and test accuracy:99.35%
```
```
iter:0 - train_cost:30.48 - train_acc:50.0 - test_acc:50.0
iter:1 - train_cost:26.89 - train_acc:87.06 - test_acc:85.7
iter:2 - train_cost:19.06 - train_acc:97.54 - test_acc:96.8
iter:3 - train_cost:15.7 - train_acc:98.12 - test_acc:97.8
iter:4 - train_cost:13.73 - train_acc:98.24 - test_acc:98.0
iter:5 - train_cost:12.4 - train_acc:98.26 - test_acc:98.0
iter:6 - train_cost:11.43 - train_acc:98.34 - test_acc:98.0
iter:7 - train_cost:10.69 - train_acc:98.38 - test_acc:98.05
iter:8 - train_cost:10.12 - train_acc:98.41 - test_acc:98.1
iter:9 - train_cost:9.66 - train_acc:98.5 - test_acc:98.2
iter:10 - train_cost:9.29 - train_acc:98.6 - test_acc:98.35
iter:11 - train_cost:8.99 - train_acc:98.66 - test_acc:98.5
iter:12 - train_cost:8.73 - train_acc:98.91 - test_acc:98.8
iter:13 - train_cost:8.51 - train_acc:99.02 - test_acc:99.15
iter:14 - train_cost:8.33 - train_acc:99.09 - test_acc:99.2
iter:15 - train_cost:8.17 - train_acc:99.12 - test_acc:99.25
iter:16 - train_cost:8.03 - train_acc:99.14 - test_acc:99.25
iter:17 - train_cost:7.91 - train_acc:99.18 - test_acc:99.3
iter:18 - train_cost:7.8 - train_acc:99.24 - test_acc:99.3
iter:19 - train_cost:7.7 - train_acc:99.25 - test_acc:99.35
```


# Making Predictions
Now using the `predict` method from `Sentiment_Analysis` class any tweet can be classfied into positive or negative

```python
p= s.predict(['god help him '])
print(p[1])
```

# Getting Started
1.Clone the repository:
```
git clone https://github.com/showmen78/twitter_sentiment_analysis.git
```

2.Install the required dependencies. You can use `pip `to install them:

```python
pip install numpy
pip install matplotlib
pip install nltk

```
3.Run the `main.py` file:

```python
python main.py
```

# Usage
To utilize the Sentiment_Analysis class for sentiment analysis using a Twitter dataset, follow these steps:

1.Import Libraries
``` python
import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt            # library for visualization
import random  

import re                                  # library for regular expression operations
import string                              # for string operations
import numpy as np
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
```
2.Download Twitter dataset
```python
# downloads sample twitter dataset.
nltk.download('twitter_samples')

# download the stopwords from NLTK
nltk.download('stopwords')

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
```

3. Training and Test data preparation
```python
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
```

4.Training the model and checking accuracy
```python
from Ann import Ann
from sentiment_analysis import Sentiment_Analysis

#setting the seed , to reproduce the result
random.seed(42) 

ann = Ann([10,1],['sigmoid','sigmoid'],_type='binary_classification')
s=Sentiment_Analysis()
s.train(train_tweets,train_label,test_tweets,test_label,batch_size=20,lr=.1,iterations=20,ann=ann)

```

5. Make prediction 
```python
p= s.predict(['god help him '])
print(p[1])

```


## Acknowledgement

I would like to express my gratitude to the following individuals and resources that have contributed to the development and success of this sentiment analysis project:

- The developers and contributors of the NLTK library for providing a comprehensive set of natural language processing tools and resources.

- The Twitter platform for providing access to real-time and diverse textual data, enabling us to analyze public sentiment on various topics.


## Contributions

I appreciate any form of contribution to enhance and expand this sentiment analysis project. Contributions can be made in several ways, including but not limited to:

- **Code Enhancements**: If you have ideas or suggestions for improving the existing codebase, feel free to submit pull requests. Your contributions can include optimizing the algorithms, improving performance, adding new features, or addressing any identified issues.
- **Bug Reports**: If you encounter any bugs or unexpected behavior while using the project, please submit detailed bug reports. This helps us identify and resolve issues promptly.
- **Documentation**: Contributions to the documentation are highly valued. You can propose improvements to the existing documentation, add code examples, or even create new sections to provide more comprehensive guidance.
- **Feedback and Ideas**: I encourage you to share your feedback, ideas, and suggestions related to this project. Your input can help shape the future direction of the project and identify areas of improvement.
- **Testing and Validation**: Thorough testing is crucial for ensuring the reliability and accuracy of the sentiment analysis model. Contributions related to testing, validation, and benchmarking are greatly appreciated.

To contribute, please fork the project repository, make your changes in a separate branch, and submit a pull request with a detailed description of your contributions. I will review your submission and collaborate with you to merge the changes into the main project.

I value and appreciate all contributions, no matter the size or scope. Your involvement helps in advancing the project and making it more valuable to the community. Thank you for considering contributing to this sentiment analysis project!

