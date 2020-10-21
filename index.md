# Twitter Sentiment Analysis with Neural Networks: K-pop Industry
## By Francis Jin 

![brain](/images/nn.png)
![bts](/images/bts.png)

## Topic

This project seeks to gain insight into trends into the ever-evolving and increasingly popular Korean pop music industry, deemed “K-pop”, which has risen to one of the top musical genres in the world. By using facets of Natural Language processing and machine learning to conduct Sentiment Analysis on Twitter data, the analysis measures trends in popularity as well as variations amongst different K-pop groups. 

Ultimately, the project attempts to test the hypothesis that Twitter sentiment is a useful signal for the overall popularity and success of a musical artist group. 

## Metric

The metric we will use to test this hypothesis is the overall score the model achieves, when comparing its rankings of the top 10 K-pop groups right now to the ranking order done by Koreaboo, one of the largest online K-pop content media platforms (https://www.koreaboo.com/news/top-30-popular-kpop-groups-korea-right-now/). 

A strong overlap will signify the model’s ability to predict a K-pop groups popularity using Twitter sentiment, and overall popularity is known to be strongly correlated with the overall success of a musical group. 

## Methods

_Search set_

Using Tweepy API, a query function retrieves the last 3,000 tweets based on the following artist names as keywords: ‘BTS’, ‘TWICE’, ‘Red Velvet’, ‘ITZY’, ‘BLACKPINK’, ‘Mamamoo’, ‘Oh My Girl’, ‘Girls’ Generation’, ‘IZ*ONE’, ‘Lovelyz’. 

_Training Set_

Built with labeled data using a corpus file with ID keys to 5000 sentiment-labeled tweets, which we grab through the Twitter API, without saving any additional information as to comply with the Twitter Developer API usage rules. Then it is written to CSV and mapped to numeric values 0 for “negative”, 1 for “neutral”, 2 for “positive”, and 4 for “irrelevant”.

![valuecounts](/images/valuecounts.png)

_Pre-Processing_

Commonly in Natural language processing endeavors, text must be processed to be suitable for modeling. Here we use the following Python libraries such as nltk, re, spacy, to edit our texts to convert to lowercase, remove whitespace, remove personal pronouns, remove URLs and the # sign in hashtags, and simplify repeated characters. We then filter out the entries labeled “4” (irrelevant). 

_Feature Selection_

We fit and transform the data with a TFIDF transformer, term frequency–inverse document frequency, which weights words based on importance in a document. 

Then we use a Glove embedding, an unsupervised learning algorithm for distributed word representation, to create vector representations for words. This is achieved by mapping words into a meaningful space where the distance between words is related to semantic similarity.

## Models

_Keras Neural Network_

A sequential neural network from Keras library is used for multi-label classification on the sentiment labels 0, 1, and 2. A single inner layer, early stopping, and Dropout are used in the neural network with a SoftMax activation layer in the output, and loss set to categorical cross-entropy. 

We achieve scores of about 80% and 80% accuracy for Training and Validation respectively.

After training the model and getting decent accuracy results, we use it to label the Search Set and generate the proportion of positive, negative, and neutral classified tweets on our keyword in question. In this example, “blackpink”.  Strongly positive!

![scores](/images/scores.png)

_AutoML_

Google Cloud Platform’s AutoML service is useful for evaluating the viability of a model for NLP, and depending on the use case can generate very sufficient ready-to-deploy machine learning models. In our case, the training set is published to AutoML to train an NLP model for Sentiment Analysis for the labels: 0 for negative, 1 for positive, 2 for neutral.

We see the results of AutoML closely mirror that of my original Sequential neural network! Precision and Recall scores of 80.47%, Confusion Matrix below.

![automl](/images/automl.png)

## Results

We rank each K-pop group in order based on ratio of positive to negative tweets to create Sentiment Index rankings, and compare to Koreaboo Magazine Rankings. Most groups landed very close to real ranking! This supports our original hypothesis that Twitter sentiment is indicative of overall popularity. 

![](/images/rankings.png)

*Note that every single group received an overall positive sentiment rating, not surprising given that our subjects are Pop music groups in the entertainment industry! (As opposed to for example, controversial political topics).

This insight can be used by players in the media and entertainment industry to measure sentiment, social climate, popularity, and inform numerous business practices such as marketing, promotion, and strategy. 

