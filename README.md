# twitter-NLP
Sentiment Analysis on Twitter data using Keras Neural Network and Google AutoML for NLP on K-pop industry [#blackpinkinyourarea](https://www.youtube.com/watch?v=ioNng23DkIM)

The main script [Twitter_public.ipynb](https://github.com/francisfjin/twitter-NLP/blob/main/Twitter_public.ipynb) lets you access Twitter API and search for a user-defined set of tweets given keyword of your choosing. Then it creates a training set through Twitter which will be fed into both a Keras Neural Network 

The notebooks are best run in Google Colab for seamless execution, as part of the project uses GCP's [AutoML](https://cloud.google.com/automl) as an alternative comparison to my original model. As such, you'll have to input your own file paths for your GCP locations and [mount your Drive](https://colab.research.google.com/notebooks/io.ipynb). 

For Twitter API, you will need to input your own API keys from your [developer account](https://developer.twitter.com/en/apply-for-access). This is necessary for building the Training Set using a corpus file from amazing Niek Sanders - you can get the file containing the corpus from this [link](https://github.com/karanluthra/twitter-sentiment-training/blob/master/corpus.csv). It has ID keys to 5000 sentiment-labeled tweets, which we then grab through the Twitter API, but do not save any unnecessary information, as to comply with their Developer API usage rules.  

Packages required not already in Anaconda Suite: 
- Tweepy
- Twitter 
- Spacy 
- nltk
- Keras
- Google Cloud AutoML

## AutoML 

Training Set: 

[Twitter_public.ipynb](https://github.com/francisfjin/twitter-NLP/blob/main/Twitter_public.ipynb) has a section that writes training set to CSV in Google AutoML required format. Then an AutoML model for NLP Sentiment Analysis in Google Cloud Platform can be trained on the set.  

Predictions: 

[Twitter_public.ipynb](https://github.com/francisfjin/twitter-NLP/blob/main/Twitter_public.ipynb): has a section that takes a search set of tweets and exports each one to a txt file as per AutoML's input file requirements, also writes a CSV in required format.

[AutoMLpythonclient_public.ipynb](https://github.com/francisfjin/twitter-NLP/blob/main/AutoMLpythonclient_public.ipynb): Python client to run the deployed model. Requires input GCP credentials, test files must saved to GCP storage bucket before running predictions. 


