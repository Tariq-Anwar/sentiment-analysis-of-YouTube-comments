# sentiment-analysis of youtube comments using machine learning and deep learning technique
# Introduction
Sentiment analysis or opinion mining is a field of study related to the analysis of user opinions, feelings, assessments, attitudes and emotions expressed on social media and other online resources. The revolution in social media sites has also attracted users to video sharing sites, such as YouTube. Online users express their opinions or feelings about the videos they watch on these sites. 
# Data collection
For Dataset we used web scraping using Selenium tool to extracted the information from Particular YouTube channel extracted data stored in csv formate
# Data Cleaning and preprocessing
After data collection we need to clean the data make the data into structure formate.The first step of my data pre-processing process is to handle the missing data. Since the missing values are supposed to be text data, there is no way to impute them, thus the only option is to remove them. For Excample, there exist only 334 missing values out of 9999 total samples, so it would not affect model performance during training.
1.	Converting to Lowercase:
This step is performed because capitalization does not make a difference in the semantic importance of the word. Eg. ‘Travel’ and ‘travel’ should be treated as the same.
2.	Removing numerical values and punctuations:
Numerical values and special characters used in punctuation($,! etc.) do not contribute to determining the correct class
3.	Removing extra white spaces:
Such that each word is separated by a single white space, else there might be problems during tokenization
4.	Tokenizing into words: 
This refers to splitting a text string into a list of ‘tokens’, where each token is a word. For example, the sentence ‘I have huge biceps’ will be converted to [‘I’, ‘have’, ‘huge’, ‘biceps’].
5.	Removing non-alphabetical words and ‘Stop words’: 
‘Stop words’ refer to words like and, the, is, etc, which are important words when learning how to construct sentences, but of no use to us for predictive analytics.

# sentiment Detection
At this stage, each sentence of the review and opinion is examined for subjectivity. Sentences with subjective expressions are retained and that which conveys objective expressions are discarded. Sentiment analysis is done at different levels using common computational techniques like Unigrams, lemmas, negation and so on.
# sentiment classification
In this model we used ensemble of three different algorithms 
1. Logistic Regression
2. Support Vector Machine
3. Convolutional neural network
based on this three algorithm which perform highest accuracy we going to take for our model.
# presentation output
The output of this project is that the user can easily visualize the changes happening in the Model specifies positive, negative, neutral for particular video. By using sentiment analysis and automating this process, you can easily drill down into different customer segments of your business and get a better understanding of sentiment in these segments.
