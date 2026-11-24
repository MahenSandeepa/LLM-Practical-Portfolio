# Lab 01 – Sentiment Analysis Using Traditional Machine Learning

## 1. Laboratory Session

**Module:** PUSL3205 – Large Language Models and Generative AI

## 2. Objective

To practise basic Natural Language Processing (NLP) techniques by preprocessing text data, exploring sentiment-labelled reviews, converting text into numerical features using Bag of Words (BoW), and training and evaluating a machine learning model for sentiment classification.

## 3. Tasks and Implementation

The practical was implemented using Python in a Jupyter Notebook within Visual Studio Code.

1. **Dataset Loading:** Loaded sentiment-labelled reviews from Amazon, IMDb, and Yelp using Pandas and combined them into a single dataset.
2. **Text Preprocessing:** Converted reviews to lowercase, removed punctuation and unnecessary characters, eliminated extra whitespace, and created a `clean_text` column.
3. **Data Exploration:** Displayed positive and negative review examples and identified frequently occurring words in each sentiment category.
4. **Dataset Splitting:** Divided the dataset into 80% training data and 20% testing data.
5. **Bag of Words:** Converted the cleaned reviews into numerical word-count features using `CountVectorizer`.
6. **Model Training:** Trained a Logistic Regression classifier using the training features.
7. **Model Evaluation:** Evaluated the classifier using accuracy, precision, recall, F1-score, and a classification report.
8. **New Sentence Prediction:** Tested the trained model using a new review sentence to predict its sentiment.

## 4. Results and Evidence

**Dataset Information**

- Total reviews: 2,748
- Amazon reviews: 1,000
- IMDb reviews: 748
- Yelp reviews: 1,000
- Training samples: 2,198
- Testing samples: 550

**Model Evaluation Results**

| Metric | Result |
|---|---|
| Accuracy | 82.18% |
| Precision | 84.03% |
| Recall | 79.78% |
| F1-score | 81.85% |

Precision, recall, and F1-score refer to the positive sentiment class.

**New Sentence Prediction**

- Input: "This product is amazing and works perfectly!"
- Predicted sentiment: Positive

**Implementation Notebook**

[Sentiment Analysis Notebook](notebooks/Sentiment_Analysis.ipynb)

**Supporting Screenshots**

The following screenshots provide evidence of the implementation and results for all eight laboratory tasks.

1. [Task 1 – Dataset Loading](screenshots/Task01.png)
2. [Task 2 – Text Preprocessing](screenshots/Task02.png)
3. [Task 3 – Positive and Negative Reviews](screenshots/Task03(1).png)
   [Task 3 – Word Frequency Analysis](screenshots/Task03(2).png)
4. [Task 4 – Dataset Splitting](screenshots/Task04.png)
5. [Task 5 – Bag of Words](screenshots/Task05.png)
6. [Task 6 – Model Training](screenshots/Task06.png)
7. [Task 7 – Model Evaluation](screenshots/Task07.png)
8. [Task 8 – New Sentence Prediction](screenshots/Task08.png)

## 5. Observations

Text preprocessing standardised the review sentences by removing punctuation, converting text to lowercase, and eliminating unnecessary whitespace.

The Bag of Words approach transformed text into numerical features suitable for machine learning.

The Logistic Regression model achieved 82.18% accuracy on the testing dataset. Its recall was slightly higher for negative reviews than positive reviews, indicating that it identified a greater proportion of negative examples correctly.

The model also successfully classified the new positive review sentence as positive.

One limitation of Bag of Words is that it does not preserve word order, which may affect sentiment classification when sentence meaning depends on context or negation.

## 6. Learning Outcomes

Through this practical, I gained experience in:

- Loading and processing labelled text datasets using Pandas.
- Performing basic NLP preprocessing operations.
- Exploring positive and negative sentiment patterns.
- Splitting data into training and testing sets.
- Converting text into numerical features using Bag of Words.
- Training a Logistic Regression classification model.
- Evaluating classification performance using standard metrics.
- Predicting the sentiment of previously unseen text.