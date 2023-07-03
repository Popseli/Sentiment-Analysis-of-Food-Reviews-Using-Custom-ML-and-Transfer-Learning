# Project Overview
The aim of this project is to evaluate and compare the sentiment prediction performances of various supervised machine learning (ML) algorithms and an unsupervised pre-trained transformer model. This project uses Kaggle's [Amazon fine food reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset for the analysis. The supervised custom ML algorithms used in this project are traditional ML algorithms namely Logistic Regression, Multinomial Naive Bayes, SVM and LightGBM, and deep learning algorithms namely deep Neural Networks and 1 dimensional CNN. For the pre-trained transformer model, a version of Bidirectional Encoder Representations from Transformers (BERT), which is fine-tuned for sentiment analysis of product reviews, is used. [Here](https://github.com/Popseli/Sentiment-Analysis-of-Food-Reviews-Using-Custom-ML-and-Transfer-Learning-Methods/blob/main/Sentiment%20Analysis%20of%20Amazon%20Food%20Reviews%20-%20ML%20%26%20BERT%207.ipynb) is the link to the notebook of the project.
# Dataset
The dataset consists of 568,454 user reviews on various food products. For each record, the dataset consists of three important data inputs (columns): review, summary and rating. The review describes the user's feedback regarding the product while the summary describes the general sentiment of the feedback. The rating of each review is given in numbers between 1 and 5, 1 and 5 being the poorest and the best respectively.
# Methodology
We aim to implement a model that will predict a review's sentiment as positive, neutral or negative. To achieve the objective, the following key tasks were performed;
   1. Data cleaning to remove redundant reviews and errors.
   2. Categorizing review ratings 1-2, 3 and 4-5 as negative, neutral and positive sentiments respectively.
   4. Performing EDA to analyse the distribution of data per rating/sentiment as well as the word cloud of the dataset per sentiment category.
   5. Selecting a small subset of unseen/test data from the dataset for evaluating and comparing the performances of the models.
   6. Balancing the remaining dataset by selecting a data subset consisting of an equal number of reviews for each sentiment category.
   7. Merging review and summary columns to form a new data input column.
   8. Text pre-processing of the merged data.
   9. Splitting the dataset into train and validation datasets.
   10. Modelling the supervised models with the train data and evaluating them with the validation dataset to find the best-performing model. TFIDF and Glove's vector representation methods were used for text vectorization for traditional ML and deep learning-based models respectively.
   11. Predicting sentiments of the unseen data using the best-performing custom ML model.
   12. Predicting sentiments of the unseen data using the pre-trained BERT model.
   13. Comparing the prediction performances of the custom ML and BERT on the unseen data.
   14. Performing false prediction analysis of the overall best-performing model.
# Prediction Result Summary
Below are sentiment best-performing prediction performances (confusion matrices and classification reports) of the best-performing custom ML and BERT on the unseen data:

A. Confusion Matrix and Classification Report of SVM

![](https://github.com/Popseli/Sentiment-Analysis-of-Food-Reviews-Using-Custom-ML-and-Transfer-Learning-Methods/blob/main/Confusion%20Matrix%20-%20Custom%20ML%2040.png)

![](https://github.com/Popseli/Sentiment-Analysis-of-Food-Reviews-Using-Custom-ML-and-Transfer-Learning-Methods/blob/main/Classification%20Report%20-%20Custom%20ML%2040.png)


B. Confusion Matrix and Classification Report of BERT

![](https://github.com/Popseli/Sentiment-Analysis-of-Food-Reviews-Using-Custom-ML-and-Transfer-Learning-Methods/blob/main/Confusion%20Matrix%20-%20BERT%2040.png)

![](https://github.com/Popseli/Sentiment-Analysis-of-Food-Reviews-Using-Custom-ML-and-Transfer-Learning-Methods/blob/main/Classification%20Report%20-%20BERT%2040.png)


It can be observed that the pre-trained BERT model has outperformed custom ML algorithms in overall accuracy, precision and recall. While the model has achieved 100% precision for the positive sentiment category, it has poorly performed in recall for the same category.



