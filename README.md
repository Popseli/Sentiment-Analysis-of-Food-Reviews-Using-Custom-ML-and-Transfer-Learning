# Project Overview
The aim of this project is to evaluate and compare the sentiment prediction of various supervised custom machine-learning algorithms and an unsupervised pre-trained sentiment prediction model. In this project, we use Kaggle's dataset of [Amazon's fine food reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) for the analysis. The pre-trained sentiment prediction model used is a version of Bidirectional Encoder Representations from Transformers (BERT) fine-tuned for sentiment analysis of product reviews. Here is the link to the notebook of the project.
# Dataset
The dataset consists of 568,454 user reviews on various food products. For each record, the dataset consists of three important data inputs (columns): review, summary and rating. The review describes the user's feedback regarding the product while the summary describes the general sentiment of the feedback. The rating of each review is given in numbers between 1 and 5, 1 and 5 being the poorest and the best respectively.
# Methodology
We aim to implement a model that will predict a review's sentiment as positive, neutral or negative. To achieve the objective, the following key tasks were performed;
   1. Data cleaning to remove redundant reviews and errors associated with other review attributes.
   2. Classifying review ratings 1-2, 3 and 4-5 as negative, neutral and positive sentiment categories respectively.
   4. EDA was performed to analyse the distribution of data per rating/sentiment as well as the word cloud of the dataset per sentiment category.
   5. Identifying a small subset of unseen data from the dataset to test and compare the performances of the models.
   6. Balancing the remaining dataset by selecting a subset of reviews consisting of an equal number of reviews for each sentiment category.
   7. Combining review and summary columns to form a new data input column.
   8. Text pre-processing of data input.
   9. 
