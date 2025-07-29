# Data-Science-Course-Projects-S2025

- [Data-Science-Course-Projects-S2024](#data-science-course-projects-s2025)
  - [Intro](#intro)
  - [Assignment 0: Statistical Inference](#assignment-0-Statistical-Inference)
  - [Assignment 1: Sampling and Data Storytelling](#assignment-1-Sampling-and-Data-Storytelling)
  - [Assignment 2: Real Time Data Pipeline](#assignment-2-Real-Time-Data-Pipeline)
  - [Assignment 3: ML Modeling Showcase](#assignment-3-ML-Modeling-Showcase)
  - [Assignment 4: Deep Learning Applications](#assignment-4-Deep-Learning-Applications)
  - [Assignment 5: Advanced Data Science](#assignment-5-Advanced-Data-Science)
  - [Project](#Final-Project)
    - [Phase 1: Data Exploration and Storytelling](#phase-1-Data-Exploration-and-Storytelling)
    - [Phase 2: Data Engineering and Pipeline](#phase-2-Data-Engineering-and-Pipeline)
    - [Phase 3: Modeling and Pipeline Integration](#phase-3-Modeling-and-Pipeline-Integration)

## Intro

This repository contains the assignments for the Data Science course at the University of Tehran. The assignments are completed by the group of students in the course and are used to demonstrate the skills and knowledge they have gained throughout the course. They are completed using the python programming language and are focused on data analysis, visualization, and machine learning. The assignments are completed using real-world data sets and are designed to give students hands-on experience with the tools and techniques used in data science. Apart from the assignments, the repository also contains the final project for the course, which is a group project that involves working on a real-world data science problem from start to finish. In the final project we were supposed to gather data, clean it, analyze it, and build a machine learning model to solve a specific problem.

## Assignment 0: Statistical Inference

This project serves as a practical introduction to core statistical principles in data science. It is divided into three distinct case studies. The first part uses Monte Carlo simulation to analyze the profitability of betting in a game of roulette, exploring concepts like expected value, standard error, and the Central Limit Theorem. The second part delves into predictive analysis by aggregating polling data from the 2016 US Presidential Election to calculate confidence intervals and test hypotheses about the electoral spread. The final part applies hypothesis testing (specifically, independent t-tests) to a drug safety dataset to determine if there are statistically significant differences between a drug and a placebo group.

## Assignment 1: Sampling and Data Storytelling

This project combines advanced statistical simulation with practical data storytelling. The first part involves a deep dive into computational statistics by implementing the Langevin Dynamics sampling algorithm from scratch to draw samples from a 2D Gaussian distribution, and then comparing the results to standard library functions. The second part shifts to data visualization, utilizing Tableau to clean, analyze, and present insights from Airbnb listing data through a series of three interactive dashboards, focusing on creating a compelling, data-driven narrative for a public audience.

## Assignment 2: Real Time Data Pipeline

This project involves designing and implementing a comprehensive, real-time data pipeline to process payment transactions for a fictional provider, "Darooghe." The architecture uses Apache Kafka for high-throughput data ingestion, Apache Spark for both batch and real-time stream processing, and MongoDB for data storage. The pipeline is built to handle data validation, calculate commissions, analyze transaction patterns, detect fraudulent activities in real-time, and visualize key business intelligence insights.

## Assignment 3: ML Modeling Showcase

This project is a comprehensive showcase of machine learning modeling, tackling three distinct challenges through Kaggle competitions: classification, regression, and recommendation. For the classification task, a model is built to predict the survival status of cancer patients. The regression task involves predicting the daily demand for bike rentals based on seasonal and weather-related features. Finally, a collaborative filtering recommendation system is developed to predict user ratings for movies. The emphasis across all tasks is on rigorous data preprocessing, feature engineering, and the application of traditional machine learning algorithms to solve real-world problems.

## Assignment 4: Deep Learning Applications

This capstone project demonstrates the application of deep learning across three different domains. It includes building a Multi-Layer Perceptron (MLP) to predict the outcomes of the FIFA World Cup 2022, developing a Convolutional Neural Network (CNN) for flower image classification, and implementing a Recurrent Neural Network (RNN/LSTM) to forecast Bitcoin prices. Each task covers the complete pipeline from data preprocessing and model architecture design to training and comprehensive evaluation.

## Assignment 5: Advanced Data Science

This capstone project explores a diverse set of advanced topics in data science through four distinct, hands-on tasks. The project begins with semi-supervised and active learning to predict video game review scores from limited labeled text data. It then moves to natural language processing, building a semantic search engine for a Persian Q&A forum using modern embedding models and a vector database. The third task delves into the power of large language models (LLMs) for complex reasoning on the SWAG multiple-choice dataset. The final task covers computer vision, applying unsupervised clustering techniques for semantic segmentation of players in football images.

## Final Project

### Phase 1: Data Exploration and Storytelling

In this initial phase of our project, we focused on understanding the key drivers of flight delays using the "[2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)" dataset. We conducted a comprehensive exploratory data analysis (EDA) to investigate the relationships between various factors—such as airline, origin and destination airports, day of the week, and time of day—and the likelihood and duration of delays. The insights gathered were then used to create a data story in Power BI, visualizing significant trends and patterns to build a foundational understanding of the problem before moving to the data processing and modeling stages.

### Phase 2: Data Engineering and Pipeline

In this second phase, we built the engineering foundation for our flight delay prediction project. We began by designing a relational database schema and migrating our cleaned dataset into an SQLite database. Building on the insights from Phase 1, we then executed an advanced feature engineering and preprocessing workflow, creating new variables such as time-of-day categories and handling missing values systematically. This entire process was automated into a modular data pipeline using a series of Python scripts. To complete the phase, we implemented a CI/CD workflow with GitHub Actions, ensuring that our data pipeline is automatically tested and executed upon every code change, establishing a robust and reproducible process for the final modeling stage.

### Phase 3: Modeling and Pipeline Integration

In this conclusive phase, we developed and evaluated machine learning models to predict flight delays using the features engineered in Phase 2. We experimented with several algorithms, treating the problem as both a classification task (predicting if a delay will occur) and a regression task (predicting the delay duration), ultimately selecting a Multi-Layer Perceptron (MLP) model for its superior performance. After rigorous evaluation using metrics such as F1-Score and Mean Absolute Error (MAE), the final trained model was integrated back into our automated pipeline. This created a complete, end-to-end system capable of automatically training the model and making new predictions, which are then saved back to the database for analysis.
