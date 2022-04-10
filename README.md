# User Analysis: CTR Prediction on Features & Behaviors
This is the final project for MScA 31008 Data Mining Principles course in UChicago. Our team members are [Yingfan Duan](https://github.com/Yingfan-Duan), [Yu Han](https://www.linkedin.com/in/hanyu99/), [Rocco Wu](https://www.linkedin.com/in/rocco-wu/), [Xinyan Gu]() and [Dawei Zhao](https://www.linkedin.com/in/dawei-zhao-901a93214/).

## Project background

Advertising CTR prediction is at the core of any advertising campaign. **Increasing the accuracy of predictions** is critical to improving the effectiveness of advertising. 
In this project, we found a real paid anonymous Ad display dataset from HUAWEI DIGIX Global Challenge Competition, which provides a full range of user behavior, basic ad task attributes, and user profile data from day 1 to day 7, with the aim to promote the development of Advertising CTR prediction algorithms and predict user engagement with advertising (that is, click probability) at day 8. 

## Data Description

The training dataset (day 1 to day 7) contains 36 variables, which can be divided into three parts: user persona, Ad task attribute and media info. 

- User persona includes uid, age, city, gender, career, membership_life_duration, online_rate_30d etc.; 
- Ad task attribute includes adv_id, task_id, industry_name, etc.; 
- media info includes slot_id, tags, app_first_class, list_time etc. 

**The response variable** is named label, which refers to whether a user clicks or not. There are 40M observations in the original dataset but we will only select 1M for our project for easier handling and lighter memory usage. 

Here is our [data](https://www.kaggle.com/louischen7/2020-digix-advertisement-ctr-prediction).

## Exploratory Data Analysis

In this part, we visualized our dataset from different perspectives. Here I displayed part of our conclusions. 

- **Target Variable**

Our target variable is binary, 1 represents clicked and 0 represents un-clicked. Usually when facing a binary variable, we need to check it's distribution. If it's imbalanced, we need to come up with a way to solve the problem. 

> You could check this [blog](https://yingfan-duan.github.io/2022/04/01/Machine-Learning-Deal-with-imbalanced-data/) for the possible solutions and use cases.

<img src="/imgs/target-distribution.png" width="300">



Clearly our target variable is highly imbalanced. Actually, only 99583 rows are 1 among 2570068 rows of data. So we need to handle this problem for better model performance.

- **CTR** 

We calculated the mean of the target variable (CTR) for each day as shown in the bar plot below. We can see CTR of Day 6 & 7 are lower, Day 2 has the highest CTR.

<img src="/imgs/target-distribution-by-day.png" width="500">

- **User exposure versus target variable**

We also compared the distribution of user exposure for clicked and non-clicked groups. The user exposure was defined as the number of users that one ad was displayed to. And the comparison graph is shown as below. We can see that the blue area is significantly lower than the red area on the left hand side. This indicated that **exposing one ad to too many users may have some relation with low clicks**.

![](/imgs/user_exposure.png)

- **Correlation Heatmap**

This plot shows the correlation between every pair of variables. We can see most of the variables don't have strong relationship. But there are some exceptions. For example, *is_on_shelf_time* and *first_class*,  *is_on_shelf_time* and *s_app_size*, etc.

![](/imgs/heatmap.png)

- **CTR versus variables**

Here we calculated the CTR for different variables levels so we can know which variables have strong impact on target variable at a glance. Basically, apart from pt_d (number of days), ctr was different for different levels in these variables. 

Also we notice that for some of the variables, although the variable itself had a lot of levels, ctr only peaked at one or a few levels, which indicated we could reduce the number of categories for these variables.

![](/imgs/ctr_vars.png)

## Feature Engineering

Our Feature Engineering part can be divided into **data preprocessing** and **feature extraction**.

### Data Preprocessing

- **Missing Value**

  We have two features:`App Score` and `Membership Life Duration` have over 90% of values missing. We dropped these two columns. Then fill the rest missing values with -1 because there are some empirical evidence that -1 representing missing value will lift the model's performance a little.

  ![](/imgs/missing_value.png)

- **Basic feature extraction**

  Feature `Online Time` is string which contains users' online time. We extracted the start time and end time from this feature. For example, if online time's value is "**4**\^5\^6\^7\^8\^\^9\^10\^11\^12\^13\^14\^15\^16\^17\^18\^19\^**20**", then we created two columns `start time` and `end time` whose values are 4 and 20.

- **nominal data encoding**

  For nominal categorical columns such as `city`, `gender`, `Advertisement ID`, `Device ID`, we use label encoder from python to encode their levels except for -1 because it represents missing values.

- **ordinal data encoding**

  For ordinal categorical columns such as `mobile device launch time`, `device price/size`, we recategorized these columns based on different criterions such as equally spaced value and frequency.

- **memory reduction**

  Considering the size of our dataset and the limitation of our computing resources, we defined a function to reduce the memory that the data cost. To be more specific, we find appropriate data type for each feature so that each feature can use as little memory as possible. As a result, we **reduced 80% of the memory** used.

### **Feature Extraction**

Based on original features, we extracted five kind of features in this part.

- **Exposure: count features**

  - Compute the count for each feature value per day 
  - **Example**: User ‘A’ appeared 3 times on Day 1
  - Apply and create the Count Variables to only part of the features

- **Interaction: crossing count features**

  - Compute the count for each feature pair per day
  - **Example**: User ‘A’ + Advertisement ‘Apple’ appeared 5 times on Day 1
  - Apply and create the Crossing features to **some pair generated by user profile and advertisement characteristics**

- **CTR features**

  - use the mean of `label` column to get CTR
  - For training dataset (day 1 - day 6), the CTR is computed using its own day’s label mean
  - For validation dataset (day 7), the CTR is evaluated using the overall label mean of the rest days
  - Apply and create the CTR features to **every features** in the data set

- **CTR of previous day features**

  - Calculate the CTR based on the previous day’s label mean
  - Set day 6 as the previous day of both day 1 (train) and day 7 (validation)
  - Apply and create the PREVDAY_CTR features to **every features** in the data set

- **embedding features**

  Use Word2Vec to capture the embedding feature for user and advertisement. 

  **For example**, if user 1 is exposed to ad 1, 2, 3, then the ad list we have for user 1 is [1, 2, 3]. Then we use trained word2vec to convert each element in this list to a 8-d vector, then calculate the mean of these 3 vectors and get one 8d vector. This vector is the advertisement embedding feature for user 1. 

  Here considering the feature space size, we constructed two set of embedding features. In the first one, we only constructed embedding features using `user_id` and `adv_id`. In the second one, we constructed embedding features for all user and ads related features. 

###  T-SNE Visualization

After we constructed embedding features,  we apply the T-SNE in order to see if there are clustering patterns among the user and ads relationship. 

From the 7 visualizations, we can clearly see that when we cluster by display form of ads, internet status, and device prices, we are more likely to have the clear clustering with patterns, while the city rank, age or slot types do not have obvious clustering pattern. 

![](/imgs/tsne.png)

## Modeling

### Model Framework

![](/imgs/model-framework.png)

First, instead of randomly splitting, We use day 1-6 for training, use day 7 for validation. 

Secondly,  we sampled rows where labels are 0 by 50% to deal with imbalanced problem and speed up model training process. 

Thirdly, we use lightgbm to compare the performance of different set of features and select part of them to enter the final model. 

Finally, we use AUC to compare model's performance on validation set.

### Feature Selection

![](/imgs/comparison.png)

This line plot here show us the performance on validation set after adding different features. 

We can see stat features increased all metrics except for precision. Then adding CTR features increased recall greatly while precision decreased a lot. Then w2v features only increased AUC slightly. 

Finally, we selected original features and the first four set of features into our model.

### Model results

Then we compared the performance of LightGBM and Random Forest. And LightGBM was better than Random Forest no matter in which metric.

![img](/imgs/model-performance.png)

After that, we tuned the hyperparameters for LightGBM and the **best AUC was around 0.79.** 

Finally let's look at the feature importance of LightGBM .  Here we displayed the top 30 important features.  We can get several findings here:

1. Apart from the id related features. we have 8 customer related features such as their age, residence, career. 
2. Also, embedding features are important, which means Using ad embedding to represent users preference are effective
3. Besides,the importance of slot_id means ad postition influence CTR greatly. 
4. Finally, 4 device related features shows features like device type, price also impact CTR to some extent. 

![](/imgs/top30features.png)

## Challenges

- **Enormous data size** - needs high computational power 
- **Masked data** - can only conclude on feature importance, but unable to generate literal recommendations
- **Imbalance issue** - 96.1% vs. 3.9%; did perform SMOTE and undersampled the data, but was still unbalanced

## Future Work

- **Project Improvements**
  - More delicate parameter tuning
  - Embedding is an important technique to predict CTR, but did not lift our model’s performance up although deemed as important. We could try more combinations of embedded features and see how they can positively affect our model performance
  - Could add weight to days when predicting day 7

- **Future extensions**
  - Geospatial location analysis - city and province names in original dataset are masked. Had we have unmasked geographical information, we could potentially analyze CTR trend within different regions
  - Time of day analysis - we only know which day each record was on within a 7-day period. Had we have the specific time of day information about when the ads were pushed to users, we could do analysis on CTR patterns throughout different periods of a day
