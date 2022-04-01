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

![](/imgs/target-distribution.png)

Clearly our target variable is highly imbalanced. Actually, only 99583 rows are 1 among 2570068 rows of data. So we need to handle this problem for better model performance.

- **CTR** 

We calculated the mean of the target variable (CTR) for each day as shown in the bar plot below. We can see CTR of Day 6 & 7 are lower,Day 2 has the highest CTR.

![](/imgs/target-distribution-by-day.png)

- User exposure versus target variable

We also compared the distribution of user exposure for clicked and non-clicked groups. The user exposure was defined as the number of users that one ad was displayed to. And the comparison graph is shown as below. We can see that the blue area is significantly lower than the red area on the left hand side. This indicates that **exposing one ad to too many users may have some relation with low clicks. **

![](/imgs/user_exposure.png)

- **Correlation Heatmap**

This plot shows the correlation between every pair of variables. We can see most of the variables don't have strong relationship. But there are some exceptions. For example, *is_on_shelf_time* and *first_class*,  *is_on_shelf_time* and *s_app_size*, etc.

![](/imgs/heatmap.png)

- **CTR versus variables**

Here we calculated the CTR for different variables levels so we can know which variables have strong impact on target variable at a glance. Basically, apart from pt_d (number of days), ctr was different for different levels in these variables. 

Also we notice that for some of the variables, although the variable itself had a lot of levels, ctr only peaked at one or a few levels, which indicated we could reduce the number of categories for these variables.

![](/imgs/ctr_vars.png)

## Feature Engineering

Our Feature Engineering part can be divided into **data preprocessing** and **feature construction**.

### Data Preprocessing

- **Missing Value**
- **Basic feature extraction**
- **nominal data encoding**
- **ordinal data encoding**
- **memory reduction**

### **Feature Construction**

- **exposure: count features**
- **interaction: crossing count features**
- **CTR features**
- **CTR of previous day features**
- **embedding features**

## Modeling



## Conclusion and Recommendation



## Future Work
