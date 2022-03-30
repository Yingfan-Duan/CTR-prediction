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



## Feature Engineering



## Modeling



## Conclusion and Recommendation



## Future Work
