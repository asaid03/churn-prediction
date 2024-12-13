# Project Diary

## Date: 1th October 2024

### Summary
- Today, I concentrated on determining the specific sub-problem I will focus on for comparing ML algorithms. After some consideration, I decided to pursue **churn prediction**. 
- I also began evaluating which datasets i would be using in this project. In contention between Telecom's data set and Bank dataset.

### Next Steps
- Contiune researching and start writing up a project plan
- Research some basic machine learning models, such as nearest neighbours, to establish a foundational understanding.
- Continue exploring potential datasets for benchmarking and implementation in the churn prediction model.

## Date: 4th October 2024

### Summary
- I looked into research papers on machine learning algorithms and churn prediction to gain a wider understanding and understand on how to approach my fyp.

- Initial idea i have is to have multiple different ml models to be used to predict churn such as decision tree, Logistic regression , support vector machines. Then to evaluate them empirically and give some findings on each model.

### Next Steps
- Continue researching and writing up project plan
- Discuss project plan and my churning sup problem with supervisor before project plan deadline.


## Date: 8/10/24

### Summary
- Completed the first draft of the project plan. I also created a set of question to be asked for supervisors meeting

## Date: 10/10/24

### Summary
- I met with my supervisor, discussed the project plan, and got feedback on my questions. I then went on to finalise the project plan and submit it.


## Date: 14/10/24

### Summary
- I have selected the telecom dataset for my project and have been researching exploratory data analysis (EDA) to understand its importance..
- I plan to carry out EDA on the dataset, and in the meantime, I have been learning about the Seaborn library and how it can be utilised effectively for visualising data during EDA.

## Date: 19/10/24

### Summary
- I have begun conducting exploratory data analysis (EDA) on the telecom dataset.
- I carried out observations on the distribution of missing data using a heatmap. 
- <img src="Documents\Images\Missing_Values_Heatmap.png" alt="Description of image" width="50%" />
- Might run into issues later, since there are a number of rows with missing data.e.g internet usage and device protection. This would mean it would be harder to establish credible relationships between features hindering churn prediction performance potentially

## Date: 21/10/24

### Summary
Today, I continued Exploration Data analysis on the Telecom dataset. I utilised the Seaborn library to observe the distribution of churn categories visually and made my observations. Similarly, I made similar observations for other continuous variables, plotting histograms to understand the data better.

However, given that the dataset contains quite a lot of missing data values, I'm planning on changing datasets and researching alternative datasets that might offer more complete information. It would mean I would have to redo EDA, but it should be easier now.


## Date: 22/10/24

### Summary
I have selected a new dataset by the company IBM that can be downloaded from Kaggle. The dataset is a telecom dataset, and I have started EDA again. It was much easier this time as I had good practice with the previous dataset. The dataset did not have much missing data, and we dealt with it by deleting those rows. I had the choice of using the median values of the row to replace the missing data but opted to delete those rows as it was a deficient number(11 rows). I ran into some issues with the types of some of the dataset as some where object when it should have been float. Made me realise how important EDA is as it prevents errors later on when building Ml models.


## Date: 26/10/24 - 30/10/24

### Summary
Over these last few days, I decided to get my EDA completed, since the goal is to move on to actually implement ML algorithms that predict churn. First, I started investigating which customer attributes might have decent or strong correlation to churn. For each attribute, visual diagrams were created, such as boxplots and histograms, in order to discover some underlying relations between attributes. 

I also did deep analysis with Demographics and Payment Types. From the customers, whether **SeniorCitizen** and **Partner**, based on these two statuses, I found some trends in the Churn behavior. Then, I moved my attention to features regarding service. I analyzed features like **Online Security, Tech Support, Phone Service, and Multiple Lines** to determine whether the kind of services provided had some impact on the churning rate. This helped me highlight customers who could be in danger of churning because of a lack of additional services. I also explored **Internet Service types** and **Payment Methods** to see if there are any service types or payment preferences that could be related to churning. An example of observation i made was that the customers using an electronic check as a payment method showed higher churn. Based on that it appears there is some relationship to the billing types and customer satisfaction. 


Having completed the EDA, it gave insight into the structure of the data and relationships that existed. That allowed me to get an idea of what features may play an important role in predicting churn. That would help me with the next part of my development, building ML models and feature engineering. 


## Date: 10/11/24

### Summary
I have been preoccupied the last few days with assignments and didn't get to do as much work as I liked on my final year project, and now I need to catch up on my timeline. Today, I began my data preprocessing stage. I realised a mistake in my Explorative Data Analysis, where I deleted 11 rows of data because the total charge was missing. While this is partially correct, each data object has valuable information that can aid churn prediction modelling. To combat this, I utilised how long each customer stayed with the company and their monthly charge to manually compute the total charge.

After this, I decided to investigate how to convert the data for ML mothe del best as some of the data was non-numerical, such as payment methods. This is where I then learnt about encoding data and the various types of encodings. Given my dataset, I utilised binary encoding for binary variables (e.g., Yes/No or Male/Female categories) and one-hot encoding for categorical variables with multiple unique values (such as PaymentMethod or InternetService). Overall today i have set a solid foundation for model training and will be moving on to normalisation/scaling next which is key for my first ML algo KNN.


## Date: 11/11/24

### Summary
Today, I completed my data preprocessing process. I did this by normalising the data and creating 2 different normalised versions of the dataset. One is the min_max normalised date, while the other is the z_score dataset. I did this to test which normalised dataset performs better with my KNN model. After this, I began developing the KNN model from scratch, utilising no libraries and using Euclidean distance as the distance metric. My initial implementation was working very slowly. This was mainly due to how I was sorting computing distances utilising insertion sort, which has a time complexity of O(n^2), which could be better given that my dataset was 7000+.

After some research, I learnt about heapq, which has a time complexity of O(logn). This reduced my computation time from 20 minutes to about 1-2 minutes, which is a huge accomplishment. I also learnt how important optimisation is with ML models.

## Date: 12/11/24 - 15/11/24

### Summary
I began actually testing out the KNN models I created on my dataset. Initially, I ran my KNN with k set to 1 neighbour and tested it on both min_max and z_score normalised churn datasets. The results were relatively okay, with an error rate of about 0.23, which is understandable since KNN is a lazy algorithm. I then ran my algorithms on larger numbers of K to find the optimal k and plotted a bar chart of error results.

<img src="Documents\Images\fidning_opt_k_knn_results.png" alt="Bar chart of KNN error rates for different values of K" width="40%" /> here the optimal k value is 34 due to having lowest error rate.

However i had some slight issues of rerunning models each time i open up my notebook which would be timeconsuming. I began investigating ways to save models suc that i would not require retraining each time. This is where i found pickle,a Python library that allows saving and reloading objects such as models or evaluation metrics.This allowed me to foucs on  interpreting the results and visualising the error rates without redundant computations especially when testing different configurations or normalisation methods. Overall my workflow has become more optimised compared to before.



## Date: 18/11/24

### Summary
After this, I, then decided to implement a known conformal predictor. This allowed me to quantify uncertainty and assign confidence levels to predictions. However my implementation could be more efficent as it takes a long time to run on a large datatset so I have decided to test on the subset of the dataset(3500). This required me to go through Vladimir Vovk's research work on conformal predictors, which was fairly interesting. 

My implementation was a bit crude and could be improved with some optimisation as the run time was very long, and this could be due to repetitive copying of conformity score. However, I did improve its optimality by only recomputing conformity score when necessary, thus reducing run time greatly from previous runs.

## Date: 20/11/24
After some research on evaluation metrics for classification, I have realised that the current metrics I'm using, accuracy and error rate, are very weak. 
So, I have decided to use the metrics F1-score, accuracy, recall, precision, and F2-score. I created these from scratch, utilising the knowledge gained. This would help a lot since my dataset is imbalanced and give further insights, such as the number of false negatives and false positives. Utilising my new metrics class, I then updated my main Jupyter notebook to have much more in-depth results and added graphs. Overall, I am making some progress, but I should investigate/research more before coding or developing so that I wouldn't have to refactor too much.


## Date 22/11/24

I have researched more about software engineer mehtodologies in ML and based on that I have decided to implment unit test cases for classes which is what I have done. These test classes where added for conformal and KNN classes and are fully working.

My knn is model is fully complete now and have merged into main branch with no issues.


## Date 26/11/24
I have been watching videos and looking into papers about decision tree models in the recent days. I specifally focused on different uniformity measures, utilising this knowledge I then implemented a decsion tree class, haven't fully tested it on my data but will do that soon. I intialy started with a basic Decision tree class but realised to accomodate for different uniformity measures I would have to split into 2 classes and adjust the paramters
Looking at my original plan I have overestimated the amount of work I could do, this mainly due to workload of other modules. I am a bit behind but not so much

I have also booked a meeting with my supervisor again tomorrow and this is mainly to establish if im ontrack and to answer any questions I have.

## Date 27/11/24
Today I attended my meeting with my supervisor and I have completed a good amount of work for interim submission. I also managed to discuss what should be in the report and how I could go about presenting my fyp later on. 

Furthermore, I have carried out inital testing for my Decision tree class and is preforming relatively well, some higher scores compared to knn.

## Date 05/12/24 - 09/2/24

I have been really sick these past couple of days. It is a shame, considering I wanted to improve the validity of my models by adding cross-validation and perhaps adding decision boundary results for my different models. This has set me back as now I need to complete my report writing; this also meant I don't think I would be able to do my SVN model or basic GUI. However, on the bright side, I have identified and fixed a key issue; this was the fact that I was normalising the whole dataset and then splitting the churn dataset for testing and training. This had led to data leakage, which affected the validity of my results. I fixed this issue by splitting and then normalising my dataset.This has led to slightly lower results in my decision tree but hasn't affected my KNN model results.

Furthermore, in this timeperiod I have finished my testing of decision tree model and am currently writing up my report/presentation.



















