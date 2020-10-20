# Geospatial Socioeconomic Mobility
## By Francis Jin

![](plotly map)


## Problem

The issue this project addresses is socioeconomic mobility amongst minority communities across America. 

Specifically attempting to answer the following questions: 

- What attributes about a given community have the biggest impact on the socioeconomic mobility of its children?
- Given data on those attributes, can we predict the level of socioeconomic mobility of children from that community? 
- What key visualizations can we create to portray this on a community-level in the US? 

Conclusions will provide insight into geographical variation, trends amongst communities, and important factors to children rising out of poverty, thus informing potential targeted solutions for minority communities and future policy reform. 

## Attempts made previously

Namely two studies from Opportunity Insights (where we drew our data from), tackled the issue in different ways. They used traditional statistical analysis and regression to investigate the impact of community and race on economic mobility. 

Studies: 
- Where is the Land of Opportunity? The Geography of Intergenerational Mobility in the United States https://opportunityinsights.org/paper/land-of-opportunity/
- Race and Economic Opportunity in the United States: An Intergenerational Perspective
https://opportunityinsights.org/paper/race/

I applied Machine Learning models for prediction and feature engineering to provide additional insight, while keeping in mind explainability given the social importance of the subject. I also approached the problem with Classification methods instead to add interpretability: given a specific community or area of communities, what level of economic mobility can we predict the children to have (low, medium, high)?  In addition, my project focuses on only minority communities, and I’ve added interactive visualizations to portray the geographical variations. 

## Data

The data is from Opportunity Insights (https://opportunityinsights.org/data/), a Harvard non-profit focused on the issue, which has a great library of data on socioeconomic and educational factors by geographic level across America.

Datasets: 
Neighborhood Characteristics by Commuting Zone (‘CZ_neighborhoodcharacteristicsbycsv.csv’)
Geography of Mobility: Commuting Zone Characteristics - Definitions and Data Sources (‘online_data_tables-8.xls')

## Data Cleaning and Pre-processing

The first dataset is a CSV of neighborhood characteristics, from which we grab certain racial share data and apply a filter for just the minority communities, grouping by Commuting Zone, a unique numeric identifier for communities ranging across the entire United States. 

Our second dataset is an XLS file from which we import the two sheets: Online Data Table 5 and Online Data Table 8. We filter and clean the sheets for the tables and features we care about, which span a range of educational, social, economic, and community attributes. Examples include - racial segregation, commuting times, fraction middle class, local tax rates, student teacher ratio, school expenditure per student, teenage labor force participation rate, violent crime rate, fraction of children with single mothers, etc. 

Finally we merge the two datasets on Commuting Zone, resulting with a dataset of 40 features and 500 entries. 

## Target Variable 

The target variable is the metric we use to measure socioeconomic mobility, deemed “Absolute Upward Mobility”, engineered from the paper (https://opportunityinsights.org/paper/land-of-opportunity/). It is the mean rank (in the national child income distribution) of children whose parents are at the 25th percentile of the national parent income distribution. The paper goes into great comprehensive detail of this ranking method, as well as data sources used such as Census Data and IRS tax filings, and adjustments for robustness of the metric. 



## EDA

Correlation tables and heat maps are printed for all features vs. the target variable (labeled ‘am, 80-82 cohort’). Immediately we see the biggest positive and negative correlations, including features such as fraction of children with single mothers, racial shares, high school dropout rate, fraction of adults married, fraction of middle class families, teenage labor force participation rate, etc. 

![](correlation table)
![](correlation heatmap)


We also investigate the distribution of the target variable with visualizations, noting a relatively normal distribution.  
We create target variable labels from the Absolute Upward Mobility metric for both Binary and Multi-label Classification. 

![](histogram)
![](distplot)
![](cdf)

For binary classification, 'am, 80-82 cohort’ is split in half by its numeric mean for labels 1 and 0, success being 1 and failure being 0, representing good or bad mobility. For multi-label classification, 'am, 80-82 cohort’ is split into quartiles 0-25%, 25-50%, 50-75%, and 75-100% - respectively representing low, medium, high, and excellent mobility. 

Note that Classification should not suffer from imbalanced classes given the distribution and engineering of the target variable labels. 

![](value counts)


## Feature Selection

I create a function for Mutual Information Classification to create feature rankings for binary and multi-label Classification and print top features. These results are consistent with the correlation EDA from before. 

![](top ten features)

## Model Selection and Results

_Binary Classification_

Started with a simple LogisticRegression model which showed poor performance. Ensemble methods RandomForestClassifier and GradientBoostingClassifier showed extremely high scores on training data (~95%) but much lower on test data (~80%), signaling overfitting. These models are likely suffering from over-cardinality and multi-collinearity, for example the features ‘fraction of adults married’ and ‘fraction of children with single mothers’ are naturally closely related. Also, given the modest size of the dataset, complicated models are prone to over-fitting. 


Here is an example of the Feature Importances from the GradientBoostingClassifier. The highest one is again fraction of children with single mothers, but other important features not identified before include manufacturing employment share, fraction religious, growth in Chinese imports. Interesting insight.

![](gradientboosted)

Given the need for regularization, I scaled the data and employed LogisticRegressionCV with elastic-net regularization, 5-fold cross-validation, and hyper-parameter tuning ranges of Cs = np.logspace(-10,10,50) and L1 ratios = np.arange(0,1,.05). 

Training and test set scores both drastically improved to ~90%, with roc_auc_score consistently above 90% as well. 

![](accuracy)


_K-Means Clustering_

Although not originally a clustering problem, I investigated the data with the K-Means Clustering method to find the optimal number of clusters to be around 4. This is theoretically consistent with our splitting of target variable labels into 4 groups for multi-label classification. I also appended cluster labels to the dataset as a feature in multi-label classification. 

![](elbow graph)

_Multi-Label Classification_

Using the same hyper-parameter tuning and regularization with 10-fold cross-validation, LogisticRegression is giving accuracy scores of ~80% on training and ~70% on test data. 

Gradient Boosting overfit again.  


## Visualizations

Utilizing Plotly for interactive graphical visualizations, displayed the results for both Binary and Multi-label Classification.

Hover over any city to view its Actual vs. Predicted mobility label. Note the higher accuracy of the model, the more identical the map colors will be. 

![](other maps)


## Conclusions

Our hypothesis is confirmed that using data on community-level attributes, we can predict the level of future socioeconomic mobility of children who grow up in that community. This shows not only that there are geographical variations in the likelihood of the success of children, but also the community-level features which are most important in determining this. 

We can identify who is disadvantaged or advantaged, why, and hopefully how to help more children rise up. Once able to identify the factors helping or preventing children’s success in rising out of poverty, we can start to use this information to inform social policy, community activism, education reform, and targeted solutions for communities across the country.












