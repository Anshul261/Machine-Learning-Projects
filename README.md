# Machine Learning
This project aimed to showcase the various skills required when working on machine learning projects. All the work done is on the same data set. This data set contains ten classes, each representing a different road sign. All the values in the data set were row vectors containing the pixel information.
## File Part 1
### Section Summary 
This file showcase the exploratory data analysis, data pre-processing, modelling, feature selection,evaluation, and data visualization.
## File Part 2
### Section Summary 
This file utilizes clustering algorithms to predict the class of the signs. The following methods are used: K-means clustering, Density-Based Spatial Clustering of Applications with Noise (DBSCAN),
Balanced Iterative Reducing and Clustering Using Hierarchies (BIRCH), and Agglomerative Clustering, Fuzzy C-Means, Gaussian Mixture Models. The highest silhouette score we got was 0.9903 in K-Means with 4 clusters. On the other hand, since this data set contains the actual class, we checked the accuracy of the different models and found the DBSCAN model to be the best, with an Accuracy of 82%.
## File Part 3
### Section Summary 
In this file, we perform cross-validation-based experimentation with Decision trees and Random Forests classifiers. The Model overfits the entire dataset as there is a consistently significant difference (10-15%) between metrics obtained from the training and validation sets. 
We also noticed a trade-off between fairness (class distribution) and accuracy. The impact of data processing is minimal, and a balance is necessary to avoid overfitting. The best overall result was using the Equalized Data with a 30% Split, producing an Accuracy of 90.61%. The following is the best decision tree classifier with Grid Search optimization and ModelEqualized Data with an accuracy of 93.31%.
## File Part 4
### Section Summary
This file utilizes the previous sections data sets and cross validation methods on Linear classifiers, Multi-layer Perceptron, and CNN's. The aim was to understand the underlying functioning of these algorithms and how to build them from scratch. I was able to learn how to work the kears and tenserflow libraries to create the models on different types of data utilizing different methods at each stage. The end result showed the linear classifier on the Normalised and outlier-removed data produce and accuracy of 88.75%, the MLP model on the same data set got an accuracy of 83.4%, and finally the CNN model with hyperparameter tunning produced and accuracy of 90%.
### EDA
The initial part of the file is dedicated to Exploratory Data Analysis (EDA). This crucial step involves visualizations and initial data exploration to gain insights into the data attributes. It also guides in selecting suitable features and building appropriate ML models. We used methods like describe(), shape(), isnull(), and columns() to extract overall information about the dataset. This helped us understand the dataset's summary, its dimensions, and the number of columns (features) each data point contained. We also used a boxplot to visualize the distribution of the feature values and identify any outliers. Z-normalisation for outlier detection revealed that the dataset contained 97 outliers.
### Data Processing
For outlier detection, we employed the Isolation Forest model. This model was chosen because the dataset contains high-dimensional data and presents a multiclass classification problem, making the Isolation Forest a suitable candidate for mining outliers. After implementing the model, we re-visualized the dataset using a box plot, this time with the outliers removed. We repeated this process with different values of contamination and with and without normalization.
We plotted two histogram graphs (Normal and Equalized using OpenCV's equalized histogram function) to visualize
the distribution of feature values. The dataset contained a heavy skew towards one side (more dark pixels). We decided to fix this using a combination of 5 data augmentation techniques, i.e., brighten, gamma
correction, noising, gaussian blur and CLAHE (Contrast limited adaptive histogram equalization) in that order. This helped improve the images' quality and normalized the pixel distribution. We output a series of
rows to help visualize the effect of each method on the image data.
### Model Building
We train the Gaussian NB classifier using two sets of train and test. 
Without preprocessing, the Gaussian NB Accuracy is 26.22%, and the F1 Score is 25.84% on the Normalized Data. With outlier mining, the accuracy drops to 23.43%, and the F1 Score is 21.59%, on the Complete Data Set, we get an Accuracy of 24.49% and an F1 score of 21.59%. 
Processed DF had an Accuracy of 58.13% and an F1 Score of 57.53 %, the Outlier Mining Data set got an Accuracy of 56.60% and an F1 Score of 56.19%, Z-normalized combined with Outlier Mining got an Accuracy of 56.60 F1 Score 56.19% and finally Z-normalized Accuracy 58.13% F1 Score 57.53%. 
### Performance metrics
The performance metrics we used were Accuracy, F1 score, Precision, TPR(or Recall/Sensitivity), AUC, Specificity
and FPR.
### Feature Selection
We used SVC for feature extraction. We were using 'svm. Coef', we obtained the weights for the features. Then, for each class (10 in total), we obtained the top 5, 10 and 20 features, respectively. These features are stored in a NumPy array. We had separate data frames representing the class associated and several features selected. E.g., a data frame 'xy0_10' refers to the top 10 features (columns) associated with the 0th class (class labels range from 0-9). Later, using these features, we created binary classifiers. After this, using the top 50, 100 and 200 features, we trained the Gaussian NB to classify all the images (multiclass). This was done using the features selected for each class in the previous step, making three separate data frames
to train the classifiers. However, the accuracy decreased as the number of features decreased; even at 200 features (20 from each class), we had a significantly lower accuracy than the entire dataset. Even if they are the top features of each class, more than two hundred features are needed to represent the dataset adequately.
