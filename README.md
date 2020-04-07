# Hotel-Booking-Demand

The dataset Hotel_Bookings from the article Hotel Booking Demand Datasets, written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019. 
https://www.sciencedirect.com/science/article/pii/S2352340918315191#s0005

It contains hotel booking information for a city and a resort hotel that includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.

Note: No personnel information is showcased in the data.

## Analysis

1. Data is first read to Python, then cleaned and prepped for analysis.
2. Descriptive statistics, and some EDA are performed to find out trends and pattern. 
3. A Decision Tree model is applied on the data to find out if a booking will be canceled or not. 
4. Model evaluation. 

The project was done in Jupyter Notebook.

## Model Evaluation

* The classification rate for the tree model is 0.82 or 82%
* While there are many ways to evaluate a model, confusion matrix is one of them. It’s a tabular representation of the algorithm. It tells us which classes are misclassified and which are not. Recall can be defined as the ratio of the total number of correctly classified positives divided by the total number of positive examples. High Recall is the class that is correctly recognized (fewer FN).
* Precision can be calculated by dividing the total True Positives by the total number of predicted positive examples (TP+FP). High Precision means that positive is correctly classified as positive. The F-Measure should be approximately the minimum value of precision or recall.
