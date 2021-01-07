# TrendsPotting

#Time Series Model: TRENDSPOTTING
We have taken 6 products and based on previous sale we are predicting the future sale for that particular product. 

Dataset contains first Column as Month and other columns shows No_of_product_sold
Sample:
	1949-01	112
	1949-02	118
	1949-03	132

**Time series should be stationary**
	• There are two major factors that make a Trendspotting series non-stationary. They are
		○ Trend: non-constant mean
		○ Seasonality: Variation at specific time-frames
	• Methods to make them Stationary:
		○ Differencing: by taking difference using time lag
		○ Decomposition: model both trend and seasonality, then remove them
		

	
##	1. ARIMA Model(Auto Regressive Integrated Moving Average)((p,d,q) ):
		a. Uses observations from previous time steps as input to a regression equation to predict the value at the next time step.  
			i. X(t+1) = b0 + b1*X(t-1) + b2*X(t-2)
		○  p : This is the number of AR (Auto-Regressive) terms .
		○  q : This is the number of MA (Moving-Average) terms. Current value is dependent on current error terms and  previous error terms. size of the “window”
		○  d :differencing operator to remove trend and seasonality
			§ Example: 
			x = c(10, 4, 2, 9, 34)
			diff(x, lag=1, differences=1)
			# Returns: array([ -6.,  -2.,   7.,  25.], dtype=float32)
		
	
        stepwise_model = auto_arima(data[c], start_p=1, start_q=1,
                                   max_p=3, max_q=3, m=12,
                                   start_P=0, seasonal=True,
                                   d=1, D=1, trace=True,
                                   error_action='ignore',  
                                   suppress_warnings=True, 
                                   stepwise=True)


##	2. RANDOM FOREST MODEL
		a. A random forest fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
			i. n_estimators:  The number of trees in the forest.
			ii. max_depth:  If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
			iii. n_jobs: No of jobs to run parallel.
			iv. random_state: Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each 
			
			
		b. Initial Data format
			i. 
		c. After Processing the data : Converting into 3 columns Product Code, Week, Sales and added extra columns.
		Last Week Sales: this is simply the amount of sales that a product had in the previous week
		Last Week Diff: the difference between the amount of sales in the previous week and the week before it (t-1 - t-2)
		
		
		
		
		for week in range(40,52):
		    train = melt4[melt4['Week'] < week]
		    val = melt4[melt4['Week'] == week]
		
		    xtr, xts = train.drop(['Sales'], axis=1), val.drop(['Sales'], axis=1)
		    ytr, yts = train['Sales'].values, val['Sales'].values
		
		    mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
		    mdl.fit(xtr, ytr)
		
		    p = mdl.predict(xts)
		
		    error = rmsle(yts, p)
		    mean_error.append(error)
		print('Mean Error = %.5f' % np.mean(mean_error))
		
		d. Advantages:
			i. solve both classification as well as regression problems.
			ii. Reduces overfitting and improves accuracy. It creates as many trees on the subset of the data and combines the output of all the trees. 
			iii. automatically handle missing values.
			iv. No feature scaling required.
		e. Disadvantages
			i. Complexity.
			ii. Longer training period.
