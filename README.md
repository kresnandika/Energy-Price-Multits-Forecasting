# About Datasets

### Context
In a paper released early 2019, forecasting in energy markets is identified as one of the highest leverage contribution areas of Machine/Deep Learning toward transitioning to a renewable based electrical infrastructure.

### Content
This dataset contains 4 years of electrical consumption, generation, pricing, and weather data for Spain. Consumption and generation data was retrieved from ENTSOE a public portal for Transmission Service Operator (TSO) data. Settlement prices were obtained from the Spanish TSO Red Electric España. Weather data was purchased as part of a personal project from the Open Weather API for the 5 largest cities in Spain and made public here.

### Acknowledgements
This data is publicly available via ENTSOE and REE and may be found in the links above.

### Inspiration
The dataset is unique because it contains hourly data for electrical consumption and the respective forecasts by the TSO for consumption and pricing. This allows prospective forecasts to be benchmarked against the current state of the art forecasts being used in industry.
- Visualize the load and marginal supply curves.
- What weather measurements, and cities influence most the electrical demand, prices, generation capacity?
- Can we forecast 24 hours in advance better than the TSO?
- Can we predict electrical price by time of day better than TSO?
- Forecast intraday price or electrical demand hour-by-hour.
- What is the next generation source to be activated on the load curve?

# Energy-Price-Multits-Forecasting

This notebook formulates a multi-variable forecasting problem to predict the next 24 hours of energy demand in Spain. 

This notebook is distinct from the previous implementation because it uses multiple input variables of past energy load, price, hour of the day, day of the week, and month. It does not use the weather inputs. The implementation is also nearly all in Tensorflow, with the exception of data prep and plotting.

**I made this notebook as practice and for submission dicoding "Pengembangan Machine Learning Developer"**

Here I compare the forecasting performance of using several different model types. Each model uses the same final two DNN layers with dropout. One of 128 units, and the final layer of 24 (the output horizon). Each of the models unique layers are:

*   A three layer DNN (one layer plus the common bottom two layers)
*   A CNN with two layers of 1D convolutions with max pooling.
*   A LSTM with two LSTM layers.
*   A CNN stacked LSTM with layers from models 2 and 3 feeding into the common DNN layer.
*   A CNN stacked LSTM with a skip connection to the common DNN layer.


A CNN stacked LSTM with a skip connection to the common DNN layer.
Each model is compared against baseline persistance models consisting of a one day persistence, and a three day average persistence. Added to the baseline error is the Transmission Service Operator's error.

**Reference for study :**

1.   https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction
2.   https://www.dicoding.com/academies/185 ( Pengembangan Machine Learning )

# Baseline Error Performance

In order to compare model perofrmance we need an estimate of bayes limit for the problem. In this case we do not have a human error reference. So we use the the lowest of the following:

ENTSOE recorded forecast. This is the collection of models used by the relevant energy authority.
Persistance 1 Day. Using the observed values from the previous days as the prediction of the next day.
Persistance 3 day mean. Using the observations from the previous 3 days as the prediction of the next day.

# Preparing the Data

We will use tf.datasets to prepare the data. The general strategy is to clean, scale, and split the data before creating the tf.dataset object. These steps can alternatively be done within the tf.dataset itself.

**Cleaning data:** Fill any missing values with a linear interpolation of the value. Same as done in the persistence dataset.

**Scaling data:** In all cases the data is min max scaled.

**Features:** As part of this simple analysis of models two feature sets are prepared. The univariate that contains energy consumption data only. The multivariate that contains energy consumption, price, day of the week, and month of the year.

**Splitting data:** One year of test data (8769 hourly samples) is put aside to evaluate all the models. The train and validation sets are created with a 80/20 split. 

-- submission for dicoding should use 20% val data --

# Windowing the Dataset

Use tf.dataset to create a window dataset. This is a vector of past timesteps (n_steps) that is used to predict on a target vector of future steps (n_horizon). The example below shows the output for n_steps = 72 and n_horizon = 24 and the 5 features. So we use the last 3 days (72 hours) to predict the next day (following 24 hours).

The resulting shape for X will be (batch size, n_steps, features) and Y will be (batch size, n_horizon, features).

# Dataset Loading Function

Wrap the above functions into a single function that allows us to build the dataset in the same way each time.

# Model Configurations

Define a set of model configurations so that we can call and run each model in the same way. The cgf_model_run dictionary will store the model, its history, and the test datasetset generated.

The default model parameters are:

*   n_steps: last 30 days
*   n_horizon: next 24 hours
*   learning rate: 3e-4

# Define Callback class

Model training will be stopped if its mae or loss is good enough

# Define Each Model and Train Model

## DNN
A single 128 unit layer plus the common 128 and 24 unit layyers with dropout.

## CNN
Two Conv 1D layers with 64 filters each, and kernel sizes of 6 and 3 respectively. After each Conv1D layer a maxpooling1D layer with size of 2.

## LSTM
Two LSTM layers with 72 and 48 units each.

## CNN and LSTM Stacked
Using the same layers from the CNN and LSTM model, stack the CNN as input to the pair of LSTMs.

## CNN and LSTM with a skip connection
The same CNN and LSTM layers as the previous models this time with a skip connection direct to the common DNN layer.
![image](https://user-images.githubusercontent.com/64974149/135601700-94cea350-74c7-4fc4-a0c1-4e556d5da9c0.png)

# Evaluation of Training/Validation Results

Loss curves across the models are fairly stable. All models show a flat validation curve while training continues to decline. The LSTM appears to begin to become very overfit from about epoch 100 where the validation loss begins to rise. The lstm_skip also has a point around epoch 50 where the val loss stops decreasing. In all cases this is a sign the models are no longer learning against the validation set. Some options to help improve this are to introduce learning rate decline, or train on longer input sequences.

Plots of the MAE show a similar pattern to the loss plots.

**Loss Curves**
![image](https://user-images.githubusercontent.com/64974149/135601732-c81d617c-ef77-46e5-875f-1d923dee5cea.png)
**MAE Curves**
![image](https://user-images.githubusercontent.com/64974149/135601742-a8d583ef-1a93-40f8-b4b6-f7ee653774fa.png)

# Evaluation of Test Results

The LSTM and the CNN stacked LSTM models clearly outperformed the other four models. Whats surprising is to see how well both a CNN and DNN did on their. LSTM would be expected to perform well because of its ability to learn and remember longer trends in the data.

Comparing to the baseline results the models' performance was poor. The dnn, cnn, lstm, and lstm_cnn models improved against the persistence error (MAE ~ 0.106) but did not improve against the TSO's prediction error (MAE ~0.015, MW error ~443).

Putting the models' performance in perspective however the results show how with a limited lookback window, and simple features a lstm, and a cnn stacked with an lstm are a good starting choice for architecture.

# Visualizing Predictions

Plot the actual and predicted 24 hour intervals. Below is the first 14 days of predictions. Interesting to note how the LSTM appears to oscilate over a longer frequency compared with the other models. The CNN also seems to capture the intra day oscillations (within the 24 hour period). Looking at the CNN stacked LSTM we can see how these two characteristics of the model's learning combine.

![image](https://user-images.githubusercontent.com/64974149/135601801-cf01f8ec-b34d-4e54-898f-3bad6635dc5c.png)


