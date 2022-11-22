# Neural Network Charity Analysis

## Overview

Using neural networks to credate predictive models for a charity to determine the rate of success of their donations.

Resources:

- SciKit
- Tensorflow
- Keras

## Results

### Pre-Processing

- The target for our model is whether or not the charity will be a success: IS_SUCCESSFUL.

- The features are:
  - APPLICATION_TYPE
  - AFFILIATION
  - CLASSIFICATION
  - USE_CASE
  - ORGANIZATION
  - INCOME_AMT
  - SPECIAL_CONSIDERATIONS
  
        # Split our preprocessed data into our features and target arrays
        y = application_df['IS_SUCCESSFUL'].values
        X = application_df.drop(['IS_SUCCESSFUL'],1).values

- We removed 'EIN' and the 'NAME' because they do not proivde relevant data to the model

        # Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
        application_df.drop(['EIN', 'NAME'], axis=1, inplace=True)

### Evaluating the Model

We chose 2 hidden layers in the model, one with 8 neurons and one with 5:

        # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
        number_input_features = len(X_train[0])
        hidden_nodes_layer1 = 8
        hidden_nodes_layer2 = 5

        nn = tf.keras.models.Sequential()

        # First hidden layer
        nn.add(
            tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
        )

        # Second hidden layer
        nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

        # Output layer
        nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
        
  With this initial model we were not able to achieve our goal of 75% accuracy:
  
          Loss: 0.5637441873550415, Accuracy: 0.7197667360305786
          
  We attempted to increase the accuracy of the model by changing the parameters of the Application and Class binning, increasing the number of hidden layers and changing the activation. They all increased the accuracy of the model marginally, but non achieved our goal of 75%
  
## Summary

Overall, this is a good start to building a for this data. There is still a lot of oppurtunity to create something that will be more accurate in predicting success for the charity. It think fine tuning the pertinet data and getting the perfect number of layers and neurons can get us some better results. If we don't see a big jump in accuracy, we may need switch up our methodolgy entirely and go with something more traditional like a Linear Regression model.
