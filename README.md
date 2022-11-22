# Neural Network Charity Analysis

## Overview

Using neural networks to credate predictive models for a charity to determine the rate of success of their donations.

Resources:

- SciKit
- Tensorflow
- Keras

## Results

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
