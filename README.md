# RC_Tipping
Reservoir Computing Model for Tipping Prediction
with the help of Shirin Panahi, Ling-wei Kong, and Zhengmeng Zhai.

This is the repository for our preprint titled **"Machine-learning Prediction of Tipping with Applications to the Atlantic Meridional Overturning Circulation"**. This research focuses on using Reservoir Computing (RC) models to predict tipping points in complex systems, with a particular emphasis on the Atlantic Meridional Overturning Circulation (AMOC). 


## Project Overview
This repository contains code for implementing a **Reservoir Computing Model for Tipping Prediction**. The model is designed to forecast tipping points in time-series data using machine learning techniques, with a focus on optimizing model hyperparameters and training the RC model on real-world data.

The code is structured in two main phases:
1. **Hyperparameter Optimization**: 
   - Utilizes Bayesian Optimization to automatically find the optimal set of hyperparameters for the RC model.
   - The results are saved in the `hyperparameter` folder for future use.
   
2. **Training**:
   - Loads time-series data from the `data` folder.
   - Initializes the RC model using the previously optimized hyperparameters.
   - Trains the model, makes predictions, and validates its performance.

## Repository Structure
- `hyperparameter/`: Contains the optimized hyperparameters obtained during the optimization phase.
- `data/`: This folder contains the time-series data files used for training and testing the model.

## Usage

### Hyperparameter Optimization
To perform hyperparameter optimization, use the phase='Hyperparameter_optimization'. The optimized hyperparameters will be saved in the `hyperparameter` folder for future use during training.

### Training
After obtaining the optimized hyperparameters, you can proceed to the training phase. Ensure that your time-series data is placed in the `data/` folder. The model will be initialized with the optimized parameters, and training will proceed.
