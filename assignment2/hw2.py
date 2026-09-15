#title: a2-Shallow network
#date:9-15-26
#description:
#two regression models (a plain linear regression and a small PyTorch neural network) 
#that predict California housing prices, compare how well they perform



#note:made using gemini + claude for debugging 
#imports will be used for data and building the model
# Import standard Python libraries for OS tools, math, tables, and plotting
import os                         # Provides functions for interacting with the operating system (paths, files, etc.)
import numpy as np                 # Numerical computing library (handling arrays & math)
import pandas as pd                # Data manipulation library (handling DataFrames)
import matplotlib.pyplot as plt    # Plotting library for graphs

# Import PyTorch core modules for deep learning
import torch                       # Core PyTorch library for tensors and automatic differentiation
import torch.nn as nn              # Neural network components (layers, activation functions, loss functions)
import torch.optim as optim        # Optimization algorithms (e.g., Adam, SGD)

# Import Scikit-Learn tools for datasets, preprocessing, baseline models, and evaluation
from sklearn.datasets import fetch_california_housing        # Function to download/load the California Housing dataset
from sklearn.model_selection import train_test_split         # Function to split data into train/test sets
from sklearn.preprocessing import StandardScaler              # Tool to standardize features (mean=0, std=1)
from sklearn.linear_model import LinearRegression             # Scikit-learn's baseline linear regression model
from sklearn.metrics import mean_squared_error, r2_score      # Functions to compute MSE and R^2 evaluation metrics

# Set random seeds for PyTorch and NumPy so the random numbers generate the exact same way
# every time you run the script (ensures reproducible results).
torch.manual_seed(0)   # Fix PyTorch's random seed so weight initialization/training is reproducible
np.random.seed(0)      # Fix NumPy's random seed so any NumPy randomness is reproducible

def main():  # Define the main function that contains the entire workflow
    # ---------------------------------------------------------
    # Part 1: Data Loading and Exploration
    # ---------------------------------------------------------
    #essentially we will fetch the data that we imported above and print it out to see what we are working with
    print("--- Part 1: Data Loading and Exploration ---")  # Print a header so console output is organized by section

    # Fetch the California Housing dataset from Scikit-Learn.
    # as_frame=True returns the data formatted as a Pandas DataFrame for easy viewing.
    housing = fetch_california_housing(as_frame=True)  # Download/load the dataset as a Bunch object with a DataFrame inside
    df = housing.frame  # Extract the combined DataFrame (features + target) from the Bunch object 
    #note data frame is just a matrix type, but behaves more like a table

    # Print out basic details about the dataset to understand its structure
    print("\nDataset Description snippet:\n", housing.DESCR[:300], "...\n")  # Print first 300 characters of the dataset's description
    print("\nFirst 5 rows:\n", df.head())            # Show top 5 rows of the DataFrame
    print("\nSummary Statistics:\n", df.describe())  # Show mean, std, min, max, etc. for each column 

    # ---------------------------------------------------------
    # Part 2: Data Preprocessing
    # ---------------------------------------------------------
    print("\n--- Part 2: Data Preprocessing ---")  # Print a header for this section

    X = housing.data            # Feature matrix (8 numeric input variables) as a DataFrame(table)
    y = housing.target.values   # Target variable (median house value in $100k) as a NumPy array(array of house median house values)

    # Split dataset into 80% Training set and 20% Testing set.
    # random_state=0 ensures the random split is consistent across runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0  # Reserve 20% of data for testing, fix the random seed for reproducibility
    )#essentially this is a dictionary that sets test set to 20%

    # Scale/Normalize features so all features have a mean of 0 and a standard deviation of 1 (Standard deviation is the spread of data from the average).
    # Neural networks converge much faster when input features are on a similar scale(more efficent when a outputs aren't outliers i.e., the data space is between 0 and 1).
    scaler = StandardScaler()  # Create a StandardScaler object to compute mean/std and transform data
    X_train_scaled = scaler.fit_transform(X_train)  # Calculate mean/std on train set AND transform it
    X_test_scaled = scaler.transform(X_test)        # Transform test set using the train set's mean/std (no re-fitting)
    print("      \n data processing complete---test and training data ready ")
    # ---------------------------------------------------------
    # Part 3: Model Building and Training
    # ---------------------------------------------------------
    print("\n--- Part 3: Model Building and Training ---")  # Print a header for this section

    # --- Model 1: Baseline Linear Regression (Scikit-Learn) ---
    lr_model = LinearRegression()               # Create an instance of scikit-learn's LinearRegression model (an object of the linear regression model)
    lr_model.fit(X_train_scaled, y_train)       # Train simple linear equation y = W*X + b on the scaled training data (train the object using our scaled data which is 80% if you recall above)

    # --- Model 2: Neural Network with PyTorch ---
    # PyTorch operates on Tensors (multidimensional arrays), not NumPy arrays.
    # Convert NumPy arrays to PyTorch float32 Tensors.(in this case a matrix)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)  # Convert scaled training features to a float32 tensor

    # .unsqueeze(1) changes shape from 1D array (N,) to 2D column vector (N, 1) to match output shape
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Convert training targets to a float32 column-vector tensor

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)  # Convert scaled test features to a float32 tensor
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Convert test targets to a float32 column-vector tensor

    # Define the Neural Network class inheriting from nn.Module
    class MLP(nn.Module):  # Define a custom neural network class that inherits PyTorch's base Module class
        def __init__(self, input_dim):  # Constructor: runs once when the model object is created
            super(MLP, self).__init__()  # Call the parent nn.Module constructor to set up internal machinery
            # Layer 1: Takes input_dim (8 features) and maps to 32 hidden neurons
            self.hidden = nn.Linear(input_dim, 32)  # Fully-connected layer: input_dim inputs -> 32 outputs
            # Activation function: Introduces non-linearity so the network learns complex patterns
            self.relu = nn.ReLU()  # ReLU activation function: outputs max(0, x)
            # Layer 2: Maps 32 hidden neurons down to 1 output (predicted house price)
            self.output = nn.Linear(32, 1)  # Fully-connected layer: 32 inputs -> 1 output (the predicted price)

        # Defines how data passes forward through the network steps
        def forward(self, x):  # Forward pass: defines how input x flows through the layers
            x = self.hidden(x)  # Linear transform: W1*x + b1
            x = self.relu(x)    # Activation: max(0, x)
            x = self.output(x)  # Linear transform: W2*x + b2
            return x             # Return the final predicted value(s)

    input_dim = X_train_scaled.shape[1]  # Number of input features (8), taken from the shape of the scaled training data
    mlp_model = MLP(input_dim)           # Instantiate the model with the correct input dimension

    # Loss Function: Mean Squared Error measures how far predictions are from ground truth
    criterion = nn.MSELoss()  # Create the Mean Squared Error loss function object

    # Optimizer: Adam algorithm updates weights to minimize loss (learning rate lr=0.01)
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)  # Create an Adam optimizer tied to the model's trainable parameters

    # --- Training Loop ---
    epochs = 100          # Number of passes through the entire training dataset
    loss_history = []     # Empty list to store loss values so we can plot the learning curve later

    for epoch in range(1, epochs + 1):  # Loop over epochs, counting from 1 to 100 inclusive
        mlp_model.train()         # Put model in training mode (enables behaviors like dropout, if present)
        optimizer.zero_grad()     # Reset accumulated gradients from the previous step to zero

        predictions = mlp_model(X_train_tensor)        # Forward pass: run training data through the model to get predictions
        loss = criterion(predictions, y_train_tensor)  # Compute the MSE loss between predictions and true values

        loss.backward()   # Backward pass: compute gradients of the loss with respect to model weights via backpropagation
        optimizer.step()  # Update model weights based on the computed gradients

        loss_history.append(loss.item())  # Save this epoch's scalar loss value into the history list

        # Print progress every 10 epochs
        if epoch % 10 == 0:  # Check if the current epoch number is a multiple of 10
            print(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")  # Print the epoch number and current loss, formatted to 4 decimals

    # --- Plot Loss Curve ---
    plt.figure(figsize=(8, 5))  # Create a new figure with a specified width and height (in inches)
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss (MSE)')  # Plot loss value against epoch number
    plt.xlabel('Epochs')  # Label the x-axis
    plt.ylabel('Loss')    # Label the y-axis
    plt.title('PyTorch MLP Training Loss Progression')  # Add a title to the chart
    plt.grid(True)   # Turn on the background grid for readability
    plt.legend()      # Show the legend (uses the 'label' set in plt.plot)
    plt.savefig('loss_curve.png')  # Save the chart image to disk as a PNG file
    plt.close()  # Close the figure to free up memory
    print("\nLoss curve saved as 'loss_curve.png'")  # Confirm to the user that the plot was saved

    # ---------------------------------------------------------
    # Part 4: Model Evaluation
    # ---------------------------------------------------------
    print("\n--- Part 4: Model Evaluation ---")  # Print a header for this section

    # Evaluate Baseline Linear Regression
    lr_preds = lr_model.predict(X_test_scaled)         # Use the trained linear regression model to predict on the test set
    lr_mse = mean_squared_error(y_test, lr_preds)      # Compute Mean Squared Error between true and predicted values
    lr_rmse = np.sqrt(lr_mse)                          # Compute Root Mean Squared Error (square root of MSE)
    lr_r2 = r2_score(y_test, lr_preds)                 # Compute R-squared score for the linear regression model

    # Evaluate PyTorch Neural Network
    mlp_model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates, etc.)

    # torch.no_grad() disables gradient calculation to save memory and CPU compute during evaluation
    with torch.no_grad():  # Temporarily disable gradient tracking since we're not training here
        mlp_preds_tensor = mlp_model(X_test_tensor)     # Forward pass on the test set to get predicted tensor
        mlp_preds = mlp_preds_tensor.numpy()            # Convert PyTorch Tensor back to NumPy array for evaluation

    # Calculate metrics for Neural Network
    mlp_mse = mean_squared_error(y_test, mlp_preds)   # Compute Mean Squared Error for the neural network's predictions
    mlp_rmse = np.sqrt(mlp_mse)                       # Compute Root Mean Squared Error for the neural network
    mlp_r2 = r2_score(y_test, mlp_preds)              # Compute R-squared score for the neural network

    # Print final evaluation metrics for comparison
    print("\n--- Summary Results ---")  # Print a header for the final results summary
    print(f"Linear Regression -> MSE: {lr_mse:.4f}, RMSE: {lr_rmse:.4f}, R2: {lr_r2:.4f}")  # Print linear regression metrics, rounded to 4 decimals
    print(f"PyTorch MLP       -> MSE: {mlp_mse:.4f}, RMSE: {mlp_rmse:.4f}, R2: {mlp_r2:.4f}")  # Print neural network metrics, rounded to 4 decimals

# Standard Python entry point idiom: run main() only when script is executed directly
if __name__ == '__main__':  # Check if this file is being run directly (not imported as a module)
    main()  # Call the main function to execute the entire workflow