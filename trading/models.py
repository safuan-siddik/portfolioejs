"""
Custom Neural Network and LSTM Implementations
=============================================
This module contains custom implementations of neural networks and LSTM
for time series prediction. It includes a custom-built neural network,
an LSTM cell, and a complete LSTM model using PyTorch.

Classes:
- CustomNeuralNetwork: A simple neural network with one hidden layer.
- CustomLSTMCell: A custom implementation of an LSTM cell.
- CustomLSTM: A complete LSTM model built with PyTorch.
- LSTMModel: A more advanced LSTM model with dropout and configurable layers.
"""

import numpy as np
import json
import math
import logging
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

logger = logging.getLogger("TradingBot")

class CustomNeuralNetwork:
    """
    A custom implementation of a neural network with one hidden layer.

    This class provides a simple, from-scratch implementation of a neural
    network, which is useful for understanding the underlying mechanics.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network with random weights.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output neurons.
            learning_rate (float): The learning rate for gradient descent.
        """
        # Initialize weights with Xavier/Glorot initialization for better stability
        scale_input = np.sqrt(2.0 / input_size)
        scale_hidden = np.sqrt(2.0 / hidden_size)
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * scale_input
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * scale_hidden
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        """
        The sigmoid activation function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The derivative of the sigmoid function.
        """
        return x * (1 - x)
    
    def tanh(self, x):
        """
        The hyperbolic tangent (tanh) activation function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output of the tanh function.
        """
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """
        The derivative of the tanh function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The derivative of the tanh function.
        """
        return 1 - np.power(x, 2)
    
    def relu(self, x):
        """
        The Rectified Linear Unit (ReLU) activation function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output of the ReLU function.
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        The derivative of the ReLU function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The derivative of the ReLU function.
        """
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """
        Perform a forward pass through the network.

        Args:
            X (np.ndarray): The input data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the output of the network
            and the output of the hidden layer.
        """
        # Shape check and log
        if X.shape[1] != self.weights_input_hidden.shape[0]:
            raise ValueError(f"Input shape mismatch in NN forward: X.shape={X.shape}, weights_input_hidden.shape={self.weights_input_hidden.shape}")
        print(f"[NN forward] X.shape: {X.shape}, weights_input_hidden.shape: {self.weights_input_hidden.shape}, bias_hidden.shape: {self.bias_hidden.shape}")
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        print(f"[NN forward] hidden_input.shape: {hidden_input.shape}")
        hidden_output = self.tanh(hidden_input)
        print(f"[NN forward] hidden_output.shape: {hidden_output.shape}, weights_hidden_output.shape: {self.weights_hidden_output.shape}, bias_output.shape: {self.bias_output.shape}")
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        print(f"[NN forward] output_input.shape: {output_input.shape}")
        output = output_input  # Linear activation for regression
        return output, hidden_output
    
    def backward(self, X, y, output, hidden_output):
        """
        Perform a backward pass to update the weights.

        This method uses gradient clipping to prevent exploding gradients.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target values.
            output (np.ndarray): The output of the network.
            hidden_output (np.ndarray): The output of the hidden layer.

        Returns:
            float: The mean squared error for the batch.
        """
        output_error = y - output
        output_delta = output_error
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.tanh_derivative(hidden_output)
        
        # Calculate gradients
        grad_weights_hidden_output = hidden_output.T.dot(output_delta)
        grad_bias_output = np.sum(output_delta, axis=0, keepdims=True)
        grad_weights_input_hidden = X.T.dot(hidden_delta)
        grad_bias_hidden = np.sum(hidden_delta, axis=0, keepdims=True)
        
        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        grad_norm = np.sqrt(np.sum(grad_weights_hidden_output**2) + np.sum(grad_bias_output**2) + 
                           np.sum(grad_weights_input_hidden**2) + np.sum(grad_bias_hidden**2))
        
        if grad_norm > max_grad_norm:
            scale = max_grad_norm / grad_norm
            grad_weights_hidden_output *= scale
            grad_bias_output *= scale
            grad_weights_input_hidden *= scale
            grad_bias_hidden *= scale
        
        # Update weights and biases
        self.weights_hidden_output += grad_weights_hidden_output * self.learning_rate
        self.bias_output += grad_bias_output * self.learning_rate
        self.weights_input_hidden += grad_weights_input_hidden * self.learning_rate
        self.bias_hidden += grad_bias_hidden * self.learning_rate
        
        return np.mean(np.square(output_error))
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network.

        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The training labels.
            epochs (int): The number of epochs to train for.
            batch_size (int): The size of each mini-batch.
            verbose (bool): Whether to print progress.

        Returns:
            Dict[str, List[float]]: A dictionary containing the loss history.
        """
        history = {'loss': []}
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Create mini-batches
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = math.ceil(n_samples / batch_size)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                y_batch = y_batch.reshape(-1, 1)
                # Check for NaN/Inf in input or target
                if np.isnan(X_batch).any() or np.isinf(X_batch).any() or np.isnan(y_batch).any() or np.isinf(y_batch).any():
                    print(f"[NN train] Skipping batch {i} due to NaN/Inf in data.")
                    continue
                print(f"[NN train] X_batch.shape: {X_batch.shape}")
                try:
                    # Forward pass
                    output, hidden_output = self.forward(X_batch)
                except Exception as e:
                    print(f"[NN train] Error in forward pass: {e}")
                    print(f"[NN train] X_batch.shape: {X_batch.shape}, weights_input_hidden.shape: {self.weights_input_hidden.shape}")
                    raise
                # Backward pass
                batch_loss = self.backward(X_batch, y_batch, output, hidden_output)
                print(f"[NN train] batch_loss: {batch_loss}")
                epoch_loss += batch_loss
            
            # Average loss for the epoch
            epoch_loss /= n_batches
            history['loss'].append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained network.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predictions.
        """
        output, _ = self.forward(X)
        return output
    
    def save_weights(self, filepath):
        """
        Save the model weights to a file.

        Args:
            filepath (str): The path to save the weights to.
        """
        weights = {
            'weights_input_hidden': self.weights_input_hidden.tolist(),
            'bias_hidden': self.bias_hidden.tolist(),
            'weights_hidden_output': self.weights_hidden_output.tolist(),
            'bias_output': self.bias_output.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(weights, f)
    
    def load_weights(self, filepath):
        """
        Load model weights from a file.

        Args:
            filepath (str): The path to load the weights from.
        """
        with open(filepath, 'r') as f:
            weights = json.load(f)
        
        self.weights_input_hidden = np.array(weights['weights_input_hidden'])
        self.bias_hidden = np.array(weights['bias_hidden'])
        self.weights_hidden_output = np.array(weights['weights_hidden_output'])
        self.bias_output = np.array(weights['bias_output'])


class CustomLSTMCell:
    """
    A custom implementation of a Long Short-Term Memory (LSTM) cell.

    This class provides a from-scratch implementation of an LSTM cell, which is
    useful for understanding the internal workings of LSTMs.
    """
    
    def __init__(self, input_size, hidden_size):
        """
        Initialize the LSTM cell parameters.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of units in the hidden state.
        """
        # Xavier/Glorot initialization for weights
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Input gate
        self.Wi = np.random.randn(input_size, hidden_size) * scale
        self.Ui = np.random.randn(hidden_size, hidden_size) * scale
        self.bi = np.zeros((1, hidden_size))
        
        # Forget gate
        self.Wf = np.random.randn(input_size, hidden_size) * scale
        self.Uf = np.random.randn(hidden_size, hidden_size) * scale
        self.bf = np.zeros((1, hidden_size))
        
        # Output gate
        self.Wo = np.random.randn(input_size, hidden_size) * scale
        self.Uo = np.random.randn(hidden_size, hidden_size) * scale
        self.bo = np.zeros((1, hidden_size))
        
        # Cell state
        self.Wc = np.random.randn(input_size, hidden_size) * scale
        self.Uc = np.random.randn(hidden_size, hidden_size) * scale
        self.bc = np.zeros((1, hidden_size))
        
        # Gradients
        self.dWi = np.zeros_like(self.Wi)
        self.dUi = np.zeros_like(self.Ui)
        self.dbi = np.zeros_like(self.bi)
        
        self.dWf = np.zeros_like(self.Wf)
        self.dUf = np.zeros_like(self.Uf)
        self.dbf = np.zeros_like(self.bf)
        
        self.dWo = np.zeros_like(self.Wo)
        self.dUo = np.zeros_like(self.Uo)
        self.dbo = np.zeros_like(self.bo)
        
        self.dWc = np.zeros_like(self.Wc)
        self.dUc = np.zeros_like(self.Uc)
        self.dbc = np.zeros_like(self.bc)
    
    def sigmoid(self, x):
        """
        The sigmoid activation function with clipping to prevent overflow.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, h_prev, c_prev):
        """
        Perform a forward pass through the LSTM cell.

        Args:
            x (np.ndarray): The input data for the current time step.
            h_prev (np.ndarray): The hidden state from the previous time step.
            c_prev (np.ndarray): The cell state from the previous time step.

        Returns:
            Tuple[np.ndarray, np.ndarray, tuple]: A tuple containing the next hidden state,
            the next cell state, and a cache of values for the backward pass.
        """
        # Input gate
        i = self.sigmoid(np.dot(x, self.Wi) + np.dot(h_prev, self.Ui) + self.bi)
        
        # Forget gate
        f = self.sigmoid(np.dot(x, self.Wf) + np.dot(h_prev, self.Uf) + self.bf)
        
        # Output gate
        o = self.sigmoid(np.dot(x, self.Wo) + np.dot(h_prev, self.Uo) + self.bo)
        
        # Cell state candidate
        c_candidate = np.tanh(np.dot(x, self.Wc) + np.dot(h_prev, self.Uc) + self.bc)
        
        # Cell state
        c_next = f * c_prev + i * c_candidate
        
        # Hidden state
        h_next = o * np.tanh(c_next)
        
        # Cache for backward pass
        cache = (x, h_prev, c_prev, i, f, o, c_candidate, c_next)
        
        return h_next, c_next, cache
    
    def backward(self, dh_next, dc_next, cache, learning_rate=0.01):
        """
        Perform a backward pass through the LSTM cell.

        Args:
            dh_next (np.ndarray): The gradient of the loss with respect to the next hidden state.
            dc_next (np.ndarray): The gradient of the loss with respect to the next cell state.
            cache (tuple): A cache of values from the forward pass.
            learning_rate (float): The learning rate for the weight updates.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the gradients
            with respect to the input, the previous hidden state, and the previous cell state.
        """
        (x, h_prev, c_prev, i, f, o, c_candidate, c_next) = cache
        
        # Output gate
        do = dh_next * np.tanh(c_next)
        do = do * o * (1 - o)
        
        # Cell state
        dc_next = dc_next + dh_next * o * (1 - np.tanh(c_next)**2)
        
        # Forget gate
        df = dc_next * c_prev
        df = df * f * (1 - f)
        
        # Input gate
        di = dc_next * c_candidate
        di = di * i * (1 - i)
        
        # Cell candidate
        dc_candidate = dc_next * i
        dc_candidate = dc_candidate * (1 - c_candidate**2)
        
        # Gradients for previous hidden state and cell state
        dh_prev = (np.dot(do, self.Uo.T) + 
                   np.dot(df, self.Uf.T) + 
                   np.dot(di, self.Ui.T) + 
                   np.dot(dc_candidate, self.Uc.T))
        dc_prev = dc_next * f
        
        # Compute gradients
        self.dWo += np.dot(x.T, do)
        self.dUo += np.dot(h_prev.T, do)
        self.dbo += np.sum(do, axis=0, keepdims=True)
        
        self.dWf += np.dot(x.T, df)
        self.dUf += np.dot(h_prev.T, df)
        self.dbf += np.sum(df, axis=0, keepdims=True)
        
        self.dWi += np.dot(x.T, di)
        self.dUi += np.dot(h_prev.T, di)
        self.dbi += np.sum(di, axis=0, keepdims=True)
        
        self.dWc += np.dot(x.T, dc_candidate)
        self.dUc += np.dot(h_prev.T, dc_candidate)
        self.dbc += np.sum(dc_candidate, axis=0, keepdims=True)
        
        # Gradient for input
        dx = (np.dot(do, self.Wo.T) + 
              np.dot(df, self.Wf.T) + 
              np.dot(di, self.Wi.T) + 
              np.dot(dc_candidate, self.Wc.T))
        
        # Update weights with gradient descent
        self.Wi -= learning_rate * self.dWi
        self.Ui -= learning_rate * self.dUi
        self.bi -= learning_rate * self.dbi
        
        self.Wf -= learning_rate * self.dWf
        self.Uf -= learning_rate * self.dUf
        self.bf -= learning_rate * self.dbf
        
        self.Wo -= learning_rate * self.dWo
        self.Uo -= learning_rate * self.dUo
        self.bo -= learning_rate * self.dbo
        
        self.Wc -= learning_rate * self.dWc
        self.Uc -= learning_rate * self.dUc
        self.bc -= learning_rate * self.dbc
        
        # Reset gradients
        self.dWi = np.zeros_like(self.Wi)
        self.dUi = np.zeros_like(self.Ui)
        self.dbi = np.zeros_like(self.bi)
        
        self.dWf = np.zeros_like(self.Wf)
        self.dUf = np.zeros_like(self.Uf)
        self.dbf = np.zeros_like(self.bf)
        
        self.dWo = np.zeros_like(self.Wo)
        self.dUo = np.zeros_like(self.Uo)
        self.dbo = np.zeros_like(self.bo)
        
        self.dWc = np.zeros_like(self.Wc)
        self.dUc = np.zeros_like(self.Uc)
        self.dbc = np.zeros_like(self.bc)
        
        return dx, dh_prev, dc_prev


class CustomLSTM(nn.Module):
    """
    A custom LSTM implementation using PyTorch with improved stability and error handling.

    This class builds upon the `nn.Module` from PyTorch to create a more robust
    LSTM model.
    """
    
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        """
        Initialize the CustomLSTM model.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of units in the hidden state.
            output_size (int): The number of output features.
            sequence_length (int): The length of the input sequences.
        """
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # Initialize LSTM layers with proper weight initialization
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()
        
        # Add batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    
    def _init_weights(self):
        """Initialize the weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Perform a forward pass with improved stability.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        try:
            # Input validation
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            
            batch_size = x.size(0)
            
            # Initialize hidden state and cell state
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
            
            # First LSTM layer with gradient clipping
            lstm1_out, (h1, c1) = self.lstm1(x, (h0, c0))
            lstm1_out = self.dropout(lstm1_out)
            
            # Second LSTM layer
            lstm2_out, (h2, c2) = self.lstm2(lstm1_out, (h1, c1))
            lstm2_out = self.dropout(lstm2_out)
            
            # Get the last output
            last_output = lstm2_out[:, -1, :]
            
            # Apply batch normalization
            last_output = self.batch_norm(last_output)
            
            # Final linear layer
            output = self.fc(last_output)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in LSTM forward pass: {e}")
            raise
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the model with improved error handling and monitoring.

        Args:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.
            epochs (int): The number of epochs to train for.
            batch_size (int): The size of each mini-batch.
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            Dict[str, float]: A dictionary containing the final loss.
        """
        try:
            # Convert data to tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            
            # Create data loader
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            # Initialize optimizer with gradient clipping
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # Training loop
            best_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                self.train()
                epoch_loss = 0
                
                for batch_X, batch_y in train_loader:
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Calculate average loss
                avg_loss = epoch_loss / len(train_loader)
                
                # Update learning rate
                scheduler.step(avg_loss)
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            return {'loss': best_loss}
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise
    
    def save_weights(self, path):
        """
        Save the model weights with error handling.

        Args:
            path (str): The path to save the weights to.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'sequence_length': self.sequence_length
            }, path)
            
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            raise
    
    def load_weights(self, path):
        """
        Load the model weights with error handling.

        Args:
            path (str): The path to load the weights from.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            # Load checkpoint
            checkpoint = torch.load(path, weights_only=False)
            # Verify model architecture
            if (checkpoint['input_size'] != self.input_size or
                checkpoint['hidden_size'] != self.hidden_size or
                checkpoint['output_size'] != self.output_size or
                checkpoint['sequence_length'] != self.sequence_length):
                raise ValueError("Model architecture mismatch")
            # Load state dict
            self.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise


class LSTMModel(nn.Module):
    """
    An advanced LSTM model with configurable layers and dropout.

    This class provides a more flexible and powerful LSTM model for time series
    prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        """
        Initialize the LSTMModel.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of units in the hidden state.
            num_layers (int): The number of LSTM layers.
            dropout (float): The dropout rate.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Final layer
        output = self.fc(last_output)
        return output
    
    def save_weights(self, path):
        """Save the model state dictionary with metadata."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.lstm.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }, path)
    
    def load_weights(self, path):
        """Load the model state dictionary."""
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Old format - just state dict
            self.load_state_dict(checkpoint)
        self.eval()

def train_model(model, train_loader, val_loader, epochs, learning_rate=0.001):
    """
    Train a PyTorch model with improved monitoring and error handling.

    This function includes features like learning rate scheduling and early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The data loader for the training set.
        val_loader (DataLoader): The data loader for the validation set.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        Dict[str, List[float]]: A dictionary containing the training and validation loss history.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    try:
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return history
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise 