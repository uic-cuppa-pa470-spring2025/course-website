# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "polars",
#     "torch",
#     "numpy",
# ]
# ///

import copy

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim


def main() -> None:
    torch.manual_seed(42)  # For reproducibility

    ## Read a CSV file into a DataFrame
    train_data = pl.read_csv("oecd_train.csv").drop("REF_AREA").cast(pl.Float32)
    test_data = pl.read_csv("oecd_test.csv").drop("REF_AREA").cast(pl.Float32)

    ## Separate the target variable from the features
    features = [
        "TIME_PERIOD",
        "Forest exposure to areas at risk of burning_Percentage of forested area",
        "Amount of burned area_Percentage of land area",
        "Population exposure to areas at risk of burning_Percentage of population",
        "Land soil moisture anomaly_Percentage change",
        "Cropland soil moisture anomaly_Percentage change",
        "Fine particulate matter (PM2.5)_Microgrammes per cubic metre",
    ]
    target = "Age-adjusted mortality rate_Total"

    train_features = train_data.select(features)
    train_target = train_data.select(target)
    test_features = test_data.select(features)
    test_target = test_data.select(target)

    ## Check if mps is available
    if not torch.backends.mps.is_available():
        device = torch.device("cpu")
        print("MPS is not available. Running on CPU.")
    else:
        device = torch.device("mps")
        print("MPS is available. Running on MPS.")

    ## Convert the DataFrame to a PyTorch tensor and normalize features
    train_features = train_features.to_torch("tensor").to(device)
    train_target = train_target.to_torch("tensor").to(device)
    test_features = test_features.to_torch("tensor").to(device)
    test_target = test_target.to_torch("tensor").to(device)

    ## Print the shape of the tensors
    print(f"Train features tensor shape: {train_features.shape}")
    print(f"Train target tensor shape: {train_target.shape}")
    print(f"Test features tensor shape: {test_features.shape}")
    print(f"Test target tensor shape: {test_target.shape}")
    ## Print the first few rows of the tensors
    print("Train features tensor:")
    print(train_features[:5])
    print("Train target tensor:")
    print(train_target[:5])
    print("Test features tensor:")
    print(test_features[:5])
    print("Test target tensor:")
    print(test_target[:5])

    ## Create a simple neural network model
    class SimpleNN(nn.Module):
        def __init__(self, input_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.relu1 = nn.LeakyReLU(0.1)
            self.batchnorm1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 64)
            self.relu2 = nn.LeakyReLU(0.1)
            self.batchnorm2 = nn.BatchNorm1d(64)
            self.fc3 = nn.Linear(64, 64)
            self.relu3 = nn.LeakyReLU(0.1)
            self.batchnorm3 = nn.BatchNorm1d(64)
            self.fc4 = nn.Linear(64, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.batchnorm1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.batchnorm2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.batchnorm3(x)
            x = self.fc4(x)
            return x

    ## Initialize the model, loss function, and optimizer
    input_size = len(features)
    output_size = 1
    model = SimpleNN(input_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ## Train the model with batching
    num_epochs = 100
    batch_size = 64
    n_samples = train_features.shape[0]

    # Number of batches
    num_samples = train_features.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Alternatively, create DataLoader for batch training
    # train_dataset = torch.utils.data.TensorDataset(train_features, train_target)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Track validation loss
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Alternatively, you can use DataLoader for batching
        # for batch_features, batch_targets in train_loader:

        for i in range(num_batches):
            # Get batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            # Extract batch data
            batch_features = train_features[start_idx:end_idx]
            batch_targets = train_target[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_features.size(0)

        # Calculate average loss over all samples
        avg_loss = total_loss / n_samples

        # Evaluate on validation set (using test data as validation)
        model.eval()
        with torch.no_grad():
            val_outputs = model(test_features)
            val_loss = criterion(val_outputs, test_target).item()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                print(f"Best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    ## Evaluate the model using the best model from validation
    # Load the best model weights
    model.load_state_dict(best_model)

    # Now evaluate with the best model
    with torch.no_grad():
        model.eval()
        test_outputs = model(test_features)
        test_loss = criterion(test_outputs, test_target)
        print(f"Test Loss: {test_loss.item():.4f}")

        # Print the first few predictions
        print("Test predictions:")
        print(test_outputs[:5])
        print("Test target values:")
        print(test_target[:5])
    ## Calculate RMSE and R^2
    mse = criterion(test_outputs, test_target)
    rmse = torch.sqrt(mse)
    # Correct R^2 calculation: 1 - (sum of squared errors / total variance)
    ss_res = torch.sum((test_target - test_outputs) ** 2)
    ss_tot = torch.sum((test_target - torch.mean(test_target)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"RMSE: {rmse.item():.4f}")
    print(f"R^2: {r2.item():.4f}")


if __name__ == "__main__":
    main()
