import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import MNIST
#xgb imports
import xgboost as xgb
import numpy as np
import seaborn as sns
import argparse
import os
from torch.utils.data import TensorDataset
import time


#Global definition for device, transforms, and loaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())
# Check for GPU availability
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# Load MNIST
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
# Define Datalodaer with batch size and shuffle 
batch_size = 10000
num_workers = 0
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=True, pin_memory=True)
    
# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.loss =[]
        self.test_loss = []
        self.test_accuracy = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        return x
    def set_loss(self, loss):
        self._loss.append(loss)
    def get_loss(self):
        return self.loss
    

class LinearLayer(nn.Module):
    def __init__(self, input_size, num_classes): # Input size must match CNN output
        super(LinearLayer, self).__init__()
        self.loss = []
        self.test_loss = []
        self.test_accuracy = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    def set_loss(self, loss):     
        self._loss.append(loss)
    def get_loss(self):
        return self.loss
    
#XGBoost
class XGBoostClassifier:
    def __init__(self, cnn, linear_layer, xgb_params=None, num_boost_round=10):
        self.cnn = cnn
        self.linear_layer = linear_layer
        self.xgb_params = xgb_params if xgb_params else {'max_depth': 3, 'eta': 0.3, 'objective': 'multi:softmax', 'num_class': 10}
        self.num_boost_round = num_boost_round
        self.bst = None

    def train(self, train_data, train_labels):
        dtrain = xgb.DMatrix(data=train_data, label=train_labels) # train_data already contains the required features
        self.bst = xgb.train(self.xgb_params, dtrain, self.num_boost_round)

    def predict(self, data):
        dmatrix = xgb.DMatrix(data=data) # data already contains the features
        predictions = self.bst.predict(dmatrix)
        return predictions
    
def train_models(cnn, linear_layer, xgb_classifier, cnn_epochs, linear_epochs, cnn_optimizer, linear_optimizer, loss_fn):
    cnn = cnn.to(device)
    linear_layer = linear_layer.to(device)

    cnn.train()
    linear_layer.train()

    for epoch in range(cnn_epochs):
        running_loss = 0.0
        correct = 0
        for data, labels in train_loader: # Use train_loader
            data, labels = data.to(device), labels.to(device) # Send directly to device
            cnn_optimizer.zero_grad()
            output = cnn(data)
            loss = loss_fn(output, labels)
            _, preds = torch.max(output, 1)  # Get predicted classes
            correct += torch.sum(preds == labels.data).item() # Accumulate correct predicitions
            running_loss += loss.item() * data.size(0)
            cnn.loss.append(loss.item())
            cnn.train_accuracy.append(correct / len(train_loader.dataset))
            loss.backward()
            cnn_optimizer.step()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        #cnn.loss.append(epoch_loss)
    cnn.eval()

    linear_layer.train()
    linear_layer = linear_layer.to(device)

    with torch.no_grad():
        cnn_outputs_and_labels = []
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            cnn_outputs_and_labels.append( (cnn(data), labels) )


    for epoch in range(linear_epochs):
        running_loss = 0.0
        for cnn_out, labels in cnn_outputs_and_labels:
            linear_optimizer.zero_grad()
            linear_output = linear_layer(cnn_out)
            loss = loss_fn(linear_output, labels)
            _, preds = torch.max(output, 1)  # Get predicted classes
            correct += torch.sum(preds == labels.data).item() # Accumulate correct predicitions
            running_loss += loss.item() * data.size(0)
            linear_layer.train_accuracy.append(correct / len(train_loader.dataset))
            linear_layer.loss.append(loss.item())
            loss.backward()
            linear_optimizer.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        #linear_layer.loss.append(epoch_loss)
    linear_layer.eval()

    #XGB
    xgb_train_input = []
    train_labels_for_xgb = []
    with torch.no_grad():
        for data, labels in train_loader:  # Iterate through the data loader
            data = data.to(device)
            cnn_output = cnn(data) 
            linear_output = linear_layer(cnn_output)
            xgb_train_input.extend(linear_output.cpu().numpy().reshape(linear_output.shape[0], -1).tolist())
            train_labels_for_xgb.extend(labels.cpu().numpy().tolist())
    train_labels_for_xgb = np.array(train_labels_for_xgb)


    xgb_classifier.train(np.array(xgb_train_input), train_labels_for_xgb)

    return cnn, linear_layer, xgb_classifier

def evaluate_pipeline(cnn, linear_layer, xgb_classifier, loss_fn, user_files=None):
    if user_files is not None:
        # Custom Image List Handling
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST normalization
        ])
        tensor_list = [transform(img) for img in user_files]
        custom_dataset = TensorDataset(torch.stack(tensor_list))  
        custom_loader = DataLoader(custom_dataset, batch_size=1) 


        cnn = cnn.to(device).eval()
        linear_layer = linear_layer.to(device).eval()

        all_linear_outputs = [] # initialize list outside of loop
        with torch.no_grad():
            for data in custom_loader:  # Iterate through custom loader
                data = data.to(device) # Custom data needs to be sent to device

                cnn_output = cnn(data)
                linear_output = linear_layer(cnn_output)
                all_linear_outputs.append(linear_output.cpu().numpy().flatten())  # No extra dimensions

        xgb_input = np.array(all_linear_outputs)  # Make numpy array from list
        xgb_predictions = xgb_classifier.predict(xgb_input)

        eval_dictionary = {
            "xgb_predictions": xgb_predictions,  # Return only predictions for custom data
        }

        return eval_dictionary
    else: # Test Evaluation
        cnn = cnn.to(device).eval()
        linear_layer = linear_layer.to(device).eval()

        all_linear_outputs = []
        test_labels_for_eval = []
        
        running_cnn_loss = 0  # Accumulate loss for each batch
        running_linear_loss = 0

        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):  # Use enumerate to get batch index
                data, labels = data.to(device), labels.to(device)

                cnn_output = cnn(data)
                cnn_loss = loss_fn(cnn_output, labels)
                cnn.test_loss.append(cnn_loss.item())
                running_cnn_loss += cnn_loss.item()

                linear_output = linear_layer(cnn_output)
                linear_loss = loss_fn(linear_output, labels)
                linear_layer.test_loss.append(linear_loss.item())
                running_linear_loss += linear_loss.item()
                
                all_linear_outputs.append(linear_output.cpu().numpy())
                test_labels_for_eval.extend(labels.cpu().numpy().tolist()) # Properly append

            xgb_input = np.concatenate(all_linear_outputs, axis=0).reshape(len(test_dataset), -1)
            xgb_predictions = xgb_classifier.predict(xgb_input)


        xgb_input = np.concatenate(all_linear_outputs, axis=0).reshape(len(test_dataset), -1)
        xgb_predictions = xgb_classifier.predict(xgb_input)

        xgb_predictions = xgb_predictions.astype(int)
        test_labels_for_eval = np.array(test_labels_for_eval, dtype=int)

        class_report = classification_report(test_labels_for_eval, xgb_predictions)
        conf_matrix = confusion_matrix(test_labels_for_eval, xgb_predictions)

        # Plotting
        plt.figure(figsize=(16, 8)) # Increased figure size for better visibility

        # Training Loss
        plt.subplot(2, 2, 1)
        plt.plot(cnn.loss, label='CNN Training Loss')
        plt.plot(linear_layer.loss, label='Linear Training Loss')
        plt.xlabel('Sample')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # Training Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(cnn.train_accuracy, label='CNN Training Accuracy')
        plt.plot(linear_layer.train_accuracy, label='Linear Training Accuracy')
        plt.xlabel('Sample')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        # Test Loss
        plt.subplot(2, 2, 3)
        plt.bar(['CNN', 'Linear'], [cnn.test_loss[-1], linear_layer.test_loss[-1]], label='Test Loss', color=['blue', 'orange'])
        plt.ylabel('Loss')
        plt.title('Test Loss')
        plt.legend()

        # Plot Confusion Matrix (using seaborn for better visualization)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=np.unique(test_labels_for_eval), yticklabels=np.unique(test_labels_for_eval))  # Use unique labels for ticks
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
        
        eval_dictionary = {
            "classification_report" : class_report,
            "confusion_matrix" : conf_matrix,
            "cnn_loss" : cnn.loss,
            "linear_layer_loss": linear_layer.loss,
        }

        return eval_dictionary
    
def get_imgs(infiles):
    image_list = []
    if os.path.isfile(infiles):  # Single file - infiles is now a string
        try:
            image = Image.open(infiles).convert('L')
            image = image.resize((28, 28))
            image = transform(image)  # Apply your transform here
            image_list.append(image)
        except Exception as e:
            print(f"Error processing file: {e}")

    elif os.path.isdir(infiles):  # Folder - infiles is now a string
        for filename in os.listdir(infiles):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Check image extensions
                filepath = os.path.join(infiles, filename)
                try:
                    image = Image.open(filepath).convert('L')
                    image = image.resize((28, 28)) # Ensure MNIST size
                    image = transform(image)
                    image_list.append(image)

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")


    else:
        print("Invalid file or folder path provided.")

    return image_list

def main():
    parser = argparse.ArgumentParser(description="CNN-Linear-XGBoost Pipeline")
    parser.add_argument("--cnn_epochs", type=int, default=10, help="Number of epochs to train CNN (default: 10)")
    parser.add_argument("--linear_epochs", type=int, default=10, help="Number of epochs to train linear layer (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.01)")
    args = parser.parse_args()

    # Default values
    cnn_epochs = args.cnn_epochs
    linear_epochs = args.linear_epochs
    loss_fn = nn.CrossEntropyLoss()  # Default loss function

    # Initialize models and optimizers
    cnn = CNN().to(device)
    linear_layer = LinearLayer(input_size=128, num_classes=10).to(device) 
    xgb_classifier = XGBoostClassifier(cnn, linear_layer)

    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate) # Default optimizer for CNN
    linear_optimizer = torch.optim.Adam(linear_layer.parameters(), lr=args.learning_rate) # Default optimizer for Linear Layer

    while True:
        print("\nMenu:")
        print("1. Train on MNIST")
        print("2. Test user image")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            start_time = time.time()
            print("Training started")
            cnn, linear_layer, xgb_classifier = train_models(cnn, linear_layer, xgb_classifier, cnn_epochs, linear_epochs, cnn_optimizer, linear_optimizer, loss_fn)  # Pass data loaders
            print("Training complete, time taken: %s seconds" % (time.time() - start_time))
            eval_results = evaluate_pipeline(cnn, linear_layer, xgb_classifier, loss_fn) # Pass data loaders
            print(eval_results["classification_report"])


        elif choice == '2':
            try:
                image_path = input("Enter path to image file: ")
                image_list = get_imgs(image_path)
                transformed_images = [transform(img) for img in image_list] #Transform PIL images to tensors
                image_tensor = torch.stack(transformed_images)
                image_tensor = image_tensor.to(device)
                eval_results = evaluate_pipeline(cnn, linear_layer, xgb_classifier, loss_fn=loss_fn, user_files=image_tensor)
                print("Predictions for user image:", eval_results["xgb_predictions"])
            except Exception as e: # Catch file errors, etc
                print(f"Error processing image: {e}")


        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()