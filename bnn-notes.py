import torch
import torch.nn as nn
import torch.nn.functional as F


# class for neural network object
class TumorNet(nn.Module):
    def __init__(self):
        super(TumorNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # extract 16 features through scanning a 3 x 3 across the image
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # extract 32 features through scanning a 3 x 3 across the image
        self.pool = nn.MaxPool2d(2, 2) # take the max of the 2x2 

        self.fc1 = nn.Linear(32 * 24 * 24, 128) # passes the 96 x 96 to a 128 hidden layer
        self.fc2 = nn.Linear(128, 1) # passes the outputs of the 128 hidden layer to a yes or no statement

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 96 → 48
        x = self.pool(F.relu(self.conv2(x)))   # 48 → 24

        x = x.view(x.size(0), -1) # flatten the image to 1D

        x = F.relu(self.fc1(x)) # pass the output to the hidden layer
        x = torch.sigmoid(self.fc2(x))  # binary output

        return x
    

def train(model, dataloader, epochs=5, lr=1e-3):
    # check if cuda is being used, if not use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize loss function and training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()

    # train model for certain cycles and compute loss after each training cycle
    for epoch in range(epochs):
        total_loss = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

def test(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    correct = 0
    total = 0

    # no backtesting 
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

def __main__():
    model = TumorNet()

    # when data set is ready

    train(model, train_loader, epochs=10)
    accuracy = test(model, test_loader)

__main__()