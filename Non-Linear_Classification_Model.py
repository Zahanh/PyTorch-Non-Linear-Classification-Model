import torch
from torch import nn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred))*100

## Setting the hyperparameters:
numClass = 4
numFeat = 2
randomSeed = 42


# Creating the multi-class data
xblob, yblob = make_blobs(n_samples=1000,n_features=numFeat,centers=numClass,cluster_std=1.5,random_state=randomSeed)

# Turning data into tensors
xblob = torch.from_numpy(xblob).type(torch.float)
yblob = torch.from_numpy(yblob).type(torch.long)

# Splitting into training/testing data
xb_train, xb_test, yb_train, yb_test = train_test_split(xblob,yblob,test_size=0.2,random_state=randomSeed)

## Plotting data
plt.figure(figsize=(10,7))
plt.scatter(xblob[:,0],xblob[:,1],c=yblob,cmap=plt.cm.RdYlBu)

## Create device agnositic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Building the multi-class classification model
class blobModel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units=8) -> None:
        super().__init__()
        self.linear_layer = nn.Sequential(nn.Linear(in_features=input_features,out_features=hidden_units),
                                          nn.ReLU(),
                                          nn.Linear(in_features=hidden_units,out_features=hidden_units),
                                          nn.ReLU(),
                                          nn.Linear(in_features=hidden_units,out_features=output_features))
    def forward(self,x):
        return self.linear_layer(x)

# print(torch.unique(yblob))
model = blobModel(2,4).to(device)
# print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

## Fitting the multiclass model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## Set the number of epochs
epochs = 100

## Putting Data to teh target device
xb_train,yb_train = xb_train.to(device),yb_train.to(device)
xb_test,yb_test = xb_test.to(device),yb_test.to(device)

## Loop through the data
for epoch in range(epochs):
    model.train()
    y_logits = model(xb_train)
    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, yb_train)
    accuracy = accuracy_fn(y_true=yb_train,y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ## Testing Code
    model.eval()
    with torch.inference_mode():
        test_logits = model(xb_test)
        test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits,yb_test)
        test_acc = accuracy_fn(y_true=yb_test,y_pred=test_preds)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}, Acc: {accuracy:.2f}  | Test loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}")


## Making predicitions
model.eval()
with torch.inference_mode():
    y_logits = model(xb_test)

pred_probs = torch.softmax(y_logits,dim=1)
# print(pred_probs[:10])

pred_labels = torch.argmax(pred_probs,dim=1)
# print(pred_labels[:10])


## Plotting data 
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model,xb_train,yb_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model,xb_test,yb_test)
