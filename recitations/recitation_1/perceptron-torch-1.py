import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],dtype=torch.float32).to(device)

majority = torch.tensor([0,0,0,1,0,1,1,1],dtype=torch.float32)
xor = torch.tensor([0,1,1,0,1,0,0,0],dtype=torch.float32)
onehotnot = torch.tensor([1,1,1,1,0,0,0,0],dtype=torch.float32)

class Perceptron(nn.Module):
    def __init__(self, input_size, activation_function):
        nn.Module.__init__(self)
        self.activation_function = activation_function
        self.l1 = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.l1(x)
        out = self.activation_function(out)
        return out


#activation_function = nn.ReLU()
#activation_function = nn.Sigmoid()
activation_function = nn.Tanh()

model = Perceptron(input_size=3, activation_function=activation_function).to(device)
criterion = nn.MSELoss()
num_epochs = 100
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def non_batched_learning(labels):
    for epoch in range(num_epochs):
        i = 0
        for input in inputs:
            output = model(input).to(device)
            label = torch.tensor([labels[i]]).to(device)
            i+=1
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}], Loss: {loss.item():.4f}')

def batched_learning(labels):

    split_labels = []
    for label in labels:
        split_labels.append([label])
    split_labels = torch.tensor(split_labels)

    for epoch in range(num_epochs*8):
        output = model(inputs)
        loss = criterion(output,split_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'Epoch [{epoch + 1}/{num_epochs}]], Loss: {loss.item():.4f}')

def show():
    with torch.no_grad():
        for input in inputs:
            output = model(input)
            print(f"{input}->{output}")

non_batched_learning(majority)
#batched_learning(majority)

show()