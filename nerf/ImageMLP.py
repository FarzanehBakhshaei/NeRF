import torch.nn as nn

class ImageMLP(nn.Module): # nn.Moule is the base class for every model and layer in PyTorch
    def __init__(self, input_dim = 2, num_hidden_layer_nodes=256, num_hidden_layers=4):
        # original paper (Mildenhall et al., 2020) uses 8 layers of MLP with 256 nodes for each hidden layer
        # however I use 4 MLP layers with 256 nodes in hidden layers for this simple example
        super().__init__()
        layers = [nn.Linear(input_dim, num_hidden_layer_nodes), nn.ReLU()] # first fully connected layer with ReLU its activision function, 2 -> 256
        for i in range(1, num_hidden_layers): # nn.linear(in, out). layers 2nd and 3rd and 4th: 256 -> 256
            layers += [nn.Linear(num_hidden_layer_nodes, num_hidden_layer_nodes), nn.ReLU()]

        layers += [nn.Linear(num_hidden_layer_nodes, 3), nn.Sigmoid()] # output layer. 256 -> 3rgb. also the sigmoid for rgb between 0 and 1

        self.net = nn.Sequential(*layers) # Sigmoid(layers[4](ReLU(layers[3](ReLU(layers[2](ReLU(layers[1](ReLU(layers[0](point))))))))))


    # __init built the layers, and now forward uses them. use this function by calling model(coords)
    def forward(self, coords):
        return self.net(coords)