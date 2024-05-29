import torch
import generativeClassification.vae_cifar as vae_cifar
from generativeClassification.smallnet import LinearNet, Net, split_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision import transforms, datasets
from torch import nn, optim
from accelerate import Accelerator
from generativeClassification.vqvae import Model
import argparse

PTH = 'vqvae_cifar.pth'
DEVICE = 'cuda'
SIZE_PER_CLASS = 2
EPOCHS = 90
HEAD = 'NonLin'
NUM_CLASSES = 10
DATASET = 'CIFAR10'

activations_encoder = {
    'layer1' : None,
    'layer2' : None,
    'layer3' : None,
    'layer4' : None,
    'layer5': None,
    'layer6': None,
    'layer7': None
}

activations_decoder = {
    'layer1' : None,
    'layer2' : None,
    'layer3' : None,
    'layer4' : None,
    'layer5' : None,
    'layer5' : None,
}

def get_activation_foo(name, activations):
    def hookFoo(model, input, output):
        # if activations[name] == None:
        activations[name] = output.detach()
        # else:
        #     activations[name] = torch.cat((activations[name], output.detach()), 0)
    return hookFoo

def get_activation_foo_input(name, activations):
    def hookFoo(model, input, output):
        # if activations[name] == None:
        activations[name] = input[0].detach()
        # else:
        #     activations[name] = torch.cat((activations[name], input[0].detach()), 0)
    return hookFoo

def transform_to_features(activations, batch_size):
    with torch.no_grad():
        feats = torch.Tensor([]).to(DEVICE)
        for key in activations.keys():
            val = activations[key]
            cur_feat = val[-batch_size:]
            if len(val.shape) == 4:
                cur_feat = val[-batch_size:].mean(dim=[2,3])

            feats = torch.cat((feats.to(DEVICE), cur_feat.to(DEVICE)), dim=1)
            activations[key] = None
    return feats

def train_model(train_loader, model, criterion, optimizer, model_vae, activations, epochs=90, loss_list=[], device='mps'):
    if activations == 'encoder':
        activations = activations_encoder
    elif activations == 'decoder':
        activations = activations_decoder

    model.train()
    loss_list = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            with torch.no_grad():
                model_vae(images)
            # print(activations_decoder['layer1'].shape)
            if activations != 'mix':
                features_enc = transform_to_features(activations, labels.shape[0])
            else:
                features_enc = transform_to_features(activations_encoder, labels.shape[0])
                features_dec = transform_to_features(activations_decoder, labels.shape[0])
                features_enc = torch.cat((features_enc, features_dec), dim=1)

            # print(features_enc)
            features_enc = features_enc.to(device)
            outputs = model(features_enc)
            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        loss_list.append(running_loss / len(train_loader))
#         print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
    return loss_list


def test_model(model, test_loader, model_vae, activations, device='mps'):
    if activations == 'encoder':
        activations = activations_encoder
    elif activations == 'decoder':
        activations = activations_decoder

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                model_vae(images)
            if activations != 'mix':
                features_enc = transform_to_features(activations, labels.shape[0])
            else:
                features_enc = transform_to_features(activations_encoder, labels.shape[0])
                features_dec = transform_to_features(activations_decoder, labels.shape[0])
                features_enc = torch.cat((features_enc, features_dec), dim=1)

            features_enc.to(device)
            model.to(device)

            outputs = model(features_enc)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")
    return accuracy

def split(dataset, num_train_per_class, num_test_per_class):
    train_indices = []
    test_indices = []
    for i in range(NUM_CLASSES):
        indices = torch.where(torch.tensor(dataset.targets) == i)[0].tolist()
        train_indices.extend(indices[:num_train_per_class])
        test_indices.extend(indices[num_train_per_class:num_train_per_class+num_test_per_class])
    return train_indices, test_indices

def main():
    accelerator = Accelerator()
    model = Model()
    model.load_state_dict(torch.load(PTH, map_location=torch.device(accelerator.device)))

    model.conv1.register_forward_hook(get_activation_foo('layer1', activations_encoder))
    model.conv2.register_forward_hook(get_activation_foo('layer2', activations_encoder))
    model.conv3.register_forward_hook(get_activation_foo('layer3', activations_encoder))
    model.resblock1.layers[0].register_forward_hook(get_activation_foo('layer4', activations_encoder))
    model.resblock1.layers[1].register_forward_hook(get_activation_foo('layer5', activations_encoder))
    model.resblock1.layers[2].register_forward_hook(get_activation_foo('layer6', activations_encoder))
    model.vq_conv.register_forward_hook(get_activation_foo('layer7', activations_encoder))

    model.conv4.register_forward_hook(get_activation_foo_input('layer1', activations_decoder))
    model.resblock2.layers[0].register_forward_hook(get_activation_foo_input('layer2', activations_decoder))
    model.resblock2.layers[1].register_forward_hook(get_activation_foo_input('layer3', activations_decoder))
    model.resblock2.layers[2].register_forward_hook(get_activation_foo_input('layer4', activations_decoder))
    model.conv5.register_forward_hook(get_activation_foo_input('layer5', activations_decoder))
    model.conv6.register_forward_hook(get_activation_foo_input('layer6', activations_decoder))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    train_dataset = None
    test_dataset = None
    if DATASET == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        NUM_CLASSES = 100
    else:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        NUM_CLASSES = 10

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_indices, _ = split(train_dataset, num_train_per_class=SIZE_PER_CLASS, num_test_per_class=SIZE_PER_CLASS)


    train_subset = Subset(train_dataset, train_indices)
    train_sub_loader = DataLoader(train_subset, batch_size=64, shuffle=False)

    test_iter = iter(test_loader)
    cur_iter = next(test_iter)

    model(cur_iter[0][:4])[1].to(DEVICE).detach()

    num_encoder_features = transform_to_features(activations_encoder, 64).shape[1]
    num_decoder_features = transform_to_features(activations_decoder, 64).shape[1]
    num_all_features = num_encoder_features + num_decoder_features

    num_classes = NUM_CLASSES

    linModel = None
    if HEAD == 'NonLin':
        linModel = Net(num_all_features, num_classes).to(accelerator.device)
    else:
        linModel = LinearNet(num_all_features, num_classes).to(accelerator.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linModel.parameters(), lr=0.001)

    linModel.to(accelerator.device)
    model.to(accelerator.device)

    train_model(train_sub_loader, linModel, criterion, optimizer, model, activations='mix', epochs=EPOCHS, loss_list=[],
                device=accelerator.device)
    test_model(linModel, test_loader, model, activations='mix', device=accelerator.device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Path of weights', default=PTH)
    parser.add_argument('-d', '--device', type=str, help='Device for training', default=DEVICE)
    parser.add_argument('-s', '--size_per_class', type=int, help='Number of images per class', default=SIZE_PER_CLASS)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=EPOCHS)
    parser.add_argument('-hd', '--head', type=str, help='Type of head model: Lin or NonLin', default=HEAD)
    parser.add_argument('-nc', '--num_classes', type=int, help='Num Classes', default=NUM_CLASSES)
    parser.add_argument('-ct', '--cifar_type', type=str, help='CIFAR10 or CIFAR100', default=DATASET)

    args = parser.parse_args()

    PTH = args.path
    DEVICE = args.device
    SIZE_PER_CLASS = args.size_per_class
    EPOCHS = args.epochs
    NUM_CLASSES = args.num_classes
    DATASET = args.cifar_type

    main()