import pandas as pd
from torchvision import transforms, models
import numpy as np
from dataset import DigitsDataset
from torch.utils.data import DataLoader
from cnn import CNN
import torch as T


if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    batch_size = 64
    n_iterations = 250
    n_epochs = 10
    learning_rate = 0.01

    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, ), std = (0.5, ))
    ])

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6}

    train_split = 0.7
    n_rows = len(train)
    len_training = int(n_rows * train_split)

    train_indexes = np.random.choice(len(train), size = len_training, replace = False)

    train_dataset = DigitsDataset(train.iloc[train_indexes], transform = transformation)
    training_generator = DataLoader(train_dataset, **params)

    validation_dataset = DigitsDataset(train.iloc[~train_indexes], transform = transformation)
    validation_generator = DataLoader(validation_dataset, **params)

    test.insert(0, 'label', [-1]*test.shape[0])
    test_dataset = DigitsDataset(test, transform=transformation, shuffle = False)
    testing_generator = DataLoader(test_dataset)

    model = CNN(alpha = learning_rate, batch_size = batch_size)

    # model.train()

    # for i in range(n_epochs):
    #     print(i)
    #     for images, labels in training_generator:
    #         images = images.to(model.device)
    #         labels = labels.to(model.device)
    #         model.optimizer.zero_grad()
    #         label_predicted = model.forward(images)
    #         loss = model.loss(label_predicted, labels).to(model.device)
    #         loss.backward()
    #         model.optimizer.step()
    #
    #     for images, labels in validation_generator:
    #         images = images.to(model.device)
    #         labels = labels.to(model.device)
    #         label_predicted = T.argmax(model.forward(images), dim = 1)
    #         n_correct_labels =  T.sum(labels.eq(label_predicted))
    #         print(f"Accuracy: {n_correct_labels*100.0/len(labels)}")
    #
    # model.save("checkpoint.pt")

    model.load("checkpoint.pt")
    model.eval()
    submission = []
    for images, labels in testing_generator:
        images = images.to(model.device)
        labels = labels.to(model.device)
        label_predicted = T.argmax(model.forward(images), dim=1)
        submission.append(label_predicted.item())

    df = pd.DataFrame.from_dict(submission)
    df.index += 1
    df.to_csv("submission.csv", header = ["Label"])


