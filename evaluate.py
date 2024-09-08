from data_loading import get_data
import torch
from collections import Counter
from train_script import CNNModel 

def evaluate_images(cnn, images):
    cnn.eval()
    outputs = cnn(images)
    _, predicted = torch.max(outputs, 1)
    return predicted 

def load_model(path):
    cnn = CNNModel()
    cnn.load_state_dict(torch.load(path, weights_only=True))
    return cnn

def percent_correct_predictions(predicted, labels):
    correct_count = 0
    for i in range(len(predicted)):
        if predicted[i].item() == labels[i].item():
            correct_count += 1
    return correct_count / len(predicted) * 100

def precent_correct_predictions_per_class(classes, predicted, labels):
    correct_count = [0] * len(classes)
    nb_classes = Counter(labels.tolist())
    for i in range(len(predicted)):
        if predicted[i].item() == labels[i].item():
            correct_count[predicted[i].item()] += 1
    nb_classes = Counter(labels.tolist())
    return ["{} : {:.2f}".format(classes[i], correct_count[i] / nb_classes[i] * 100)  for i in range(len(nb_classes)) if nb_classes[i] != 0]

if __name__ == "__main__":
    dataset_path = "./Vegetable Images"
    dataset_type = "/test"
    testdata_loader, classes = get_data(dataset_path, dataset_type)
    path = "test_classification_model.pth"
    cnn = load_model(path)
    images, labels = next(iter(testdata_loader))
    predicted = evaluate_images(cnn, images)
    print("Overall percentage of correct predictions:")
    print(percent_correct_predictions(predicted, labels))
    print("Percentage of correct predictions for each class:")
    print(precent_correct_predictions_per_class(classes, predicted, labels))