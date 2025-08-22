import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def normalize_numeric_data(numeric_data):
    min_vals = np.min(numeric_data, axis=0)
    max_vals = np.max(numeric_data, axis=0)

    normalized_data = (numeric_data - min_vals) / (max_vals - min_vals)

    return normalized_data


def split_data_into_test_train(file_path, test_ratio=0.2):
    data = pd.read_csv(file_path)

    classes = np.array(data.iloc[:, 4].values)

    ignore_columns = [0, 1, 4, data.shape[1] - 1]
    numeric_columns = [i for i in range(8, data.shape[1] - 1) if i not in ignore_columns]
    numeric_data = data.iloc[:, numeric_columns].values

    numeric_data_normalized = normalize_numeric_data(numeric_data)

    train_attributes, test_attributes, train_classes, test_classes = train_test_split(
        numeric_data_normalized, classes, test_size=test_ratio, random_state=42
    )

    return train_attributes, test_attributes, train_classes, test_classes


def initialize_parameters(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
    biases_hidden = np.random.uniform(-0.5, 0.5, (1, hidden_size))
    weights_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
    biases_output = np.random.uniform(-0.5, 0.5, (1, output_size))
    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


def forward_propagation(atributes, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
    hidden_raw_input = np.dot(atributes, weights_input_hidden) + biases_hidden
    activated_hidden = sigmoid(hidden_raw_input)

    hidden_raw_output = np.dot(activated_hidden, weights_hidden_output) + biases_output
    activated_output = softmax(hidden_raw_output)

    return activated_hidden, activated_output


def backward_propagation(atributes, hidden_layer, output_layer, expected_output, weights_hidden_output,
                         weights_input_hidden, biases_output, biases_hidden, learning_rate=0.001):
    output_error = output_layer - expected_output
    output_delta = output_error

    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

    weights_hidden_output -= learning_rate * np.dot(hidden_layer.T, output_delta)
    biases_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    weights_input_hidden -= learning_rate * np.dot(atributes.T, hidden_delta)
    biases_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output


def train_neural_network(x_train, y_train, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.001):
    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = initialize_parameters(input_size,
                                                                                                      hidden_size,
                                                                                                      output_size)
    losses = []
    for epoch in range(epochs):
        hidden_layer, output_layer = forward_propagation(x_train, weights_input_hidden, biases_hidden,
                                                         weights_hidden_output, biases_output)

        loss = cross_entropy_loss(y_train, output_layer)
        losses.append(loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = backward_propagation(
            x_train, hidden_layer, output_layer, y_train, weights_hidden_output, weights_input_hidden, biases_output,
            biases_hidden, learning_rate
        )

    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, losses


def predict(x_test, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
    _, output_layer = forward_propagation(x_test, weights_input_hidden, biases_hidden, weights_hidden_output,
                                          biases_output)
    return np.argmax(output_layer, axis=1)


def training_loss_plot(epochs, losses):
    plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()


def visualize_misclassified(data, true_labels, predicted_labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_data = tsne.fit_transform(data)
    misclassified = true_labels != predicted_labels
    correctly_classified = ~misclassified
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data[correctly_classified, 0], tsne_data[correctly_classified, 1], c='green',
                label='Instante clasificate corect', alpha=0.5)
    plt.scatter(tsne_data[misclassified, 0], tsne_data[misclassified, 1], c='red', label='Instante clasificate eronat',
                alpha=0.7)
    plt.xlabel('Atribut 1')
    plt.ylabel('Atribut 2')
    plt.title('Distributia punctelor clasificate corect si incorect')
    plt.legend()
    plt.show()


def predict_custom_input(custom_input, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output,
                         unique_classes):
    _, output_layer = forward_propagation(custom_input, weights_input_hidden, biases_hidden, weights_hidden_output,
                                          biases_output)

    predicted_class_index = np.argmax(output_layer, axis=1)

    predicted_breed = unique_classes[predicted_class_index]

    return predicted_breed


def neural_network_training(custom_input):
    file_path = "./datasets/Numeric_Dataset.csv"

    x_train, x_test, y_train, y_test = split_data_into_test_train(file_path,
                                                                  test_ratio=0.2)
    unique_classes = np.unique(y_train)
    y_train_one_hot = np.eye(len(unique_classes))[np.searchsorted(unique_classes, y_train)]
    y_test_one_hot = np.eye(len(unique_classes))[np.searchsorted(unique_classes, y_test)]

    input_size = x_train.shape[1]
    hidden_size = 50
    output_size = y_train_one_hot.shape[1]

    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, losses = train_neural_network(
        x_train, y_train_one_hot, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.001
    )
    predictions = predict(x_test, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)

    # training_loss_plot(1000, losses)

    predicted_labels = predictions
    true_labels = np.argmax(y_test_one_hot, axis=1)

    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Accuracy on the test set: {accuracy:.2%}")

    # visualize_misclassified(x_test, true_labels, predicted_labels)

    predicted_class = predict_custom_input(custom_input, weights_input_hidden, biases_hidden, weights_hidden_output,
                                           biases_output, unique_classes)

    print(f"Predicted class for the custom input: {predicted_class}")

    return predicted_class
