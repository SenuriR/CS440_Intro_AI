from fdutil import *
from fdperceptron_models import *

print("\nWelcome to FDPerceptron run!")

# Setup for face images
face_train_data = read_image("./data/facedata/facedatatrain", 70, 60)
face_train_label = extract_labels("./data/facedata/facedatatrainlabels.txt")
face_test_data = read_image("./data/facedata/facedatatest", 70, 60)
face_test_label = extract_labels("./data/facedata/facedatatestlabels.txt")
face_validation_data = read_image("./data/facedata/facedatavalidation", 70, 60)
face_validation_label = extract_labels("./data/facedata/facedatavalidationlabels.txt")
print("Shape of Face train data array:", face_train_data.shape)
print("Shape of Face train label array:", face_train_label.shape)

# train the face perceptron
face_perceptron = FDPerceptronFace(num_features=4200, rate=0.01)
face_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Percentages of the data to use for training
face_percentage_accuracies = []  # List to store the accuracies for mean_acc and std_acc calculations
for percentage in face_percentages:
    sample_size = int(len(face_train_data) * percentage)
    face_perceptron.train(face_train_data[:sample_size],
                          face_train_label[:sample_size],
                          face_validation_data[:sample_size],
                          face_validation_label[:sample_size],
                          100, 3.2)
    face_accuracy = face_perceptron.accuracy(face_train_data[:sample_size], face_train_label[:sample_size])
    face_percentage_accuracies.append(face_accuracy)
    print(f"Training with {percentage * 100}% of data: Accuracy = {face_accuracy:.2f}%")

face_mean_acc = np.mean(face_percentage_accuracies)
face_std_acc = np.std(face_percentage_accuracies)
# print(f"\n------------------")
print(f"Mean Accuracy for FACE training data: {face_mean_acc:.2f}%")
print(f"Standard Deviation for FACE training data: {face_std_acc:.2f}%")

face_perceptron.export_face_weights()
# test_accuracy = per_face.accuracy(X_test, Y_test)
# print(f"Testing with test data: Test Accuracy = {test_accuracy:.2f}%")

# face data
face_test = read_image("./data/facedata/facedatatest", 70, 60)
face_labels = extract_labels("./data/facedata/facedatatestlabels.txt")
face_perceptron = FDPerceptronFace(num_features=4200, rate=0.01)
face_perceptron.weights = extract_face_weights("./fd_perceptron_face_weights.txt")
face_test_accuracy = face_perceptron.accuracy(face_test, face_labels)
print(f"Face Test Accuracy = {face_test_accuracy:.2f}%")

print(f"\n===============************====================")

# ### DIGIT IMAGES ###

# Setup for digit images
digit_train_data = read_image("./data/digitdata/trainingimages", 28, 28)
digit_train_label = extract_labels("./data/digitdata/traininglabels.txt")
print("Shape of digit train data array:", digit_train_data.shape)
print("Shape of digit train label array:", digit_train_label.shape)

# train the Digit perceptron
digit_perceptron = FDPerceptronDigit(num_features=784, rate=0.001)
digit_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Percentages of the data to use for training
digit_percentage_accuracies = []  # List to store the accuracies for digit_mean_acc and digit_std_acc calculations
for percentage in digit_percentages:
    sample_size = int(len(digit_train_data) * percentage)
    digit_perceptron.train(digit_train_data[:sample_size],
                           digit_train_label[:sample_size],
                           100)
    digit_accuracy = digit_perceptron.accuracy(digit_train_data[:sample_size], digit_train_label[:sample_size])
    digit_percentage_accuracies.append(digit_accuracy)
    print(f"Training with {percentage * 100}% of data: Accuracy = {digit_accuracy:.2f}%")

digit_mean_acc = np.mean(digit_percentage_accuracies)
digit_std_acc = np.std(digit_percentage_accuracies)
print(f"Mean Accuracy for Digit training data: {digit_mean_acc:.2f}%")
print(f"Standard Deviation for Digit training data: {digit_std_acc:.2f}%")

digit_perceptron.export_digit_weights()

digit_test_data = read_image("./data/digitdata/testimages", 28, 28)
digit_test_label = extract_labels("./data/digitdata/testlabels.txt")

digit_test_accuracy = digit_perceptron.accuracy(digit_test_data, digit_test_label)
print(f"Test Digit Accuracy = {digit_test_accuracy:.2f}%")

# test_digit_accuracy = per_digit.accuracy(x_test_digit, y_test_digit)
# print(f"Testing with test data: Test Accuracy = {test_accuracy:.2f}%")
# digit_perceptron = FDPerceptronDigit(num_features=784, rate=0.001, num_classes=10)
# digit_weights_dict = extract_digit_weights("./fd_perceptron_digit_weights.txt")
# for label in digit_weights_dict:
#     digit_perceptron.weights[label] = digit_weights_dict[label]
#
# digit_test_accuracy = digit_perceptron.accuracy(digit_test_data, digit_test_label)
# print(f"Test Digit Accuracy = {digit_test_accuracy:.2f}%")

print("\nExit  FDPerceptron run!")
