import numpy as np


# convert data
def read_image(image_path, height, width):
    lines = []
    with open(image_path, 'r') as file:
        for line in file:
            lines.append(line)
    char_mapping = {' ': 0, '+': 0.5, '#': 1}
    arrays = []
    for line in lines:
        if len(line) == width + 1:  # Ensure the line length is correct
            array = np.array([char_mapping[char] for char in line[:-1]])
            arrays.append(array)
    numpy_array = np.array(arrays)
    return numpy_array.reshape((-1, height, width))


def extract_labels(file):
    with open(file, 'r') as file:
        label = [int(line.strip()) for line in file if line.strip()]

    label_array = np.array(label)
    return label_array


# read our stored weights
def extract_face_weights(file):
    with open(file, 'r') as file:
        weight = [float(line.strip()) for line in file if line.strip()]

    weight_array = np.array(weight)
    return weight_array


def extract_digit_weights(file):
    weights_dict = {}
    with open("./saved_weights/digit_perceptron_weights.txt", 'r') as file:
        current_label = None
        for line in file:
            if line.strip():  # Check if line is not empty
                parts = line.split(':')
                if len(parts) == 2:
                    if current_label is not None:
                        weights_dict[current_label] = np.array(current_weights, dtype=float)
                    current_label = int(parts[0].split()[-1])  # Get the label number
                    current_weights = [float(x) for x in parts[1].split()]
                else:
                    # Continuation of weights on the next line
                    current_weights.extend(float(x) for x in line.split())

        # save the last set of weights
        if current_label is not None:
            weights_dict[current_label] = np.array(current_weights, dtype=float)

    return weights_dict
