import matplotlib.pyplot as plt
from fdperceptron_driver import *

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# FACE ACCURACY PLOT
# face_percentage_accuracies, face_percentages, face_mean_acc, face_std_acc
ax[0].plot(face_percentages, face_percentage_accuracies,
           marker='o', linestyle='-', color='b')
ax[0].axhline(y=face_mean_acc,
              color='g', linestyle='--')
ax[0].axhline(y=face_mean_acc + face_std_acc,
              color='y', linestyle='--')
ax[0].axhline(y=face_mean_acc - face_std_acc,
              color='y', linestyle='--')
ax[0].axhline(y=face_test_accuracy,
              color='r', linestyle='--')
ax[0].set_title("Face Recognition Accuracy with Mean and Std. Dev.")
ax[0].set_xlabel("% of Data Used for Training")
ax[0].set_ylabel("Accuracy (%)")
ax[0].grid(True)

# Face Text Labels
ax[0].text(1.0, face_mean_acc, 'Train Acc',
           verticalalignment='bottom', horizontalalignment='left')
ax[0].text(1.0, face_mean_acc + face_std_acc, 'Train +Std. Dev',
           verticalalignment='bottom', horizontalalignment='right')
ax[0].text(1.0, face_mean_acc - face_std_acc, 'Train -Std. Dev',
           verticalalignment='top', horizontalalignment='right')
ax[0].text(1.0, face_test_accuracy, 'Test Acc.',
           verticalalignment='bottom', horizontalalignment='right')


# DIGIT ACCURACY PLOT
# digit_percentage_accuracies, digit_percentages, digit_mean_acc, digit_std_acc
ax[1].plot(digit_percentages, digit_percentage_accuracies,
           marker='o', linestyle='-', color='b')
ax[1].axhline(y=digit_mean_acc,
              color='g', linestyle='--')
ax[1].axhline(y=digit_mean_acc + digit_std_acc,
              color='y', linestyle='--')
ax[1].axhline(y=digit_mean_acc - digit_std_acc,
              color='y', linestyle='--')
ax[1].axhline(y=digit_test_accuracy,
              color='r', linestyle='--')
ax[1].set_title("Digit Recognition Accuracy with Mean and Std. Dev.")
ax[1].set_xlabel("% of Data Used for Training")
ax[1].set_ylabel("Accuracy (%)")
ax[1].grid(True)

# Digit Text Labels
ax[1].text(1.0, digit_mean_acc, 'Train Acc',
           verticalalignment='bottom', horizontalalignment='right')
ax[1].text(1.0, digit_mean_acc + digit_std_acc, 'Train +Std. Dev',
           verticalalignment='bottom', horizontalalignment='right')
ax[1].text(1.0, digit_mean_acc - digit_std_acc, 'Train -Std. Dev',
           verticalalignment='top', horizontalalignment='right')
ax[1].text(1.0, digit_test_accuracy, 'Test Acc.',
           verticalalignment='bottom', horizontalalignment='right')


# Display plot
plt.tight_layout()
plt.show()
