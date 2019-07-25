import argparse
import os

import autokeras
from keras.datasets import cifar10
from sklearn.metrics import classification_report

CIFAR_10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-t', '--time', type=int, help='Max training time (in hours).', required=False)
arguments = vars(argument_parser.parse_args())


# We need to define a main method to avoid threading issues between TensorFlow and Auto-Keras
def main():
    output_path = './output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def hours_to_seconds(hours):
        return hours * 60 * 60

    training_times = map(hours_to_seconds, [1, 2, 4, 8, 12, 24])

    if 'time' in arguments:
        training_times = [hours_to_seconds(arguments['time'])]

    print('[INFO] Loading CIFAR-10 dataset.')
    ((X_train, y_train), (X_test, y_test)) = cifar10.load_data()

    # Now, we need to normalize the data
    X_train = X_train.astype('float') / 255.0
    X_test = X_test.astype('float') / 255.0

    for seconds in training_times:
        print(f'[INFO] Training model for at most {seconds} seconds.')

        classifier = autokeras.ImageClassifier(verbose=True)

        # Trains and tries to find the best architecture.
        classifier.fit(X_train, y_train, time_limit=seconds)

        # Trains the best found architecture.
        classifier.final_fit(X_train, y_train, X_test, y_test, retrain=True)

        print('[INFO] Evaluating model.')
        score = classifier.evaluate(X_test, y_test)
        predictions = classifier.predict(X_test)
        report = classification_report(y_test, predictions, target_names=CIFAR_10_LABELS)

        print('[INFO] Saving report to disk.')
        path = os.path.sep.join([output_path, f'{seconds}.txt'])
        with open(path, 'w') as f:
            f.write(report)
            f.write(f'\nScore: {score}')


if __name__ == '__main__':
    main()
