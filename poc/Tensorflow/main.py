import random
import sys

import tensorflow as tf
import pandas as pd
import numpy as np


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def train_model(choice):
    dataset = pd.read_csv("./test_data/numbers.csv")
    even_target = dataset.pop(choice)
    feat_name = ['Number1', 'Number2', 'Number3']
    train_feature = dataset[feat_name]
    real_dataset = tf.data.Dataset.from_tensor_slices((train_feature.values, even_target.values))
    real_dataset_shuffled = real_dataset.shuffle(len(train_feature)).batch(1)
    model = get_compiled_model()
    model.fit(real_dataset_shuffled, epochs=4)
    return model


def train_model_choice(choice):
    if choice == 1:
        return train_model("isEven")
    elif choice == 2:
        return train_model("isMultipleOf3")
    elif choice == 3:
        return train_model("isMultipleOf5")
    else:
        return


def generate_data(data_number):
    file_numbers = open(r"./test_data/numbers.csv", "w")
    file_numbers.write('"Number1","Number2","Number3","isEven","isMultipleOf3","isMultipleOf5"\n')
    for i in range(0, data_number):
        nb = random.randint(0, 100000)
        nb2 = random.randint(0, 100000)
        nb3 = random.randint(0, 100000)
        is_even = nb % 2 == 0 and nb2 % 2 == 0 and nb3 % 2 == 0
        is_multiple_of_3 = nb % 3 == 0 and nb2 % 3 == 0 and nb3 % 3 == 0
        is_multiple_of_5 = nb % 5 == 0 and nb2 % 5 == 0 and nb3 % 5 == 0

        line = str(nb) + ',' + str(nb2) + ',' + str(nb3) + ',' + str(is_even) + ',' + str(is_multiple_of_3) + ',' + str(
            is_multiple_of_5) + '\n'
        file_numbers.write(line)
    file_numbers.close()


def main():
    inp = "0"
    while inp != "1" and inp != "2" and inp != "3":
        inp = input("What do you want?\n1 : Generate random numbers (will take disk space)\n2 : Train AI to do "
                    "stuff\n3 : Quit\n> ")

    if inp == "1":
        print("----- Generating Data -----")
        generate_data(2000000)
        print("----- End of data generation -----")

    if inp == "2":
        inp = "0"
        while inp != "1" and inp != "2" and inp != "3" and inp != "4":
            inp = input(
                "What do you want?\n1 : Train to detect series of even numbers\n2 : detect multiple of 3\n3 : detect "
                "multiple of 5\n4 : Quit\n> ")
        print("----- Starting Neural Network -----")
        train_model_choice(int(inp))


if __name__ == '__main__':
    main()
