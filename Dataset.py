#############
# Libraries #
#############

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#################
# Dataset Class #
#################

class Dataset(object):
    """
    Manages dataset options.
    """
    def __init__(self):
        """
        Loads the dataset specified by the user.
        """
        self.on = False
        user_input = input("Which dataset do you want to load?\n>> ")
        try:
            X = pd.read_csv(user_input)
            labels = list(X.columns)
            print()
            while True:
                print("Available labels:\n  {}\n".format(labels))
                label = input("Which label do you want to predict:\n>> ")
                if label not in labels:
                    print("\033[1m\033[91mError. Invalid label.\n\33[0m")
                else:
                    self.ind_values = X
                    self.dep_value = self.ind_values.pop(label)
                    self.ind_labels = list(self.ind_values.columns)
                    self.dep_label = self.dep_value.name
                    self.X = np.array(self.ind_values)
                    self.X = np.insert(self.X, 0, [1.0] * self.X.shape[0], 1)
                    self.Y = np.array(self.dep_value).reshape((-1, 1))
                    self.thetas = np.array([0.0] * (self.X.shape[1])).reshape((-1, 1))
                    self.shape = self.X.shape
                    self.on = True
                    os.system("clear")
                    print("\033[1m\033[92mData successfully loaded :)\n\033[0m")
                    break
        except:
            os.system("clear")
            print("\033[1m\033[91mError. Invalid dataset.\n\33[0m")
        return

    def display_data(self):
        """
        Displays information about the dataset.
        """
        if not self.on:
            os.system("clear")
            print("\033[1m\033[93mCAUTION! Dataset object is not valid :(\n\033[0m")
            return
        while True:
            print("Available labels:\n  {}\n".format(self.ind_labels))
            label = input("Which label do you want to display:\n>> ")
            if label in self.ind_labels:
                break
            print("\033[1m\033[91mError. Invalid label.\n\33[0m")
        os.system("clear")
        data_table = self.ind_values.copy()
        data_table.insert(self.shape[1] - 1, self.dep_label, self.dep_value)
        print(data_table)
        Y_hat = np.matmul(self.X, self.thetas)
        plt.scatter(self.ind_values[label], self.Y, alpha = 0.5, label = "Real Values")
        plt.scatter(self.ind_values[label], Y_hat, alpha = 0.5, label = "Predicted Values")
        plt.ylabel(self.dep_label)
        plt.xlabel(label)
        plt.legend()
        plt.grid()
        plt.show()
        os.system("clear")
        return

    def train(self):
        """
        Trains the model using multivariate linear regression.
        """
        if not self.on:
            os.system("clear")
            print("\033[1m\033[93mCAUTION! Dataset object is not valid :(\n\033[0m")
            return
        first_term = np.linalg.pinv(np.matmul(self.X.transpose(), self.X))
        second_term = np.matmul(self.X.transpose(), self.Y)
        self.thetas = np.matmul(first_term, second_term)
        os.system("clear")
        print("\033[1m\033[92mModel successfully trained :)\n\033[0m")
        return

    def cost(self):
        """
        Computes the cost of the actual model.
        """
        if not self.on:
            os.system("clear")
            print("\033[1m\033[93mCAUTION! Dataset object is not valid :(\n\033[0m")
            return
        Y_hat = np.matmul(self.X, self.thetas).reshape((-1, 1))
        cost = sum((Y_hat - self.Y) ** 2) / (2 * self.shape[0])
        user_input = input("Do you want to plot the cost function? (Yes/No)\n>> ")
        if user_input == "Yes":
            while True:
                print("Available labels:\n  {}\n".format(self.ind_labels))
                label = input("Which label do you want to measure?\n>> ")
                if label in self.ind_labels:
                    break
                print("\033[1m\033[91mError. Invalid label.\n\33[0m")
            cost_table = [list(), list()]
            index = self.ind_labels.index(label) + 1
            count = -50.0
            while count <= 50.1:
                self.thetas[index][0] += count
                Y_hat = np.matmul(self.X, self.thetas).reshape((-1, 1))
                cost_ = sum((Y_hat - self.Y) ** 2) / (2 * self.shape[0])
                cost_table[0].append(self.thetas[index][0])
                cost_table[1].append(cost_)
                self.thetas[index][0] -= count
                count += 0.1
            plt.plot(cost_table[0], cost_table[1])
            plt.scatter(self.thetas[index][0], cost, color = "red")
            plt.xlabel(label)
            plt.ylabel("cost")
            plt.grid()
            print("\nThe current cost for the model is: {}".format(cost))
            plt.show()
        else:
            print("\nThe current cost for the model is: {}\n".format(cost))
            input("Press any key to continue...")
        os.system("clear")
        return

    def predict(self):
        """
        Predicts a new value for the given parameters.
        """
        if not self.on:
            os.system("clear")
            print("\033[1m\033[93mCAUTION! Dataset object is not valid :(\n\033[0m")
            return
        try:
            theta_pos = 1
            output = self.thetas[0][0]
            for item in self.ind_labels:
                user_input = input("Introduce {} value:\n>> ".format(item))
                print()
                output += (self.thetas[theta_pos][0] * float(user_input))
                theta_pos += 1
            print("The predicted value for {} is: {}\n".format(self.dep_label, output))
            input("Press any key to continue...")
            os.system("clear")
        except:
            os.system("clear")
            print("\033[1m\033[91mError. Invalid values.\n\33[0m")
        return
