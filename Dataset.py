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
                    self.X = X
                    self.Y = self.X.pop(label)
                    self.shape = self.X.shape
                    self.thetas = pd.DataFrame([0.0] * (self.shape[1] + 1), columns = ["Thetas"])
                    self.X.insert(0, "None", [1.0] * self.shape[0])
                    self.labels = list(self.X.columns)
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
            print("Available labels:\n  {}\n".format(self.labels[1:]))
            label = input("Which label do you want to display:\n>> ")
            if label in self.labels[1:]:
                break
            print("\033[1m\033[91mError. Invalid label.\n\33[0m")
        os.system("clear")
        data_table = self.X.copy()
        data_table.insert(self.shape[1] + 1, self.Y.name, self.Y)
        print(data_table)
        X = np.array(self.X)
        thetas = np.array(self.thetas).reshape((-1, 1))
        Y_hat = np.matmul(X, thetas).reshape((-1, 1))
        plt.scatter(self.X[label], self.Y, alpha = 0.5, label = "Real Values")
        plt.scatter(self.X[label], Y_hat, alpha = 0.5, label = "Predicted Values")
        plt.xlabel(label)
        plt.ylabel(self.Y.name)
        plt.grid()
        plt.legend()
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
        thetas = np.array(self.thetas).reshape((-1, 1))
        X = np.array(self.X)
        Y = np.array(self.Y).reshape((-1, 1))
        alpha = 0.01
        iterations = 0
        while iterations < 10000:
            Y_hat = np.matmul(X, thetas).reshape((-1, 1))
            cost = (Y_hat - Y).transpose()
            tmp_thetas = ((sum(np.matmul(cost, X)).transpose()) / self.shape[0]).reshape((-1, 1))
            thetas -= alpha * tmp_thetas
            iterations += 1
        self.thetas = pd.DataFrame(thetas)
        os.system("clear")
        print("\033[1m\033[92mData successfully trained :)\n\033[0m")
        return

    def cost(self):
        """
        Computes the cost of the actual model.
        """
        if not self.on:
            os.system("clear")
            print("\033[1m\033[93mCAUTION! Dataset object is not valid :(\n\033[0m")
            return
        thetas = np.array(self.thetas)
        X = np.array(self.X)
        Y = np.array(self.Y).reshape((1, -1))
        Y_hat = np.matmul(X, thetas).reshape((1, -1))
        cost = sum((Y_hat[0] - Y[0]) ** 2) / (2 * self.shape[0])
        user_input = input("Do you want to plot the cost function? (Yes/No)\n>> ")
        if user_input == "Yes":
            print("\nAvailable labels:\n  {}".format(self.labels[1:]))
            user_input = input("\nWhich label do you want to measure?\n>> ")
            while user_input not in self.labels:
                print("\033[1m\033[91m\nError. Invalid label.\33[0m")
                print("\nAvailable labels:\n  {}".format(self.labels[1:]))
                user_input = input("\nWhich label do you want to measure?\n>> ")
            data_table = [list(), list()]
            index = self.labels.index(user_input)
            count = -50.0
            while count <= 50.0:
                thetas[index][0] += count
                Y_hat = np.matmul(X, thetas).reshape((1, -1))
                cost_ = sum((Y_hat[0] - Y[0]) ** 2) / (2 * self.shape[0])
                data_table[0].append(thetas[index][0])
                data_table[1].append(cost_)
                thetas[index] -= count
                count += 0.1
            plt.plot(data_table[0], data_table[1])
            plt.scatter(thetas[index][0], cost, color = "red")
            plt.xlabel(user_input)
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
        output = 0.0
        theta_pos = 1
        thetas = np.array(self.thetas)
        try:
            output = thetas[0] * 1.0
            for item in self.labels:
                if item == "None":
                    continue
                user_input = input("Introduce {} value:\n>> ".format(item))
                print()
                output += (thetas[theta_pos] * float(user_input))
                theta_pos += 1
            print("The predicted value for {} is: {}\n".format(self.Y.name, output[0]))
            input("Press any key to continue...")
            os.system("clear")
        except:
            os.system("clear")
            print("\033[1m\033[91mError. Invalid values.\n\33[0m")
        return
