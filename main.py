#############
# Libraries #
#############

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from Dataset import Dataset

#########
# Tools #
#########

def display_prompt():
    """
    Displays a prompt showing the valid options.
    """
    option = -1
    while option < 0:
        print("\033[1mVALID OPTIONS")
        print("  1. Load dataset")
        print("  2. See dataset")
        print("  3. Train model")
        print("  4. Analyze reliability")
        print("  5. Predict values")
        print("  6. Quit\033[0m")
        print()
        user_input = input("Which operation do you want to do?\n>> ")
        try:
            print()
            user_input = int(user_input)
            if user_input <= 0 or user_input >= 7:
                os.system("clear")
                print("\033[1m\033[91mError. You should choose a number between 1 and 6.\n\033[0m")
            else:
                option = user_input
        except:
            os.system("clear")
            print("\033[1m\033[91mError. You should choose a number between 1 and 6.\n\033[0m")
    return option

###########
# Program #
###########

if __name__ == '__main__':
    os.system("clear")
    print("\033[1mWELCOME TO PREDICT\033[0m")
    try:
        on = 0
        while True:
            option = display_prompt()
            if option == 1:
                dataset = Dataset()
                on = 1
            elif on == 1 and option == 2:
                dataset.display_data()
            elif on == 1 and option == 3:
                dataset.train()
            elif on == 1 and option == 4:
                dataset.cost()
            elif on == 1 and option == 5:
                dataset.predict()
            elif option == 6:
                print("\033[1m\033[93mSee you soon :)\n\033[0m")
                break
            else:
                os.system("clear")
                print("\033[1m\033[93mCAUTION! Dataset object is not valid :(\n\033[0m")
    except:
        print("\033[1m\033[93m\n\nSee you soon :)\n\033[0m")
