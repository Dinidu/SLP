import os,sys
import argparse
import csv
from numpy import genfromtxt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perceptron.perceptron import Perceptron

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single layer perceptron")
    parser.add_argument("-f", "--file", help="dataset file in csv", required=True)
    parser.add_argument("-lr", "--learning_rate", help="learning rate of the algorithm", required=True)
    parser.add_argument("-lmse", "--lmse", help="Lowest MSE", required=True)


    args = parser.parse_args()
    learning_rate = args.learning_rate
    filename = str(args.file)
    LMSE = float(args.lmse)

    datalist = []
    with open(args.file) as csvfile:
        pamreader = csv.reader(csvfile, delimiter=',')
        for row in pamreader:
            lin = list(map(float, row[:-2]))
            lout = float(row[-1])
            lnew = [lin,lout]
            datalist.append(lnew)

        print(datalist)

    # Create the perceptron
    p = Perceptron(len(datalist[0][0]))

    # Number of full iterations
    epochs = 0

    # Instantiate mse for the loop
    mse = 999

    while (abs(mse - LMSE) > 0.002):

        # Epoch cumulative error
        error = 0

        # For each set in the training_data
        for value in datalist:
            # Calculate the result
            output = p.result(value[0])

            # Calculate the error
            iter_error = value[1] - output

            # Add the error to the epoch error
            error += iter_error

            # Adjust the weights based on inputs and the error
            p.update_weight(value[0], iter_error)

        # Calculate the MSE - epoch error / number of sets
        mse = float(error / len(datalist))

        # Print the MSE for each epoch
        print "The MSE of %d epochs is %.10f" % (epochs, mse)

        # Every 100 epochs show the weight values
        if epochs % 100 == 0:
            print "0: %.10f - 1: %.10f - 2: %.10f - 3: %.10f" % (p.wR[0], p.wR[1], p.wR[2], p.wR[3])
        # Increment the epoch number
        epochs += 1

    print(p)


