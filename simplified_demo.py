#!/usr/bin/python
import os
import numpy as np
from scipy.ndimage import imread
from scipy.spatial.distance import cdist
import time
# Parameters
nrun = 20  # Number of classification runs
path_to_script_dir = os.path.abspath(os.getcwd())
path_to_all_runs = os.path.join(path_to_script_dir, 'all_runs')
fname_label = 'class_labels.txt'  # Where class labels are stored for each run


def modified_hausdorf_distance(itemA, itemB):
    # Modified Hausdorff Distance
    D = cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    return max(mean_A, mean_B)
def load_img_as_points(filename):
    I = imread(filename, flatten=True)
    # Convert to boolean array and invert the pixel values
    I = ~np.array(I, dtype=np.bool)
    # Create a new array of all the non-zero element coordinates
    D = np.array(I.nonzero()).T
    return D - D.mean(axis=0)
def classification(folder, ftype='cost'):
    assert ftype in {'cost', 'score'}
    pairs = []
    f = open(os.path.join(path_to_all_runs, folder, fname_label),"r")
    for line in f.readlines():
        pairs.append(line.split())
    test_files, train_files = zip(*pairs)
    answers_files = list(train_files)  # Copy the training file list
    test_files = sorted(test_files)
    train_files = sorted(train_files)
    n_train = len(train_files)
    n_test = len(test_files)
    train_items = []
    test_items = []
    for f in train_files:
        x = os.path.join(path_to_all_runs,f)
        train_items.append(load_img_as_points(x))
    for f in test_files:
        x = os.path.join(path_to_all_runs,f)
        test_items.append(load_img_as_points(x))
        # Compute cost matrix 20*20
    costM = np.zeros((n_test, n_train))
    for i, test_i in enumerate(test_items):
        for j, train_j in enumerate(train_items):
            costM[i, j] = modified_hausdorf_distance(test_i, train_j)
            #print "i = " + str(i) + " j = " + str(j) + " Cost = " + str(costM[i,j])
    if ftype == 'cost':
        y_hats = np.argmin(costM, axis=1)
    elif ftype == 'score':
        y_hats = np.argmax(costM, axis=1)
    else:
        # This should never be reached due to the assert above
        raise ValueError('Unexpected ftype: {}'.format(ftype))
    temp = []
    for y_hat,answer in zip(y_hats,answers_files):
        if train_files[y_hat] == answer :
            temp.append(1)
    correct = len(temp)
    pcorrect = correct/float(n_test)
    perror = 1.0- pcorrect
    return perror*100


# Main function
if __name__ == "__main__":
    #   M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
    #     International Conference on Pattern Recognition, pp. 566-568.
    print('One-shot classification demo with Modified Hausdorff Distance')
    perror = np.zeros(nrun)
    for r in range(nrun):
        arg =  'run{:02d}'.format(r + 1)
        perror[r] = classification(arg,'cost')
        print(' run {:02d} (error {:.1f}%)'.format(r, perror[r]))
    total = np.mean(perror)
    print('Average error {:.1f}%'.format(total))