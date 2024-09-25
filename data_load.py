import numpy as np 
from collections import Counter


def read_signal(filename):
    with open(filename,"r") as file:
        data = file.read().splitlines()   
        data = map(lambda x: x.strip().split(),data) # splitting a signal into its samples
        data = [list(map(float,line)) for line in data] # converting str to float
    return data 

def read_labels(filename):
    with open(filename,"r") as file:
        activities = file.read().splitlines()
        activities = list(map(int,activities))
    return activities        

# the reason for using this is the train and test data are already separated. So, we need to just shuffle them 
def randomize(dataset,labels):
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels



def load_dataset(train_dir, trian_files, train_labels, test_dir, test_file, test_labels):
    train_signals, test_signals = list(),list()

    for input_file in trian_files:
        signal = read_signal(train_dir + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals),(1,2,0)) # the way of transposing (signals, samples of signal, components of signal)

    for input_file in test_file:
        signal = read_signal(test_dir + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals),(1,2,0)) 

    train_labels = np.array(read_labels(train_labels))
    test_labels = np.array(read_labels(test_labels))

    no_signals_train, no_steps_train, no_components_train = train_signals.shape
    no_signals_test, no_steps_test, no_components_test = test_signals.shape

    print(f"The train dataset contains {no_signals_train} signals, each one of length {no_steps_train} and {no_components_train} components ")
    print("--------------------------------------------------------------------------------------------")
    print(f"The test dataset contains {no_signals_test} signals, each one of length {no_steps_test} and {no_components_test} components ")
    print("--------------------------------------------------------------------------------------------")
    print(f"The train dataset contains {len(train_labels)} labels, with the following distribution:\n {dict(sorted(Counter(train_labels).items()))}")
    print("--------------------------------------------------------------------------------------------")
    print(f"The test dataset contains {len(test_labels)} labels, with the following distribution:\n {dict(sorted(Counter(test_labels).items()))}")
    
    return train_signals, train_labels, test_signals, test_labels