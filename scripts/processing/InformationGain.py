import pandas as pd
from info_gain import info_gain


import numpy as np
import math

def informationGain(firtDf, nameFirtColumn, secondDf, nameSecondColumn):
    if ((not firtDf.empty) and (not secondDf.empty)):
        try:
            mergeDf = pd.merge(secondDf, firtDf, how='left', on=['year', 'countryName'])        
            ig  = info_gain.info_gain(mergeDf[nameSecondColumn], mergeDf[nameFirtColumn])
            iv  = info_gain.intrinsic_value(mergeDf[nameSecondColumn], mergeDf[nameFirtColumn])
            igr = info_gain.info_gain_ratio(mergeDf[nameSecondColumn], mergeDf[nameFirtColumn])
            return ig, iv, igr
            
        except Exception:
            print('Error during processing Information Gain')
            return 0,0,0
    else:
        return 0,0,0
        




def calc_entropy(column):
    counts=np.bincount(column) #This code will return the number of each unique value in a column
    probability=counts/(len(column))#here we calculate the probability of each value vy dividing the length of the entire column
    entropy=0#we start 0 as the intial value of entropy
    for prob in probability: # here we a for loop to go throuh each probability of each unique value in the column
        if prob >0:
            entropy += prob * math.log(prob, 2) # here calculate entropy of each value and add them to find the total emtropy
    return -entropy # we should return - * entropy due to the formula
    


#TODO preciso estudar esse information gain 2
def information_gain2(data, split,target):
    try:
        original_entropy=calc_entropy(data[target])
        values=data[split].unique()
        left_split=data[data[split]==values[0]]
        right_split=data[data[split]==values[1]]
        subract=0
        for subset in [left_split,right_split]:
            prob=(subset.shape[0])/data.shape[0]
            subract += prob * calc_entropy(subset[target])
        return  original_entropy - subract
    
    except Exception:
        return -1