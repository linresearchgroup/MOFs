###############################################
# This file mainly realize series work of 
# data preprocessing, including noise extracting, 
# normalization, transposition and adding noise, 
# to expand the data set which could be used to  
# train our model in the next step.
# ---------------------------------------------
# This is the third step to prodess the data.
# From step 2 we got theory data list, 
# we will generate the training data set after 
# these data preprocessing.(This is an example 2000)
###############################################


#------------------------------------------
# Import some basic packages we need to use
#------------------------------------------
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')												
import time
start = time.clock()

#------------------------------------------
# Read data list file
#------------------------------------------
file_to_read = open('D_origin.pickle', 'rb')									
tmp = pickle.load(file_to_read)
data = tmp.values

file_to_read2 = open('prep_theo_data_2000_1814.pickle', 'rb')					
tmp2 = pickle.load(file_to_read2)
data2 = tmp2.values

file_to_read1 = open('T_origin.pickle', 'rb')									
tmp1 = pickle.load(file_to_read1)
test_data = tmp1.values

# x1 = np.linspace(0, 2252,num=2251)  #for plot use
# y1 = data[0,:2251]
#------------------------------------------
# The total data size
#------------------------------------------
add_data_size   = 2314
input_data_size = 186 + add_data_size											

#---------------------------------------------------------------------
# Select a ['ZIF-67' 'ZIF-67' 'ZIF-67'], b ['ZIF-71' 'ZIF-71' 
# 'ZIF-71' 'ZIF-8' 'ZIF-8' 'ZIF-8' 'ZIF-8' 'ZIF-90' 'ZIF-90' 'ZIF-90']
# as the noise to be added
#---------------------------------------------------------------------
a = test_data[0:2, :]															
b = test_data[3:, :]
test_data = np.concatenate((a, b), axis=0)

#------------------------------------------
# Extract low noise to generate data 
#------------------------------------------
test_data = test_data[6:,:]
data = data[:,:-1]																# get data without lable of original 186 data
data = np.concatenate((data, data2), axis=0)									# combine data values from different lists togehter

file_to_read3 = open('prep_theo_label_2000_1814.pickle', 'rb')					# read label list of theory data
tmp3 = pickle.load(file_to_read3)
data3 = tmp3

label186 = tmp.values[:,-1]
label86 = data3
label = np.concatenate((label186, label86), axis=0)								# combine labels from different lists togehter
# pd.to_pickle(label, "labelinput_data_size")

print('the previous lable total number is:  {}'.format(label186.shape))
print('the new imported lable total number is:  {}'.format(len(label86)))
print('the total combined lable number is:  {}'.format(label.shape))


#------------------------------------------
# Normalization and transposition
#------------------------------------------
test_data = test_data[:,:-1]
test_data = test_data.T
scaler = MinMaxScaler()
test_data = scaler.fit_transform(test_data)
test_data=test_data.T
data = data.T
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = data.T


#------------------------------------------
# Add low noise to generate training data
#------------------------------------------
lc_num = 401
lngth1 = int(test_data.shape[1] - lc_num)
new_training_data = np.zeros((input_data_size * 6 * 12, 2251)) 					# creat an empty container for data
label_lists = []

np.random.seed(1)

for i in range(input_data_size):

	label_lists.append([label[i]] * 6 * 12)

	if i == 175:                             									# zif-90
		j_lists = [0, 1, 2, 3, 0, 1]
	elif i == 155:                            									# zif-8
		j_lists = [4, 5, 6, 4, 5, 6]
	else:
		j_lists = np.arange(6)	

	theoretical = data[i]
	h_the = np.partition(theoretical, -lc_num)[-lc_num]

	assert(len(j_lists) == 6)

	lc_id1 = i * 6 * 12

	# print(i)

	for j in range(len(j_lists)):
		
		experiment = test_data[ j_lists[j] ]
		h_exp = np.partition(experiment, -lc_num)[-lc_num]

		part_data = experiment[experiment <= h_exp][:lngth1]

		lc_id2 = j * 12

		for k in range(12): 													# expand each data to 6*12=72 data
			
			copy_theo = np.copy(theoretical)
			copy_theo[theoretical < h_the] = np.random.permutation(part_data)

			lc_id = lc_id1 + lc_id2 + k

			new_training_data[lc_id] = copy_theo

new_train = new_training_data.tolist()

flat_list = [item for sublist in label_lists for item in sublist]				


print('double check the total combined lable number is:  {}'.format(len(label_lists)))
print('after filtering the repeated labels, the lable number is')
print(len(set(flat_list)))

print('the total train data number is:  {}'.format(len(new_train)))
print('the total train label number is:  {}'.format(len(flat_list)))


#------------------------------------
# Add labels to generated data
#------------------------------------
for i in range(input_data_size * 6 * 12):
	new_train[i].append(flat_list[i])

end = time.clock()
print('time cost')
print(str(end-start))


#----------------------------------------------
# Generate the pickle list of generated data
# Training data set
#----------------------------------------------
pickle.dump(new_train, open( "new_data_filter_2000.pickle", "wb" ))

