# this file used for FFT & iFFT to small part of noisy data
# Only used for noisy data which CNN can't recognized
# otherwise, you don't need to do this step


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

os.environ['PYTHONHASHSEED'] = '0'


# change the data file name!!!
test_file="T_origin19.pickle"
tmp = np.load(test_file)
data_14 = tmp.values
print(data_14.shape)

zif67_3 = np.zeros((4,2304))
list_test = [data_14[12],data_14[13],data_14[14],data_14[15]]

# pick larger 400 value 

num_large = 400

fft_len = 2304
effect_fft = int(fft_len / 2)


for k in range(3,4):

	data_zif90 = list_test[k][:-1]	
	data_zif90 -= np.median(data_zif90)
	data_zif90[data_zif90<0] = 0
	data_zif90 /= np.max(np.abs(data_zif90))

	########################change peak value to mean	
	copy_no_peak = np.copy(data_zif90)

	h_the = np.partition(data_zif90, -num_large)[-num_large]
	# copy_no_peak[data_zif90 > h_the] = np.median(data_zif90)
	copy_no_peak[data_zif90 > h_the] = 0

	########################change noise to 0
	copy_only_peak = np.copy(data_zif90)
	copy_only_peak[data_zif90 < h_the] = 0
	# copy_only_peak[data_zif90 < h_the] = np.median(data_zif90)

	names= ['original', 'no_peak', 'only_peak']	

	data = np.zeros((len(names), fft_len))

	data[0, 20:2251+20] = data_zif90
	data[1, 20:2251+20] = copy_no_peak
	data[2, 20:2251+20] = copy_only_peak

	magnitd = np.zeros((len(names), effect_fft+1))

	for i in range(len(names)):
		# data[i] /= np.max(np.abs(data[i]))

		plt.clf()
		axes = plt.gca()
		axes.set_ylim([-0.1, 1.0])
		figname = str(i) + '.eps'
		#plt.plot(magnitd[i],     'r', label='freq')
		plt.plot(data[i],     'r', label='freq')

		plt.title(str(i) + names[i])
		plt.legend(bbox_to_anchor=(0.2, 1.08, 1., .102), loc=1)
		plt.savefig(figname)
		plt.close(0)


		# if names[i] == 'only_peak':
		recov_freq = np.zeros((fft_len), dtype=np.complex_)
		recov_sigl = np.zeros((2304), dtype=np.complex_)

		tmp_complex = (np.fft.fft(data[i]))[:int(fft_len/2+1)]		
		magnitd[i] = np.log(np.abs(tmp_complex) + 0.000001)
		# recov_freq[:200] = tmp_complex[:200]
		recov_freq[:150] = tmp_complex[:150]

		#duplicate range when doning fft
		duplicate_range = int(fft_len / 2 - 1)

		for freq_id in range(duplicate_range):
			recov_freq[fft_len - freq_id - 1] = np.conjugate(recov_freq[freq_id + 1])
		
		recov_sigl = np.fft.ifft(recov_freq)

		plt.clf()
		axes = plt.gca()
		axes.set_ylim([-0.1, 1.0])
		figname = str(i) + '_recover.eps'
		plt.plot(recov_sigl.real,     'r', label='time')
		# plt.plot(magnitd[i],     'r', label='time')

		plt.title(str(i) + names[i] + "_recover")
		plt.legend(bbox_to_anchor=(0.2, 1.08, 1., .102), loc=1)
		plt.savefig(figname)
		plt.close(0)

