import sys
import getopt
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_xyscatter(TIME,filename):

	# Example data
	file_path = filename  # Replace with your data file path

	df = pd.read_csv(file_path, na_values=['None'])
	df = df.dropna()
	df['smile_length'] = df['smile'].apply(len)
	x = np.array(df['smile_length'])
	if TIME:
		y = np.array(df['time'])
	else:
		y = np.array(df['score'])

	# Calculate moving average
	#window_size = 10
	#moving_average_y = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
	#moving_average_x = np.convolve(x, np.ones(window_size) / window_size, mode='valid')

	# Create the scatter plot with moving average line
	plt.scatter(x, y, label='Data Points', s=5, color='blue')
	#plt.scatter(moving_average_x, moving_average_y, label='Data Points', color='red')
	#plt.plot(moving_average_x, moving_average_y, label=f'{window_size}-Point Moving Average', color='red')

	# Add line_trace
	#line_trace = plt.Line2D(moving_average_x, moving_average_y, color='green', label=f'X Moving Average')
	#plt.gca().add_artist(line_trace)

	# Add labels and title
	plt.xlabel('SMILES Length')
	if TIME:
		plt.ylabel('Time')
	else:
		plt.ylabel('score')
	if TIME:
		plt.title('SMILES Length vs. Time')
	else:
		plt.title('SMILES Length vs. Score')

	# Add legend
	plt.legend()



	# Save the plot as an image file
	if TIME: 
		output_image_path = "dataset_docking_4m_smiles_time.png"  # Replace with your desired output path
	else:
		output_image_path = "dataset_docking_4m_smiles_score.png"  # Replace with your desired output path
	plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

	# Show the plot
	plt.show()


def main(argv):
    DEBUG = False
    if len(argv) != 1:
        print ('python smi-to-knn-serial.py <smi_dock_file>')
        sys.exit(2)
    else:
        smi_dock_file = argv[0]
        
    plot_xyscatter(True,smi_dock_file)
    plot_xyscatter(False,smi_dock_file)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])

