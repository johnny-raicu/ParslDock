#import numpy as np
#np.bool = np.bool_

import sys
import getopt

import pandas as pd
import os
import time
import shutil
import subprocess
from pathlib import Path
from typing import Tuple
import pip

import parsl
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config

import pydockvina
from pydockvina import smi_txt_to_pdb
from pydockvina import set_element
from pydockvina import pdb_to_pdbqt
from pydockvina import make_autodock_vina_config
from pydockvina import run_autodock_vina
#from pydockvina import move_file
#from pydockvina import remove_file
from pydockvina import buf_count_newlines_gen


@python_app
def smi_to_dock(pdbqt_file_name_receptor, index, ligand_name, ligand_smi, results_dir, exhaustiveness,log_file_name,DEBUG):
	import time
	import shutil
	import subprocess
	from pathlib import Path
	from typing import Tuple
	import pip
	import pydockvina
	from pydockvina import smi_txt_to_pdb
	from pydockvina import set_element
	from pydockvina import pdb_to_pdbqt
	from pydockvina import make_autodock_vina_config
	from pydockvina import run_autodock_vina
	#from pydockvina import move_file
	#from pydockvina import remove_file
	from pydockvina import buf_count_newlines_gen




	start_time_dock = time.time()
	if DEBUG:
		print("opening log file",log_file_name)
	log_file = open(log_file_name,"a")
	num_cpu=exhaustiveness
	#specific to 1iep receptor
	cx=15.614
	cy=53.380
	cz=15.455
	sx=20
	sy=20
	sz=20
	if DEBUG:
		print("smi_to_dock("+pdbqt_file_name_receptor+","+str(index)+","+ligand_name+","+ligand_smi+","+results_dir+","+str(exhaustiveness)+")")
	#ver = 1
	if DEBUG:
		print(ligand_name,"-",str(index),".pdb")
	pdb_file_name = ligand_name+"-"+str(index)+".pdb"
	
	#stage1 start
	start_time_stage1 = time.time()
	smi_txt_to_pdb(ligand_smi, pdb_file_name, "mmff")
	end_time_stage1 = time.time()
	#stage1 end

	#stage2 start
	start_time_stage2 = end_time_stage1
	output_pdb_file = pdb_file_name+"_v2.pdb"
	set_element(pdb_file_name, output_pdb_file, results_dir) 
	end_time_stage2 = time.time()
	#stage2 end

	#stage3 start
	start_time_stage3 = end_time_stage2
	autodocktools_path = "/home/iraicu/python-code/mgltools"
	my_autodocktools_path = "mgltools_x86_64Linux2_1.5.7/"
	pdbqt_file_name = pdb_file_name+"qt"
	if DEBUG:
		print("pdb_to_pdbqt()",output_pdb_file,pdbqt_file_name,autodocktools_path)
	pdb_to_pdbqt(output_pdb_file,pdbqt_file_name,autodocktools_path,my_autodocktools_path,results_dir,True)
	x=cx
	y=cy
	z=cz
	end_time_stage3 = time.time()
	start_time_stage4 = end_time_stage3
	#stage3 end

	#stage4 start
	#uncomment out to run stage 4 with docking
	make_autodock_vina_config(pdbqt_file_name_receptor,pdbqt_file_name,"output_conf_file","output_ligand_pdbqt_file","output_log_file",(cx,cy,cz),(sx, sy, sz),exhaustiveness)
	score = run_autodock_vina("vina", "output_conf_file", pdbqt_file_name,results_dir,num_cpu)
	#score = None
	shutil.move(pdbqt_file_name,results_dir+"data/"+pdbqt_file_name)
	#move_file(pdbqt_file_name,results_dir+"data/"+pdbqt_file_name,DEBUG)
	os.remove(output_pdb_file)
	os.remove(pdb_file_name)
	#remove_file(output_pdb_file,DEBUG)
	#remove_file(pdb_file_name,DEBUG)
	end_time_stage4 = time.time()
	#stage4 end
	
	
	end_time_dock = time.time()
	elapsed_time_dock = end_time_dock - start_time_dock
	print("dock,"+ligand_name+","+ligand_smi+","+str(elapsed_time_dock)+","+str(score) +","+str(end_time_stage1 - start_time_stage1) +","+str(end_time_stage2 - start_time_stage2) +","+str(end_time_stage3 - start_time_stage3) +","+str(end_time_stage4 - start_time_stage4) )
	log_file.write("dock,"+ligand_name+","+ligand_smi+","+str(elapsed_time_dock)+","+str(score) +","+str(end_time_stage1 - start_time_stage1) +","+str(end_time_stage2 - start_time_stage2) +","+str(end_time_stage3 - start_time_stage3) +","+str(end_time_stage4 - start_time_stage4)+"\n")
	log_file.flush()
	log_file.close()
	
	return ligand_name,score


def main(argv):
	DEBUG = False
	if len(argv) != 8:
		print ('python smi-to-dock-parallel.py <receptor_file_pdbqt> <ligand_file_smi> <num_dockings> <results_dir> <exhaustiveness> <log_file> <parallelism> <debug>')
		sys.exit(2)
	else:
		pdbqt_file_name_receptor = argv[0]
		smi_file_name_ligand = argv[1]
		chunksize = 1000
		num_dockings = int(argv[2])
		results_dir = argv[3]
		exhaustiveness = int(argv[4])
		log_file_name = argv[5]
		parallelism = int(argv[6])
		if argv[7] == "True":
			DEBUG = True
		elif argv[7] == "False":
			DEBUG = False
		else:
			print("wrong debug flag:",argv[7])
			sys.exit(2)
		#print(DEBUG,argv[7])

	import time
	#parsl.load(config)
	from parsl.config import Config
	from parsl.executors.threads import ThreadPoolExecutor

	local_threads = Config(
		executors=[
			ThreadPoolExecutor(
				max_threads=1, 
				label='local_threads'
			)
		]
	)

	from parsl.providers import LocalProvider
	from parsl.channels import LocalChannel
	from parsl.config import Config
	from parsl.executors import HighThroughputExecutor

	local_htex = Config(
		executors=[
			HighThroughputExecutor(
				label="local_htex",
				worker_debug=False,
				cores_per_worker=exhaustiveness,
				max_workers=parallelism,
				prefetch_capacity=32,
				provider=LocalProvider(
					channel=LocalChannel(),
					init_blocks=1,
					parallelism=1.0,
					max_blocks=10,
				),
			)
		],
		strategy=None,
	)

	parsl.clear()
	parsl.load(local_htex)

		

	if os.path.exists(pdbqt_file_name_receptor) == False:
		print(pdbqt_file_name_receptor + " does not exist, change the receptor file...")
		return -1

	if os.path.exists(smi_file_name_ligand) == False:
		print(smi_file_name_ligand + " does not exist, change the ligand file...")
		return -1

	
	if os.path.exists(results_dir) == False:
		os.makedirs(results_dir) 

	if os.path.exists(results_dir+"data/") == False:
		os.makedirs(results_dir+"data/") 
	if os.path.exists(results_dir+"output/") == False:
		os.makedirs(results_dir+"output/") 

		
	#log_file = open(results_dir+log_file_name,"w")
	try:
		file_path = results_dir+log_file_name
		os.remove(file_path)
		if DEBUG:
			print(f"File '{file_path}' removed successfully.")
	except Exception as e:
		if DEBUG:
			print(f"Error: {e}")

	
	start_time = time.time()
	chunkindex = 0
	filename = smi_file_name_ligand
    #"orz_table_"+str(run)+".csv"
	if DEBUG:
		print("counting the number of lines in file " + filename + "; this may take longer for large files")
	amount = pydockvina.buf_count_newlines_gen(filename) - 1
	if DEBUG: 
		print(str(amount) + " number of rows")
	
	if num_dockings <= 0:
		num_dockings = amount

	log_file = open(results_dir+log_file_name,"a")

	#log_file.write("dock,"+ligand_name+","+str(elapsed_time_dock)+","+str(score) +","+str(end_time_stage1 - start_time_stage1) +","+str(end_time_stage2 - start_time_stage2) +","+str(end_time_stage3 - start_time_stage3) +","+str(end_time_stage4 - start_time_stage4)+"\n")
	#print("operation,TITLE,total_time,score,stage1_time,stage2_time,,stage3_time,stage4_time")
	log_file.write("operation,ligand,smile,time,score,stage1_time,stage2_time,stage3_time,stage4_time\n")
	log_file.flush()
	log_file.close()
	


	dock_results = []

	for chunk in pd.read_csv(filename, chunksize=chunksize, nrows=num_dockings):
		print("submitting task "+str(chunkindex*chunksize+1) + "/" + str(num_dockings))
		for index, row in chunk.iterrows():
			result = smi_to_dock(pdbqt_file_name_receptor,chunkindex*chunksize+index,row["TITLE"], row["SMILES"],results_dir,exhaustiveness,results_dir+log_file_name,DEBUG)
			dock_results.append(result)
			
		chunkindex += 1
	
	print("waiting for",len(dock_results),"results... check out log ==> tail -f",results_dir+log_file_name)
		# Wait for all apps to finish and collect the results
	final_outputs = [i.result() for i in dock_results]

	maxCount = 10
	print("printing only first",maxCount,"results... see",results_dir+log_file_name,"for all the results!")
	count = 0
	for result in final_outputs:
		ligand,score = result
		print("ligand=",ligand,"score=",score)
		count += 1
		if count >= maxCount:
			break
	
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time run: ", elapsed_time, "seconds")
	
if __name__ == '__main__':
	main(sys.argv[1:])
