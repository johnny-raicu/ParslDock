import time
import shutil
import subprocess
from pathlib import Path
from typing import Tuple
import pip

def smi_txt_to_pdb(smiles: str, pdb_file: str, forcefield: str = "mmff") -> None:
	"""Converts SMILE text to a PDB file.
​
	Parameters
	----------
	smiles_file : str
		Input SMILES text.
	pdb_file : str
		Output PDB file.
	forcefield : str, optional
		Forcefield to use for 3D conformation generation
		(either "mmff" or "etkdg"), by default "mmff".
	"""
	from rdkit import Chem
	from rdkit.Chem import AllChem

	# Read the SMILES file
	#with open(smiles_file, "r") as f:
		#smiles = f.read().strip()
	# Convert SMILES to RDKit molecule object
	mol = Chem.MolFromSmiles(smiles)
	# Add hydrogens to the molecule
	mol = Chem.AddHs(mol)
	# Generate a 3D conformation for the molecule
	if forcefield == "mmff":
		AllChem.EmbedMolecule(mol)
		AllChem.MMFFOptimizeMolecule(mol)
	elif forcefield == "etkdg":
		AllChem.EmbedMolecule(mol, AllChem.ETKDG())
	else:
		raise ValueError(f"Unknown forcefield: {forcefield}")
	# Write the molecule to a PDB file
	writer = Chem.PDBWriter(pdb_file)
	writer.write(mol)
	writer.close()




def set_element(input_pdb_file: str, output_pdb_file: str ,results_dir: str) -> None:
	"""Set the element of each atom in a PDB file.
​
	Parameters
	----------
	input_pdb_file : str
		Input PDB file.
	output_pdb_file : str
		Output PDB file.
	"""
	log_file_name = results_dir+"output/"+output_pdb_file+".txt"
	tcl_script = Path(pip.__file__).parent / "tcl_utils" / "set_element.tcl"
	command = (
		f"vmd -dispdev text -e {tcl_script} -args {input_pdb_file} {output_pdb_file}"
	)
	
	#print("set_element():",command)
	result = subprocess.check_output(command.split())
	#print("set_element():",result)
	log_file = open(log_file_name,"w")
	log_file.write(str(result))
	log_file.close()
	#return result



def pdb_to_pdbqt(
	pdb_file: str, pdbqt_file: str, autodocktools_path: str,my_autodocktools_install:str, results_dir: str,  ligand: bool = True
) -> None:
	"""Convert a PDB file to a PDBQT file for a receptor.
	Parameters
	----------
	pdb_file : str
		Input PDB file.
	pdbqt_file : str
		Output PDBQT file.
	autodocktools_path : str
		Path to AutoDockTools folder.
	ligand : bool, optional
		Whether the PDB file is a ligand or receptor, by default True.
	"""
	log_file_name = results_dir+"output/"+pdbqt_file+".txt"
	# Select the correct settings for ligand or receptor preparation
	script, flag = (
		("prepare_ligand4.py", "l") if ligand else ("prepare_receptor4.py", "r")
	)
	
	command = (
		f"{Path(autodocktools_path) / my_autodocktools_install / 'bin/pythonsh'}"
		f" {Path(autodocktools_path) / my_autodocktools_install / 'MGLToolsPckgs/AutoDockTools/Utilities24' / script}"
		f" -{flag} {pdb_file}"
		f" -o {pdbqt_file}"
		f" -U nphs_lps_waters"
	)
	result = subprocess.check_output(command.split(), encoding="utf-8")
	log_file = open(log_file_name,"w")
	log_file.write(str(result))
	log_file.close()



def make_autodock_vina_config(
	input_receptor_pdbqt_file: str,
	input_ligand_pdbqt_file: str,
	output_conf_file: str,
	output_ligand_pdbqt_file: str,
	output_log_file: str,
	center: Tuple[float, float, float],
	size: Tuple[int, int, int],
	exhaustiveness: int = 20,
	num_modes: int = 20,
	energy_range: int = 10,
) -> None:
	"""Make a configuration file for AutoDock Vina.
	Parameters
	----------
	input_receptor_pdbqt_file : str
		Input receptor PDBQT file.
	input_ligand_pdbqt_file : str
		Input ligand PDBQT file.
	output_conf_file : str
		Output configuration file.
	output_ligand_pdbqt_file : str
		Output ligand PDBQT file.
	output_log_file : str
		Output log file.
	center : Tuple[float, float, float]
		Center of the search space.
	size : Tuple[int, int, int]
		Size of the search space.
	exhaustiveness : int, optional
		Exhaustiveness of the search, by default 20.
	num_modes : int, optional
		Number of binding modes to generate, by default 20.
	energy_range : int, optional
		Energy range, by default 10.
	"""
	# Format configuration file
	file_contents = (
		f"receptor = {input_receptor_pdbqt_file}\n"
		f"ligand = {input_ligand_pdbqt_file}\n"
		f"center_x = {center[0]}\n"
		f"center_y = {center[1]}\n"
		f"center_z = {center[2]}\n"
		f"size_x = {size[0]}\n"
		f"size_y = {size[1]}\n"
		f"size_z = {size[2]}\n"
		f"exhaustiveness = {exhaustiveness}\n"
		f"num_modes = {num_modes}\n"
		f"energy_range = {energy_range}\n"
		f"out = {output_ligand_pdbqt_file}\n"
		#f"log = {output_log_file}\n"
	)
	# Write configuration file
	with open(output_conf_file, "w") as f:
		f.write(file_contents)
	
	

def run_autodock_vina(
	autodock_vina_exe: str, config_file: str, ligand_file_name: str, results_dir: str, num_cpu: int = 8
) -> float:
	"""Run AutoDock Vina.
​
	Parameters
	----------
	autodock_vina_exe : str
		Path to AutoDock Vina executable.
	config_file : str
		Path to AutoDock Vina configuration file.
	num_cpu : int, optional
		Number of CPUs to use, by default 8.
	"""
	try:
		log_file_name = results_dir+"output/"+ligand_file_name+"_dock.txt"
		command = f"{autodock_vina_exe} --config {config_file} --cpu {num_cpu}"
		result = subprocess.check_output(command.split(), encoding="utf-8")
		result_list = result.split('\n')
		score = result_list[38].split()
		log_file = open(log_file_name,"w")
		log_file.write(str(result))
		log_file.close()
		return float(score[1])
	except subprocess.CalledProcessError as e:
		print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")
		return None
	except Exception as e:
		print(f"Error: {e}")
		return None


def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

