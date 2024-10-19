import os
import argparse
import subprocess
import tempfile
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')  

root = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='Generates conformers for a given list of SMILES strings.')

    parser.add_argument('-i', dest='in_file',
                        required=True,
                        metavar='<file>',
                        help='Molecule input file in CSV whose columns which are "smiles" and "id" are required or SMI file or SMILES string.')
    parser.add_argument('-o', dest='out_file',
                        required=True,
                        metavar='<file>',
                        help='Output file in SDF format.')
    parser.add_argument('--smi_col', type=str,
                        default='smiles',
                        help='Name of the column containing SMILES strings (default: "smiles")')
    parser.add_argument('--id_col', type=str,
                        default='id',
                        help='Name of the column containing IDs (default: "id")')
    parser.add_argument('--is_remove_duplicates', action="store_true",
                        default=False,
                        help='If set True, flag to remove duplicate smiles in files.')
    parser.add_argument('--is_combined', action="store_true",
                        default=False,
                        help='If set False, flag to create singly file.')
    parser.add_argument('-t', dest='max_time',
                        metavar='<int>',
                        type=int,
                        default=60,
                        help='Max. allowed molecule processing time (default: 60 sec)')
    parser.add_argument('-q', dest='quiet',
                        action='store_true',
                        default=False,
                        help='Disable progress output (default: false)')
    parser.add_argument('-n', dest='num_confs',
                        metavar='<int>',
                        type=int,
                        default=10,
                        help='Number of conformers to generate (default: 10)')
    
    return parser.parse_args()

def preprocess_mol_with_RDKit(temp_input_file, smiles_list, id_list):
    """
    Preprocesses a list of SMILES strings and IDs by adding hydrogens,
    computing 2D coordinates, and adding properties with the input SMILES,
    input InChIKey, molecule ID, and input index.

    Also adds properties with the RDKit computed SMILES and InChIKey.

    :param temp_input_file: Temporary SDF file to store the preprocessed molecules
    :param smiles_list: List of SMILES strings to preprocess
    :param id_list: List of IDs for the molecules
    """
    writer = Chem.SDWriter(temp_input_file)
    for i, (smi, mol_id) in enumerate(zip(smiles_list, id_list)):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.Compute2DCoords(mol)
        if mol is None:
            continue
        mol.SetProp("_MoleculeID", str(mol_id))
        mol.SetProp("_InputIndex", str(i))
        mol.SetProp("_InputSMILES", str(smi))
        mol.SetProp("_InputInChIKey", str(inchi.InchiToInchiKey(inchi.MolToInchi(Chem.MolFromSmiles(smi)))))
        mol.SetProp("_SMILES", str(Chem.MolToSmiles(mol)))
        mol.SetProp("_InChIKey", str(inchi.InchiToInchiKey(inchi.MolToInchi(mol))))
        writer.write(mol)
    writer.close()

def write_output(temp_output_file, output_file):
    """
    Reads the output SDF file and adds _Name property to each molecule conformer.
    Each conformer gets a name like "mol_id_confX" where X is the conformer number.
    """
    sdf_supplier = Chem.SDMolSupplier(temp_output_file, removeHs=False)
    conf_counts = {}
    sdf_writer = None
    if args.is_combined:
        sdf_writer = Chem.SDWriter(output_file)
    for mol in sdf_supplier:
        if mol is None:
            continue
        mol_id = mol.GetProp("_MoleculeID")
        if mol_id not in conf_counts:
            conf_counts[mol_id] = 0
        conf_label = f"{mol_id}_conf{conf_counts[mol_id]}"
        mol.SetProp("_Name", str(conf_label))
        conf_counts[mol_id] += 1
        if sdf_writer:
            sdf_writer.write(mol)
        else:
            sdf_writer_single = Chem.SDWriter(output_file.replace(".sdf", f"_{conf_label}.sdf"))
            sdf_writer_single.write(mol)
            sdf_writer_single.close()
    if sdf_writer:
        sdf_writer.close()

def standardize_ID(df):
    """
    This function standardizes the IDs in a DataFrame by removing any punctuation from the IDs.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the IDs to be standardized. The IDs are assumed to be in a column named by the value of the 'id_col' argument.

    Returns:
    pandas.DataFrame: The DataFrame with the standardized IDs. The IDs are stored in a column named by the value of the 'id_col' argument.

    Ref: https://stackoverflow.com/a/50444347/10987905
    """
    punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'   # `|` is not present here
    transtab = str.maketrans(dict.fromkeys(punct, ''))

    df[args.id_col] = '|'.join(df[args.id_col].tolist()).translate(transtab).split('|')

    return df

def read_input(in_file):
    """
    Read input file or string and return a pandas DataFrame.

    The function assumes that the input is either a csv file, a smi file, or a SMILES string.
    If the input is a csv file, it is read into a pandas DataFrame using pd.read_csv.
    If the input is a smi file, it is read into a pandas DataFrame using pd.read_csv with the
    header set to None and the column name set to the value of the smi_col argument.
    If the input is a SMILES string, it is converted into a pandas DataFrame with a single
    column with the name of the smi_col argument.

    If the id_col argument is not found in the input DataFrame, it is created by joining
    the output file name with the row number.

    If the is_remove_duplicates argument is True, the function removes duplicate SMILES
    strings from the input DataFrame.

    Parameters
    ----------
    None

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with the ID column added if necessary and duplicates removed
        if necessary.
    """
    if in_file.endswith('.csv'):
        df = pd.read_csv(args.in_file)
    elif in_file.endswith('.smi'):
        df = pd.read_csv(in_file, header=None, names=[args.smi_col])
    else:
        df = pd.DataFrame({args.smi_col: [in_file]})
    if args.id_col not in df.columns:
        df[args.id_col] = ['_'.join([os.path.basename(args.out_file).replace('.sdf', ''), str(x)]) for x in range(1, df.shape[0]+1)]
    else:
        standardize_ID(df)
    if args.is_remove_duplicates:
        df.drop_duplicates(subset=[args.smi_col], inplace=True, keep='first')
    return df

def generate_conformers(in_file, out_file, smiles_col='smiles', id_col='id', max_time=60, num_confs=10, quiet=False):
    """
    Generate conformers for a given list of SMILES strings.

    Parameters
    ----------
    in_file : str
        Molecule input file in CSV format. Columns which are "smiles" and "id" are required.
    out_file : str
        Output file in SDF format.
    smiles_col : str, optional
        Name of the column containing SMILES strings (default: "smiles").
    id_col : str, optional
        Name of the column containing IDs (default: "id").
    max_time : int, optional
        Max. allowed molecule processing time (default: 60 sec).
    num_confs : int, optional
        Number of conformers to generate (default: 10).
    quiet : bool, optional
        Disable progress output (default: false).

    Returns
    -------
    None
    """
    tmp_dir = tempfile.mkdtemp(prefix='smi3d_')
    temp_input_file = os.path.join(tmp_dir, 'temp_input.sdf')
    temp_output_file = os.path.join(tmp_dir, 'temp_output.sdf')
    
    df = read_input(in_file)
    if (smiles_col not in df.columns) or (id_col not in df.columns):
        raise ValueError(f"The CSV file does not contain the required columns which are {smiles_col} and {id_col}.")
    smiles_list = df[smiles_col].to_list()
    id_list = df[id_col].to_list()
    preprocess_mol_with_RDKit(temp_input_file, smiles_list, id_list)

    if num_confs == 1:
        cmd = f"python {root}/tools/gen_3d_structs.py -i {temp_input_file} -o {temp_output_file} -t {max_time}"
    else:
        cmd = f"python {root}/tools/gen_confs.py -i {temp_input_file} -o {temp_output_file} -t {max_time} -n {num_confs}"

    subprocess.Popen(cmd, shell=True).wait()

    write_output(temp_output_file, out_file)
    shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    args = parse_args()
    generate_conformers(args.in_file, args.out_file, args.smi_col, args.id_col, args.max_time, args.num_confs, args.quiet)