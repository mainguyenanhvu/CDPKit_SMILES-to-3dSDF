#!/bin/env python

##
# gen_3d_structs.py 
#
# This file is part of the Chemical Data Processing Toolkit
#
# Copyright (C) 2003 Thomas Seidel <thomas.seidel@univie.ac.at>
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with
# or without fee is hereby granted.
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN
# NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
##


import argparse
import logging as log
import CDPL.Chem as CDPLChem
import CDPL.ConfGen as CDPLConfGen
def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generates conformer ensembles for the given input molecules.')

    parser.add_argument('-i', type=str,
                        dest='in_file',
                        required=True,
                        # metavar='<file>',
                        help='Molecule input file')
    parser.add_argument('-o',
                        dest='out_file',
                        required=True,
                        metavar='<file>',
                        help='Conformer ensemble output file')
    parser.add_argument('-t',
                        dest='max_time',
                        metavar='<int>',
                        type=int,
                        default=3600,
                        help='Max. allowed molecule processing time (default: 3600 sec)')
    parser.add_argument('-q',
                        dest='quiet',
                        action='store_true',
                        default=False,
                        help='Disable progress output (default: false)')
    
    return parser.parse_args()

def gen3DStructure(mol: CDPLChem.Molecule, struct_gen: CDPLConfGen.StructureGenerator) -> int:
    """
    Generates a low-energy 3D structure for the given molecule using the provided initialized 
    CDPLConfGen.StructureGenerator instance.

    Parameters:
    mol (CDPLChem.Molecule): The molecule for which the 3D structure needs to be generated.
    struct_gen (CDPLConfGen.StructureGenerator): An instance of the class CDPLConfGen.StructureGenerator 
                                            that will perform the actual 3D structure generation work.

    Returns:
    int: The status code indicating the success or failure of the 3D structure generation.
         The status code can be one of the following:
         - CDPLConfGen.ReturnCode.SUCCESS: The 3D structure generation was successful.
         - Other status codes indicate different types of errors.
    """
    # prepare the molecule for 3D structure generation
    CDPLConfGen.prepareForConformerGeneration(mol) 

    # generate the 3D structure
    status = struct_gen.generate(mol)             

    # if successful, store the generated conformer ensemble as
    # per atom 3D coordinates arrays (= the way conformers are represented in CDPKit)
    if status == CDPLConfGen.ReturnCode.SUCCESS:
        struct_gen.setCoordinates(mol)                

    return status


def readMolecule(in_file: str) -> list:
    """
    Reads a molecule from the specified input file.

    Parameters:
    in_file (str): The name of the input file.

    Returns:
    CDPLChem.Molecule: The molecule read from the input file.
    """
    mol_list = []
    reader = CDPLChem.MoleculeReader(in_file)
    mol = CDPLChem.BasicMolecule()
    try:
        while reader.read(mol):
            mol_list.append(mol)
            mol = CDPLChem.BasicMolecule()
    except Exception as e:
        log.error(f'Error: reading molecule failed: {str(e)}')
    return mol_list

def process_mol(in_file: str, out_file: str, max_time: int) -> None:
    """
    Reads a molecule from the specified input file, generates a 3D structure
    using the provided instance of CDPLConfGen.StructureGenerator, and writes the
    generated 3D structure to the specified output file.

    Parameters:
    in_file (str): The name of the input file containing the molecule to process.
    out_file (str): The name of the output file to which the generated 3D structure
        should be written.
    max_time (int): The maximum allowed time (in seconds) for the 3D structure
        generation of a single molecule.

    Returns:
    None
    """
    status_to_str = { CDPLConfGen.ReturnCode.UNINITIALIZED                  : 'uninitialized',
                      CDPLConfGen.ReturnCode.TIMEOUT                        : 'max. processing time exceeded',
                      CDPLConfGen.ReturnCode.ABORTED                        : 'aborted',
                      CDPLConfGen.ReturnCode.FORCEFIELD_SETUP_FAILED        : 'force field setup failed',
                      CDPLConfGen.ReturnCode.FORCEFIELD_MINIMIZATION_FAILED : 'force field structure refinement failed',
                      CDPLConfGen.ReturnCode.FRAGMENT_LIBRARY_NOT_SET       : 'fragment library not available',
                      CDPLConfGen.ReturnCode.FRAGMENT_CONF_GEN_FAILED       : 'fragment conformer generation failed',
                      CDPLConfGen.ReturnCode.FRAGMENT_CONF_GEN_TIMEOUT      : 'fragment conformer generation timeout',
                      CDPLConfGen.ReturnCode.FRAGMENT_ALREADY_PROCESSED     : 'fragment already processed',
                      CDPLConfGen.ReturnCode.TORSION_DRIVING_FAILED         : 'torsion driving failed',
                      CDPLConfGen.ReturnCode.CONF_GEN_FAILED                : 'conformer generation failed' }

    # create writer for the generated 3D structures (format specified by file extension)
    writer = CDPLChem.MolecularGraphWriter(out_file) 
    # export only a single 3D structure (in case of multi-conf. input molecules)
    CDPLChem.setMultiConfExportParameter(writer, False)
    
    # create and initialize an instance of the class CDPLConfGen.StructureGenerator which will
    # perform the actual 3D structure generation work
    struct_gen = CDPLConfGen.StructureGenerator()
    struct_gen.settings.timeout = max_time * 1000 # apply the -t argument

    mol_list = readMolecule(in_file)

    for ind, mol in enumerate(mol_list):
        mol_id = ''.join([CDPLChem.getName(mol).strip(), '#'+str(ind+1)]).strip()
        log.info(f'- Generating 3D structure of molecule {mol_id}...')
        try:
            status = gen3DStructure(mol, struct_gen)
        except Exception as e:
            log.error(f'Error: 3D structure generation or output for molecule {mol_id} failed: {str(e)}')
            continue
        if status == CDPLConfGen.ReturnCode.SUCCESS:
            # enforce the output of 3D coordinates in case of MDL file formats
            CDPLChem.setMDLDimensionality(mol, 3)
            if not writer.write(mol):
                log.error(f'Error: writing 3D structure of molecule {mol_id} failed')   
        else:
            log.error(f'Error generating 3D structure for molecule {ind+1}: {status_to_str[status]}')
    writer.close()

def main() -> None:
    args = parseArgs()
    if args.quiet:
        log.basicConfig(format="%(levelname)s: %(message)s")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    process_mol(args.in_file, args.out_file, args.max_time)

if __name__ == '__main__':
    main()
