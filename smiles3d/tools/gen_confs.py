#!/bin/env python

##
# gen_confs.py 
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

    parser.add_argument('-i',
                        dest='in_file',
                        required=True,
                        metavar='<file>',
                        help='Molecule input file')
    parser.add_argument('-o',
                        dest='out_file',
                        required=True,
                        metavar='<file>',
                        help='Conformer ensemble output file')
    parser.add_argument('-e',
                        dest='e_window',
                        required=False,
                        metavar='<float>',
                        type=float,
                        default=20.0,
                        help='Output conformer energy window (default: 20.0)')
    parser.add_argument('-r',
                        dest='min_rmsd',
                        required=False,
                        metavar='<float>',
                        type=float,
                        default=0.5,
                        help='Output conformer RMSD threshold (default: 0.5)')
    parser.add_argument('-t',
                        dest='max_time',
                        required=False,
                        metavar='<int>',
                        type=int,
                        default=3600,
                        help='Max. allowed molecule processing time (default: 3600 sec)')
    parser.add_argument('-n',
                        dest='max_confs',
                        required=False,
                        metavar='<int>',
                        type=int,
                        default=100,
                        help='Max. output ensemble size (default: 100)')
    parser.add_argument('-q',
                        dest='quiet',
                        required=False,
                        action='store_true',
                        default=False,
                        help='Disable progress output (default: false)')
    
    return parser.parse_args()


# generates a conformer ensemble of the argument molecule using
# the provided initialized ConfGen.ConformerGenerator instance
def genConfEnsemble(mol: CDPLChem.Molecule, conf_gen: CDPLConfGen.ConformerGenerator) -> tuple[int,int]:
    # prepare the molecule for conformer generation
    CDPLConfGen.prepareForConformerGeneration(mol) 

    # generate the conformer ensemble
    status = conf_gen.generate(mol)             
    num_confs = conf_gen.getNumConformers()
    
    # if successful, store the generated conformer ensemble as
    # per atom 3D coordinates arrays (= the way conformers are represented in CDPKit)
    if status == CDPLConfGen.ReturnCode.SUCCESS or status == CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
        conf_gen.setConformers(mol)                
    else:
        num_confs = 0
        
    # return status code and the number of generated conformers
    return (status, num_confs)

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

def process_mol(in_file: str, out_file: str, max_time: int, min_rmsd: float, e_window: float, max_confs: int) -> None:
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
    
    # create and initialize an instance of the class ConfGen.ConformerGenerator which
    # will perform the actual conformer ensemble generation work
    conf_gen = CDPLConfGen.ConformerGenerator()

    conf_gen.settings.timeout = max_time * 1000          # apply the -t argument
    conf_gen.settings.minRMSD = min_rmsd                 # apply the -r argument
    conf_gen.settings.energyWindow = e_window            # apply the -e argument
    conf_gen.settings.maxNumOutputConformers = max_confs # apply the -n argument

    mol_list = readMolecule(in_file)

    for ind, mol in enumerate(mol_list):
        mol_id = ''.join([CDPLChem.getName(mol).strip(), '#'+str(ind+1)]).strip()
        log.info(f'- Generating 3D structure of molecule {mol_id}...')
        try:
            status, num_confs = genConfEnsemble(mol, conf_gen)
        except Exception as e:
            log.error(f'Error: conformer ensemble generation or output for molecule {mol_id} failed: {str(e)}')
            continue
        if status != CDPLConfGen.ReturnCode.SUCCESS and status != CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
            log.error(f'Conformer ensemble generation for molecule {mol_id} failed: {status_to_str[status]}')

        elif status == CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
            log.error(f' -> Generated {str(num_confs)} conformers (warning: too much top. symmetry - output ensemble may contain duplicates)')
        else:
            log.info(f' -> Generated {str(num_confs)} conformer(s)')

        if num_confs > 0:
            if not writer.write(mol):
                log.error(f'Error: writing 3D structure of molecule {mol_id} failed')   
        
    writer.close()

def main() -> None:
    args = parseArgs()
    if args.quiet:
        log.basicConfig(format="%(levelname)s: %(message)s")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    process_mol(args.in_file, args.out_file, args.max_time, args.min_rmsd, args.e_window, args.max_confs)
        

if __name__ == '__main__':
    main()