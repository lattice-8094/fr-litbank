import glob
import argparse
import os
import re
import pandas as pd
import numpy as np
import glob
import sys
import subprocess

if __name__ == '__main__':

    #prendre les arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='input directory containing brat files', required=True)
    parser.add_argument(
        '-o', '--output', help='output directory to write new tsv files to', required=True)

    args = vars(parser.parse_args())
    brat_dir = args["input"]
    outputDir = args["output"]

    #créer le dossier de sortie si inexistant
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)


    for fn in glob.glob(os.path.join('.', brat_dir+'/*.ann')):
        print(fn)
        with open(fn, 'r') as f:
            s = f.read()
        
        all_lines = s.split('\n')
        if all_lines[-1]=='':
            all_lines=all_lines[:-1]
        l = len(all_lines)
        assert(l%2==0)
        
        ent_lines = all_lines[:len(all_lines)//2]
        
        output=''
        for line in ent_lines:
            if not line:
                continue
            def rem_first_part(ent):
                return ent.group(0).split('_',maxsplit=1)[-1]
            line = re.sub('(?<=\t)[A-Za-z:_]*_[A-Za-z_\:]*(?= )', rem_first_part, line)

            output+= line+'\n'
        
        #a developper ! pour préparer le coref et produire un truc comme celui d'Ada
        cmd = "perl -pe 's/ (?=[\w\W]*-[0-9]* [0-9]* [0-9]*\t)/_/g' {} | sed 's/ /\t/' | sed 's/ /\t/' | tail -n{} | sort -k 2,2 -k 3,3 -k 4,4 -V".format(fn,l//2)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sorted_links, error = p.communicate()
       
        ent_ctr=1
        last_ent=""
        last_ref=""
        for line in sorted_links.decode().split('\n'):
            if not line:
                continue
            new_ent = line.split('\t')[1]
            if new_ent==last_ent:
                ref_ent_idx = last_ref
                this_idx = int(line.split('\t')[0][1:])   
                this_ent_idx = this_idx - l//2 
                new_line = 'R{}\tCoreference Arg1:T{} Arg2:T{}'.format(ent_ctr,this_ent_idx,ref_ent_idx)
                ent_ctr+=1
                output+= new_line+'\n'
            else:
                ref_idx = int(line.split('\t')[0][1:])
                ref_ent_idx = ref_idx - l//2
            
            #output+= line+'\n'
            last_ent=new_ent
            last_ref=ref_ent_idx
        
        with open(fn.replace(brat_dir,outputDir+'/'),'w+') as f:
            f.write(output)
