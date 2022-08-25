import glob
import argparse
import os
import re
import pandas as pd
import numpy as np
import glob
import sys

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

    #types d'entités à supprimer
    to_clean=['TO_DISCUSS','OTHER','None', 'METALEPSE']

    ctr=0
    for fn in glob.glob(os.path.join('.', brat_dir+'/*.ann')):
        print(fn)
        with open(fn, 'r') as f:
            s = f.read()

        entities = []
        for line in s.split('\n'):
            if not line:
                continue
            id, range, text = tuple(line.split('\t'))
            type, st, nd = tuple(range.split())
            st, nd, text, type
            entities.append({'st':int(st), 'nd': int(nd), 'text':text.replace('’','\''), 'type':type, 'id':id})

        output = ''
        entities = sorted(entities, key= lambda x : (x['st'],-x['nd']) )
        for i,e in enumerate(entities):
            if i<len(entities)-2:
                next_e= entities[i+1]
                aft_e = entities[i+2]
                prev_e = entities[i-1]
                #si l'on par exemple ((Bouvard)PER et (Pécuchet)PER )PER
                #on saute l'entité contenante
                if e['st']<=next_e['st'] and e['nd']>=next_e['nd'] and next_e['type']==e['type'] and aft_e['text'] in e['text'] and any(x in e['text'] for x in [' et ', ' ou ', ', ', ' ni ']):
                    ctr+=1
                    continue
                #Sauter ces 3 fausses entités que j'ai remarquées
                if any(e['text'].startswith(x) and e['id']==y for x,y in \
                    zip(["Le carreau de briques","New-York","St-Pierre"],['T209','T26','T356'])):
                    ctr+=1
                    continue
                #si l'on a par exemple (Un (des paquebots)VEH)VEH
                #on saute l'entité contenue
                if prev_e['st']<=e['st'] and prev_e['nd']>=e['nd'] and e['type']==prev_e['type'] and any(prev_e['text'].lower().replace('l\'','').startswith(x+' de') for x in ['un', 'une','deux','quelques-uns']):
                    ctr+=1
                    continue
                #on saute les entités ayant un des types dans to_clean
                if e['type'] in to_clean:
                    ctr+=1
                    continue
            output+='\t'.join([e['id'],' '.join([e['type'],str(e['st']),str(e['nd'])]),e['text']])+'\n'

        #NO_PER est fusionné avec PER
        with open(fn.replace(brat_dir,outputDir+'/'),'w+') as f:
            f.write(output.replace('NO_PER','PER').replace('HIST','TIME'))

    print(ctr)
