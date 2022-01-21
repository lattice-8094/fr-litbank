import glob
import argparse
import os
import re
import pandas as pd
import numpy as np

if __name__ == '__main__':

    #prendre les arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='input directory containing brat files', required=True)
    parser.add_argument(
        '-o', '--output', help='output directory to write new tsv files to', required=True)

    args = vars(parser.parse_args())
    inputDir = args["input"]
    outputDir = args["output"]

    #créer le dossier de sortie si inexistant
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    #parcourir les fichier brat un par un
    for filename in glob.glob(os.path.join(inputDir, '*.ann')):
        # initialiser df, contenant les entités
        # et text, contenant le text brut
        print(filename)
        with open(filename) as f:
            content = [l.split('\t')[1].split(' ') for l in f]
            df = pd.DataFrame(content, columns=['ct','st','nd'])
            df.st = pd.to_numeric(df.st)
            df.nd = pd.to_numeric(df.nd)
            df.sort_values(['st','nd'], ascending=(True,False), inplace=True)
        with open(filename[:-3]+"txt") as f:
            text = f.read().replace('’','\'')

        #le but est d'initialiser, à partir du texte brut, un dictionnaire words
        #qui contient autant d'éléments que de mots dans le texte.
        #un élément de ce dictionnaire a comme clé l'indice de début du mot et comme valeur,
        #le dictionnaire {'next': l'indice du mot suivant,'text':le texte du mot,
        #'entities': liste contenant les entités qui contiennent ce mot (cette liste
        #est vide dans un premier temps)}
        #
        #Remarque: on sépare les "mots" sur les espaces, mais aussi sur les apostrophes,
        #les points, les virgules, et les parenthèses.
        idx = 0
        word_start_idx = 0
        words = {}
        #parcourir les caractères du texte
        while idx<=len(text):
            c = text[idx] if idx<len(text) else '.' #pour être sûr que la dernière phrase est terminée par un point
            #on ne fait rien jusqu'à ce qu'on rencontre un séparateur de mot
            #si c est un séparateur de mot :
            if c in ['\'', '.', ',','\n', '(', ')','…','–','’','[',']','-', ' ']:
                #cas particulier si c est un séparateur qui vient collé au caractère précédent, et qu'il faut donc séparer
                if c in ['.', ',', ')',']','-']:
                    words[word_start_idx] = {'next':idx, 'text':text[word_start_idx:idx], 'entities':[]}
                    word_start_idx = idx
                    idx+=1 if idx+1< len(text) and text[idx+1] == ' ' else 0 #décaler idx sauf s'il après un tiret, comme le mot suivant est collé
                #c est un séparateur, donc on coupe le mot. on ajoute le mot qu'on a jusque là
                #puis on redéfinit le word_start_idx pour que ce soit le idx actuel
                words[word_start_idx] = {'next':idx+1, 'text':text[word_start_idx:idx+1], 'entities':[]}
                word_start_idx = idx+1
            idx+=1

        #Maintenant, le but est de remplir les listes 'entities' de chaque mot, la liste
        #des étiquettes (du type "B-PER"...) qui seront à côté de chaque mot
        max_depth=0
        df.index = range(len(df))
        df['lvl'] = [0]*len(df)
        for i, ent in df.iterrows():
            for level_ctr in range(1, min(5, len(df)-i)):
                this_ent = (next_ent if level_ctr>1 else ent).copy()
                next_ent = df.loc[i+level_ctr]
                if this_ent.st <= next_ent.st < this_ent.nd:
                    ent.lvl+=1
                    max_depth = max(max_depth,ent.lvl+1)
                else:
                    break
            w = words[ent.st]
            l = len(w['entities'])
            if w['next'] != ent.nd:
                w['entities'].append(('B-'+ent.ct,ent.lvl))
            else:
                w['entities'].append(('S-'+ent.ct,ent.lvl))
            while w['next'] <= len(text) and words[w['next']]['next'] < ent.nd:
                w = words[w['next']]
                while len(w['entities'])!=l:
                    assert len(w['entities'])<l
                    w['entities'].append(('O',l-len(w['entities'])))
                w['entities'].append(('I-'+ent.ct,ent.lvl))
            w = words[w['next']]
            if w['next'] <= ent.nd or (w['next'] == ent.nd+1 and w['text'][-1]=='\n') :
                w['entities'].append(('E-'+ent.ct,ent.lvl))
            w = words[w['next']]

        output = ''
        contained_first = True
        for i,w in enumerate(list(words.values())):
            sent_end=False
            if w['text']=='\n' or w['text']=='':
                continue
            output+=w['text'].replace('\n','').replace(' ','')
            if w['text'][0]=='.' and list(words.values())[i-1]['text'] not in ['M','MM','Mm','L','etc']:
                sent_end= True
            ents = {k:v for v,k in w['entities']}
            for i in range(max_depth):
                if contained_first:
                    output+= '\t{}'.format(ents[i] if i in ents.keys() else 'O')
                else :
                    output+= '\t{}'.format(w['entities'][i][0] if i<len(w['entities']) else 'O')
            output+='\n'
            if sent_end:
                output+='\n'

        with open(os.path.join(outputDir, os.path.basename(filename)[:-3]+"tsv"), 'w', encoding='utf8') as f:
            f.write(output)
