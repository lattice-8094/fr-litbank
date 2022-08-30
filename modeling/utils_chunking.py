import glob
import sys
import argparse
import os
import re
import pandas as pd
import numpy as np
from itertools import groupby
from collections import Counter

def chunk_brat(inputDir, outputDir, interval, bioes, max_seq_len, contained_first=True, coref_pred=True, labels_to_ignore=[],labels_to_replace=[]):
    #créer le dossier de sortie si inexistant
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)
    
    print("Conversion des fichiers brat en tsv :")
    #parcourir les fichier brat un par un
    for filename_txt in glob.glob(os.path.join(inputDir, '*.txt')):
        # initialiser df, contenant les entités
        # et text, contenant le text brut
        filename = filename_txt[:-3]+"ann"
        if os.path.isfile(filename):
            with open(filename) as f:
                content_with_limits = [(l.split('\t')[0],
                                        l.split('\t')[1].split(' '))
                                        for l in f if l.startswith('T')]
                content = [[idx ,c[0], c[1], c[-1]] for idx,c in content_with_limits]
                #print(content)
                df = pd.DataFrame(content, columns=['idx','ct','st','nd'])
                df.st = pd.to_numeric(df.st)
                df.nd = pd.to_numeric(df.nd)
                for rep_rule in labels_to_replace:
                    parts=rep_rule.split(':')
                    if len(parts)!=2:
                        print("""{} est une règle de remplacement invalide.\nSi vous voulez remplace ABCD par EFGH, entrez "ABCD:EFGH".""".format(rep_rule))
                    else:
                        old,new=tuple(parts)
                        df.ct = df.ct.apply(lambda s : s.replace(old,new))
                #df.ct = df.ct.apply(lambda s : s.split('_')[-1])
                #On trie les entités selon par ordre croissant de début.
                #Si plusieurs entités commencent en même temps,
                #celle qui se termine en dernier passe en premier,
                #Cela est utile plus tard
                df.sort_values(['st','nd'], ascending=(True,False), inplace=True)
            with open(filename) as f:
                if coref_pred:
                    links = [tuple(l.split('\t')[1]
                                              .replace('Coreference ','')
                                              .replace('Arg1:','')
                                              .replace('Arg2:','')
                                              .replace('\n','')
                                              .split(' '))
                            for l in f if l.startswith('R')]
                    links_dict = dict(links)
                    indices_x = df['idx']
                    indices_y = []
                    for ind_x in indices_x:
                        if ind_x in links_dict:
                            indices_y.append(links_dict[ind_x])
                        else:
                            indices_y.append(ind_x)
                    df['coref'] = indices_y

        else:
            df= pd.DataFrame({'ct' : [], 'coref' : [], 'st' : [], 'nd' : []})
        with open(filename_txt) as f:
            text = f.read().replace('’','\'').replace(' ',' ').replace('\n',' ').replace('\t',' ')

        df = df.loc[~df['ct'].isin(labels_to_ignore)]
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
        par_ctr = 0
        words = {}
        ind_in_par=1
        #parcourir les caractères du texte
        while idx<=len(text):
            c = text[idx] if idx<len(text) else '.' #pour être sûr que la dernière phrase est terminée par un point
            #on ne fait rien jusqu'à ce qu'on rencontre un séparateur de mot
            #si c est un séparateur de mot :
            word_seps = ['\'', '.', ',','\n', '(', ')','…','–','’','[',']','-',' ']
            if c in word_seps:
                #cas particulier si c est un séparateur qui vient collé au caractère précédent, et qu'il faut donc séparer
                if c in ['.', ',', ')',']','-','…']:
                    words[word_start_idx] = {'next':idx, 'text':text[word_start_idx:idx], 'entities':[], 'par_ind': par_ctr, 'ind_in_par':ind_in_par}
                    word_start_idx = idx
                    idx+=1 if idx+1< len(text) and text[idx+1] == ' ' else 0 #décaler idx sauf s'il après un tiret, comme le mot suivant est collé
                    ind_in_par+=1
                #c est un séparateur, donc on coupe le mot. on ajoute le mot qu'on a jusque là
                #puis on redéfinit le word_start_idx pour que ce soit le idx actuel
                words[word_start_idx] = {'next':idx+1, 'text':text[word_start_idx:idx+1], 'entities':[], 'par_ind': par_ctr, 'ind_in_par':ind_in_par}
                word_start_idx = idx+1
                ind_in_par+=1
            idx+=1

        #Maintenant, le but est de remplir les listes 'entities' de chaque mot, la liste
        #des étiquettes (du type "B-PER"...) qui seront à côté de chaque mot
        max_depth=1
        df.index = range(len(df))
        df['lvl'] = [0]*len(df)
        references_connues = {i:{} for i in range(par_ctr+1)}
        for i, ent in df.iterrows():
            #Si une entité A contient une entité B,
            #B suit forcément A dans df, grâce au tri défini ci-dessus.
            #On vérifie alors pour chaque entité si elle contient l'entité
            #suivante, et ce, jusqu'à 5 niveaux d'imbrication.
            for level_ctr in range(1, min(5, len(df)-i)):
                this_ent = (next_ent if level_ctr>1 else ent).copy()
                next_ent = df.loc[i+level_ctr]
                if this_ent.st <= next_ent.st < this_ent.nd: #this_ent contient next_ent
                    ent.lvl+=1
                    max_depth = max(max_depth,ent.lvl+1)
                else: #plus d'entités imbriquées
                    break
            #ent.lvl indique maintenant pour chaque entité son niveau d'imbrication
            #on accède successivement aux mots de l'entités
            w = words[ent.st] if ent.st in words else words[ent.st+1] #on accède aux mots par l'indice de leur premier caractère
            par_ind = w ['par_ind']
            reference = ent.coref if coref_pred else ''
            if reference not in references_connues[par_ind].keys():
                references_connues[par_ind][reference] = reference
                ref_id = reference
            else :
                ref_id = references_connues[par_ind][reference]
            w ['coref'] = ref_id

            l = len(w['entities'])
            def next_word(w):
                nb_spaces = len(w['text']) - len(w['text'].replace('\n','').replace(' ',''))
                return w['next']-nb_spaces

            if next_word(w) != ent.nd or not bioes:
                w['entities'].append(('B-'+ent.ct,ent.lvl))
            else:
                w['entities'].append(('S-'+ent.ct,ent.lvl))
            while w['next'] <= len(text) and next_word(words[w['next']]) < ent.nd:
                w = words[w['next']]
                while len(w['entities'])!=l:
                    #remplir "les trous" à côté de l'entité contenue par des O
                    assert len(w['entities'])<l
                    w['entities'].append(('O',l-len(w['entities'])))
                w ['coref'] = ref_id
                w['entities'].append(('I-'+ent.ct,ent.lvl))
            
            w = words[w['next']]
            if next_word(w) <= ent.nd:
                w ['coref'] = ref_id
                if bioes :
                    w['entities'].append(('E-'+ent.ct,ent.lvl))
                else:
                    w['entities'].append(('I-'+ent.ct,ent.lvl))
            w = words[w['next']]

        new_words = {k:v for k,v in words.items() if v['text'] not in ['', ' ', '\n','\t']}
        words = new_words
        print(filename, 'présentant une entité tous les',len(words)//len(df),'mots.')
        #Chaque mot possède sa liste 'entities', il faut remplir le fichier tsv
        output = ''
        for k in range(0,len(words),interval):
            chunk_refs_connues = {}
            word_ctr=0
            for w in list(words.values())[k:k+max_seq_len]:
                word_ctr+=1
                output+=w['text'].replace('\n','').replace(' ','')
                ents = {k:v for v,k in w['entities']}
                for i in range(max_depth):
                    if contained_first:
                        output+= '\t{}'.format(ents[i] if i in ents.keys() else 'O')
                    else :
                        output+= '\t{}'.format(w['entities'][i][0] if i<len(w['entities']) else 'O')
                if 'coref' in w and ents[min(ents.keys())][0] in ['B','S']:
                        if w['coref'] not in chunk_refs_connues :
                            chunk_refs_connues[w['coref']] = word_ctr
                        output+= '\t'+str(chunk_refs_connues[w['coref']])
                else:
                    output+= '\t-'
                output+='\n'
            output+='\n'

        with open(os.path.join(outputDir, os.path.basename(filename)[:-3]+"tsv"), 'w', encoding='utf8') as f:
            f.write(output)
    print('==========')

def write_chunk_and_all_predictions(sentences, filename, chunk_int):
    words = {}
    ents = {}
    refs = {}
    with open(filename, "w") as writer:
        for i,s in enumerate(sentences):
            words_idx=i*chunk_int
            for j,(w, ent, ref) in enumerate(s):
                words[words_idx] = w
                if words_idx in ents:
                    ents[words_idx].append(ent)
                else:
                    ents[words_idx] = [ent]
                ref_offset = j-int(ref)+1 if ref!='O' else ref
                if words_idx in refs:
                    refs[words_idx].append(ref_offset)
                else:
                    refs[words_idx] = [ref_offset]
                words_idx += 1
                writer.write(f'{w} {ent} {ref_offset} \n')
            writer.write('\n\n')

    def select_ent(lst, i):
        lst = [x for x in lst if x!='O']
        if lst ==[]:
            return 'O'
        data = Counter(lst)
        return max(lst, key=data.get)

    def listify(dico, default, end):
        l = []
        for i in range(end):
            if i in dico:
                if isinstance(dico[i], list):
                    l.append(select_ent(dico[i],i))
                else :
                    l.append(dico[i])
            else :
                l.append(default)
        return(l)

    actual_ents = listify(ents, 'O', words_idx)
    actual_words = listify(words, '', words_idx)

    def select_ref(lst, i):
        if (lst.count('O')>0.6*len(lst) and actual_ents[i]=='O') or lst.count('O')==len(lst):
            return 'O'
        lst = [x for x in lst if x!='O']
        if lst[0]==0:
            return 0
        lst = [x for x in lst if x!=0]
        data = Counter(lst)
        return max(lst, key=data.get)

    def read_name(start):
        if actual_ents[start][0]=='S' and (start+1>=len(actual_ents) or actual_ents[start+1][0]=='O'):
            return words[start]
        name = words[start]
        j = start
        nb_unclosed_entits=1
        while nb_unclosed_entits>0 and j+1< len(actual_ents) and actual_ents[j+1][0] not in ['O']:
            j+=1
            if actual_ents[j][0]=='E':
                nb_unclosed_entits-=1
                if j+1<len(actual_ents) and actual_ents[j+1][0]=='S':
                    nb_unclosed_entits=0
            elif actual_ents[j][0]=='B':
                nb_unclosed_entits+=1
            name+=' ' +  words[j]
        return name

    c = Counter([x for x in actual_words if x!='' and x[0].isupper()])

    previous_refs = {}
    for i in range(words_idx):
        if i in words:
            ref_offset = select_ref(refs[i],i)
            if ref_offset != 'O':
                if i-ref_offset in previous_refs:
                    previous_refs[i] = previous_refs[i-ref_offset]
                else:
                    previous_refs[i] = read_name(i-ref_offset)

    with open(filename.replace('_chunk_', '_'),"w+") as f:
        for i in range(words_idx):
            if i in words:
                ent = actual_ents[i]
                ent = '' if ent == 'O' else ' '+ent
                ref = (' <'+previous_refs[i]+'>') if i in previous_refs else ''
                f.write(f'{words[i]}{ent}{ref}\n')
