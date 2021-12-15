import glob
import argparse
import os
import re
import pandas as pd
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='input directory containing brat files', required=True)
    parser.add_argument(
        '-o', '--output', help='output directory to write new tsv files to', required=True)

    args = vars(parser.parse_args())
    inputDir = args["input"]
    outputDir = args["output"]

    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    for filename in glob.glob(os.path.join(inputDir, '*.ann')):
        print(filename)
        with open(filename) as f:
            content = [l.split('\t')[1].split(' ') for l in f]
            df = pd.DataFrame(content, columns=['ct','st','nd'])
            df.st = pd.to_numeric(df.st)
            df.nd = pd.to_numeric(df.nd)
            df.sort_values(['st','nd'], ascending=(True,False), inplace=True)
        with open(filename[:-3]+"txt") as f:
            text = f.read()

        idx = 0
        idx_st = 0
        words = {}
        while idx<=len(text):
            c = text[idx] if idx<len(text) else '.'
            if c in [' ','\'', '.', ',','\n', '(', ')','…','–','’','[',']']:
                if c in ['.', ',', ')',']']:
                    words[idx_st] = {'this': idx_st, 'next':idx, 'text':text[idx_st:idx], 'entities':[]}
                    idx_st = idx
                    idx+=1
                words[idx_st] = {'this': idx_st, 'next':idx+1, 'text':text[idx_st:idx+1], 'entities':[]}
                idx_st = idx+1
            idx+=1
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
            w['entities'].append(('B-'+ent.ct,ent.lvl))
            while w['next'] <= len(text) and w['next'] < ent.nd:
                w = words[w['next']]
                while len(w['entities'])!=l:
                    assert len(w['entities'])<l
                    w['entities'].append(('O',l-len(w['entities'])))
                w['entities'].append(('I-'+ent.ct,ent.lvl))
            w = words[w['next']]

        output = ''
        contained_first = True
        for w in words.values():
            sent_end=False
            if w['text']=='\n' or w['text']=='':
                continue
            output+=''+w['text'].replace('\n','').replace(' ','')+' '
            if w['text'][0]=='.':
                sent_end= True
            ents = {k:v for v,k in w['entities']}
            for i in range(max_depth):
                if contained_first:
                    output+= '{}\t'.format(ents[i] if i in ents.keys() else 'O')
                else :
                    output+= '{}\t'.format(w['entities'][i][0] if i<len(w['entities']) else 'O')
            output+='\n'
            if sent_end:
                output+='\n'

        with open(os.path.join(outputDir, os.path.basename(filename)[:-3]+"tsv"), 'w', encoding='utf8') as f:
            f.write(output)
