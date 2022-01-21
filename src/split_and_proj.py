import re
import glob
import os
import argparse

if __name__ == '__main__':

    #prendre les arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='input directory containing brat files', required=True)

    args = vars(parser.parse_args())
    tsv_dir = args["input"]

    dev_titles = ["Bouvard_et_Pecuchet", "De_la_ville_au_moulin", "Douce_Lumiere"]
    test_titles = ["elisabeth_Seton", "La_morte_amoureuse", "Le_capitaine_Fracasse"]

    corpus = {"train":'', "dev":'', "test":''}
    for fn in glob.glob(os.path.join('.', tsv_dir+'/*.tsv')):
        print(fn)
        with open(fn,'r') as f:
            if any(x in fn for x in dev_titles):
                corpus['dev']+=f.read()
            elif any(x in fn for x in test_titles):
                corpus['test']+=f.read()
            else:
                corpus['train']+=f.read()

    for split,data in corpus.items():
        s_out = ''
        total=0
        for l in data.split('\n'):
            if l =='':
                s_out+='\n'
                continue
            elems = l.split('\t')
            ents = [ent for ent in elems[1:] if ent!='O']
            s_out+= elems[0]+' '
            if len(ents)==0 or ents[0]=='':
                s_out+='O'
            else:
                s_out+= ents[0]
            s_out+='\n'

        with open('{}/{}.txt'.format(tsv_dir,split), "w+") as f:
            f.write(s_out)
