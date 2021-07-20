# -*- coding: utf-8 -*-

## Passage du format XML Democrat au format BRAT


import re
import os
from lxml import etree
import argparse




nsd = {'tei': 'http://www.tei-c.org/ns/1.0',
        'txm':  'http://textometrie.org/1.0'}

# -----------------------------------------------
def get_w_id_text(xmlroot):

    d={}
    textecomplet=""
    niv1=etree.XML(etree.tostring(xmlroot)) # renvoie les 2 éléments niv1 : teiHeader et text
    niv2=etree.XML(etree.tostring(niv1[1])) # niv2 = contenu de <text></text>

    for s in niv2:
        if s.tag.endswith('s'):
            niv3=etree.XML(etree.tostring(s))
            for w in niv3:
                if w.tag.endswith('w'):
                    nbw=int(w.get("n"))
                    text=w[0].text # w[0] = <txm:form> child
                    start_offset=len(textecomplet)+1
                    # la création du texte complet sera à revoir selon la feuille de style
                    textecomplet += '{} '.format(text)
                    end_offset=len(textecomplet)+1
                    d[nbw]={'texte':w[0].text,'start':start_offset,'end':end_offset}


    #return({107:{"texte":"c'"},108:{"texte":"était"}})
    return(d,textecomplet)
# -----------------------------------------------
def get_mentions_w_id(ursroot):
    d={}
    niv1=etree.XML(etree.tostring(ursroot)) # renvoie les 3 éléments niv1 : teiHeader, soHeader et standOff
    niv2=etree.XML(etree.tostring(niv1[2])) # on prend le 3ème élément niv1[2]-> standOff
    niv3=etree.XML(etree.tostring(niv2[1])) # on prend le 2ème élément de standOff -> annotations

    # niv3 a 5 éléments : annotationsGrp type Unit, annotationsGrp type Schema, et 3 div type unit-fs, relation-fs et schema-fs
    # ici on s'intéresse au 1er élément : annotationsGrp type Unit
    u=etree.XML(etree.tostring(niv3[0]))
    for m in u:
        ident=int(m.get("id").split('-')[-1])
        start=int(m.get("from").split('_')[-1])
        end=int(m.get("to").split('_')[-1])
        if start <= end:
            d[ident]=[i for i in range(start,end+1)]
        
            # ne pas tenir copte des erreurs from > to
            # ex : <span id="u-MENTION-111" from="text:w_FC_NAR_EXT_181Pauline_brut_PRIS_PAR_MARINE_499" to="text:w_FC_NAR_EXT_181Pauline_brut_PRIS_PAR_MARINE_488" ana="#u-MENTION-111-fs"></span>    

    return(d)
    #return({27:[119,120,121,122],29:[128,129,130,131]})
# -----------------------------------------------
def get_chaines(ursroot):
    d1={}
    d2={}
    niv1=etree.XML(etree.tostring(ursroot)) # renvoie les 3 éléments niv1 : teiHeader, soHeader et standOff
    niv2=etree.XML(etree.tostring(niv1[2])) # on prend le 3ème élément niv1[2]-> standOff
    niv3=etree.XML(etree.tostring(niv2[1])) # on prend le 2ème élément de standOff -> annotations
    # niv3 a 5 éléments : annotationsGrp type Unit, annotationsGrp type Schema, et 3 div type unit-fs, relation-fs et schema-fs

    # On va d'abord récupérer les chaines qui ont un TYPE REFERENT
    # Ensuite, on y ajoute les mentions correspondantes

    s=etree.XML(etree.tostring(niv3[4])) # on prend le 4ème élément de annotations : schema-fs
    for r in s:
        if len(r)==3: # on doit avoir <f name="REF">, <f name="NB MAILLONS"> et <f name="TYPE REFERENT">
            ident=int(r.get("id").split('-')[-2]) # le n° de chaine n'est pas en dernier mais avant dernier -> -2
            if r[2].get("name")=="TYPE REFERENT":
                #d1[ident]={"type":r[2][0].text} # on récupère le contenu de <string></string>
                d1[ident]=r[2][0].text 


    # ici on s'intéresse au 2nd élément : annotationsGrp type Schema
    u=etree.XML(etree.tostring(niv3[1]))
    # u contient des éléments de type <link id="s-CHAINE-nbchaine" target="#u-MENTION-nbmention1 #u-MENTION-nbmention2 #u-MENTION-nbmention3" ana="#s-CHAINE-xx-fs"></link>
    for c in u:
        ident=int(c.get("id").split('-')[-1])
        if ident in d1.keys():
            l=[]
            listementions=c.get("target").split(' ')
            listenbmentions=[int(i.split('-')[-1]) for i in listementions]
            #d2[ident]={"mentions":listenbmentions}
            d2[ident]=listenbmentions
        

    return(d1,d2)

    #return({19:{"mentions":[21,20,23,22,19]},25:{"mentions":[210,29,448,449,306,678]}})



# -----------------------------------------------
def get_ann(d1,d2,d3):
    # A FAIRE
    
    return("T1    Organization 0 4    Sony\nT2  MERGE-ORG 14 27 joint venture\nT3  Organization 33 41  Ericsson\nE1  MERGE-ORG:T2 Org1:T1 Org2:T3\n")  
# -----------------------------------------------
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Passage du format Democrat XML au format BRAT")
    print(parser)
    ... reprendre dans compare_urs.py
    '''
    # Fichiers d'entrée
    path1="../xml/"
    path2="../urs-xml/"
    f_xml=path1+'FC_NAR_EXT_18-1-Pauline_brut_PRIS_PAR_MARINE.xml'
    f_ursxml=path2+'pauline-urs.xml'
    # fichiers de sortie
    path3="../brat/"
    f_brat_txt=path3+"Pauline.txt"
    f_brat_ann=path3+"Pauline.ann"


    xml_tree = etree.parse(f_xml)
    xml_root = etree.XML(etree.tostring(xml_tree))

    ursxml_tree = etree.parse(f_ursxml)
    ursxml_root = ursxml_tree.getroot()

    # récupération des informations du XML et URS_XML DEMOCRAT et création des structures de données
    # ----------------------------------------------------------------------------------------------

    
    # on crée le dictionnaire avec le texte et les offset (les offset doivent être créés ici car à partir des mentions c'est impossible : 2 mentions peuvent contenir le même w id)
    w_id_text_offsets, brat_txt=get_w_id_text(xml_root)

    # on créé les mentions
    # A NOTER : 2 mentions différents peuvent avoir les même w id !
    # ex la mention 3350 dans Pauline va du w id 12281 au w id 12283 et la mention 3351 contient le w id 12282 
    mentions_w_id={}
    mentions_w_id=get_mentions_w_id(ursxml_root)

    chaines={}
    # on créé le dictionnaire avec le TYPE REFERENT et les mentions
    # A NOTER : chaines ne contient que les chaines qui ont un TYPE REFERENT
    typeref,mentions=get_chaines(ursxml_root)

    for k in typeref.keys():
        chaines[k]={"type":typeref[k],"mentions":mentions[k]}


    # création des fichiers BRAT
    # --------------------------

    # on complète le dictionnaire w_id_text_offsets avec les start-offset et end_offset
    # cette fonction changera en fonction de la feuille de style de TXM
    # --- juste pour test 
    brat_text2=""
    for w in list(sorted(w_id_text_offsets.keys())):
        brat_text2 += '{} '.format(w_id_text_offsets[w]['texte'])
    # ---    
    

    # Création fichier brat .ann 
    # A FINIR
    brat_ann=get_ann(w_id_text_offsets,mentions_w_id,chaines)

    # Création des fichiers de sortie

    f_txt=open(f_brat_txt,"w")
    f_txt.write(brat_txt)
    f_txt.close()
    f_ann=open(f_brat_ann,"w")
    f_ann.write(brat_ann)
    f_ann.close()