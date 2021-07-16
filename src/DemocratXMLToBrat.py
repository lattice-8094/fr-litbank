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
    niv1=etree.XML(etree.tostring(xmlroot)) # renvoie les 2 éléments niv1 : teiHeader et text
    niv2=etree.XML(etree.tostring(niv1[1])) # niv2 = contenu de <text></text>

    for s in niv2:
        if s.tag.endswith('s'):
            niv3=etree.XML(etree.tostring(s))
            for w in niv3:
                if w.tag.endswith('w'):
                    nbw=int(w.get("n"))
                    text=w[0].text # w[0] = <txm:form> child
                    d[nbw]={"texte":text}


    #return({107:{"texte":"c'"},108:{"texte":"était"}})
    return(d)
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
        d[ident]=[i for i in [start,end]]

    return(d)
    #return({27:[119,120,121,122],29:[128,129,130,131]})
# -----------------------------------------------
def get_chaines_mentions_id(ursroot):
    # A FAIRE
    return({19:{"mentions":[21,20,23,22,19]},25:{"mentions":[210,29,448,449,306,678]}})
# -----------------------------------------------
def get_chaines_typereferent(ursroot, ch):
    # A FAIRE
    return({19:{"mentions":[21,20,23,22,19],"type":"EVENT"},25:{"mentions":[210,29,448,449,306,678],"type":"FAC"}})
# -----------------------------------------------
def get_textebrat_offset(d):
    # A FAIRE
    return("Il était une fois patati ... patata ",{107:{"texte":"c'","offsets":[1,4]},108:{"texte":"était","offsets":[24,42]}})

# -----------------------------------------------
def get_ann(d1,d2,d3):
    # A FAIRE
    return("T1\tPER 24 32\tLaurence\nT2\tPER 90 97\tPauline")   
# -----------------------------------------------
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Passage du format Democrat XML au format BRAT")
    print(parser)
    ... reprendre dans compare_urs.py
    '''
    path1="../xml/"
    path2="../urs-xml/"
    f_xml=path1+'FC_NAR_EXT_18-1-Pauline_brut_PRIS_PAR_MARINE.xml'
    f_ursxml=path2+'pauline-urs.xml'

    xml_tree = etree.parse(f_xml)
    xml_root = etree.XML(etree.tostring(xml_tree))

    ursxml_tree = etree.parse(f_ursxml)
    ursxml_root = ursxml_tree.getroot()

    # récupération des informations du XML et URS_XML DEMOCRAT et création des structures de données
    # ----------------------------------------------------------------------------------------------

    w_id_text_offsets={}
    # on crée le dictionnaire avec un sous-dictionnaire texte
    w_id_text=get_w_id_text(xml_root)

    mentions_w_id={}
    mentions_w_id=get_mentions_w_id(ursxml_root)

    chaines={}
    # on créé le dictionnaire avec un sous-dictionnaire contenant les mentions pour chaque chaine
    chaines=get_chaines_mentions_id(ursxml_root)
    # on complète avec le TYPE REFERENT 
    chaines=get_chaines_typereferent(ursxml_root, chaines)

    # création des fichiers BRAT
    # --------------------------

    # on complète le dictionnaire w_id_text_offsets avec les start-offset et end_offset
    # cette fonction chnagera en fonction de la feuille de style de TXM
    textebrat=""
    textebrat,w_id_text_offsets=get_textebrat_offset(w_id_text_offsets)

    annbrat=""
    annbrat=get_ann(w_id_text_offsets,mentions_w_id,chaines)

    # Création fichiers brat .txt et .ann à partir de textebrat et annbrat
    # A finir