# -*- coding: utf-8 -*-

## Passage du format XML Democrat au format BRAT


import re
import os
from lxml import etree
import argparse

from democratXMLToBrat_class import Sentence, Word, Chaine, Mention


nsd = {"tei": "http://www.tei-c.org/ns/1.0", "txm": "http://textometrie.org/1.0"}


def get_sentence_content(sentence_elem):
    """
    Finds and retrieves all 'w' elements as 
    Word objects in the given sentence element

    Parameters
    ----------
    sentence_elem: node
        a sentence element

    Returns
    -------
    List
        list of Word objects
    """
    res = []
    for child in sentence_elem:
        if child.tag == "{http://www.tei-c.org/ns/1.0}w":
            form = child.find("./txm:form", namespaces=nsd).text
            pos = child.find('./txm:ana[@type="#frpos"]', namespaces=nsd).text
            lemma = child.find('./txm:ana[@type="#frlemma"]', namespaces=nsd).text
            word = Word(child.get("id"), form, pos, lemma)
            res.append(word)
        elif child.tag == "{http://www.tei-c.org/ns/1.0}lb":
            res.append("\n")
    return res


def get_sentences(xmlroot):
    """
    Finds and retrieves all 's' elements in 
    the given xml root node

    Parameters
    ----------
    xmlroot: node
        the root node

    Returns
    -------
    List
        list of Sentence objects
    """
    sentences = []
    s_elems = xmlroot.findall(".//tei:text//tei:s", namespaces=nsd)
    start = 0
    for s_elem in s_elems:
        id = s_elem.get("n")
        s = Sentence(id, start)
        s.set_content(get_sentence_content(s_elem))
        sentences.append(s)
        start = s.get_end()
    return sentences


def get_doc_title(root):
    title = root.find(".//tei:titleStmt/tei:title", namespaces=nsd)
    return title.text.strip()


def get_mention_num_id(mention_id):
    search = re.search(".+_(\d+)$", mention_id)
    if search:
        return int(search.group(1))


def get_mentions(chaine, urs_root, words):
    """
    Finds and returns the 'mentions' in the given 'chaine'

    Parameters
    ----------
    chaine: Chaine
        the Chaine object
    urs_root: node
        the urs root node
    words: dict
        dict of all Word objects in text, index by word id (w.id: w)

    Returns
    -------
    List
        list of Mentions objects
    """
    mentions = []
    link = urs_root.find('.//link[@id="' + chaine.id + '"]', namespaces=nsd)
    targets = link.get("target")
    mentions_ids = [
        it[1:] for it in targets.split(" ")
    ]  #  on supprime le '#' initial des ids des mentions
    # on peut avoir plusieurs mentions par chaîne
    for mention_id in mentions_ids:
        # dans une mention on peut avoir plusieurs mots
        # ils sont indiqués par 'span' (id du début, id de fin) : il faut retrouver tous les mots entre les bornes
        mention_words = []
        mention_ref_elem = urs_root.find(
            './/fs[@id="' + mention_id + '-fs"]/f[@name="REF"]/string', namespaces=nsd
        )
        mention_ref = mention_ref_elem.text
        mention_span = urs_root.find(
            './/span[@id="' + mention_id + '"]', namespaces=nsd
        )
        span_from = mention_span.get("from")[5:]
        span_to = mention_span.get("to")[5:]
        from_int = get_mention_num_id(span_from)
        to_int = get_mention_num_id(span_to)
        for i in range(from_int, to_int + 1):
            w_id = re.sub(r"(.+)_\d+", rf"\1_{i}", span_from)
            mention_words.append(words[w_id])
        mentions.append(Mention(mention_id, mention_ref, mention_words))
        # print(chaine.id, mention_id, mention_ref, ','.join([w.form for w in mention_words]))
    return mentions


def get_chaines(urs_root, words):
    """
    Finds and retrieves all 'chaines' in 
    the given urs xml root node

    Parameters
    ----------
    ursroot: node
        the root node
    words: dict
        dict of all Word objects in text, index by word id (w.id: w)

    Returns
    -------
    List
        list of Chaine bjects
    """
    chaines = []
    schemas = urs_root.findall('.//div[@type="schema-fs"]/fs', namespaces=nsd)
    for schema in schemas:
        type = schema.find('./f[@name="TYPE REFERENT"]/string', namespaces=nsd)
        if type is not None:
            id = schema.get("id")[:-3]
            nb_maillons = schema.find('./f[@name="NB MAILLONS"]/string', namespaces=nsd)
            ref = schema.find('./f[@name="REF"]/string', namespaces=nsd)
            if ref is None:
                ref = etree.Element("none")
                ref.text = ""
            chaine = Chaine(id, ref.text, nb_maillons.text, type.text)
            mentions = get_mentions(chaine, urs_root, words)
            chaine.mentions = mentions
            chaines.append(chaine)
    return chaines


# -----------------------------------------------
def get_mentions_w_id(ursroot):
    d = {}
    niv1 = etree.XML(
        etree.tostring(ursroot)
    )  # renvoie les 3 éléments niv1 : teiHeader, soHeader et standOff
    niv2 = etree.XML(
        etree.tostring(niv1[2])
    )  # on prend le 3ème élément niv1[2]-> standOff
    niv3 = etree.XML(
        etree.tostring(niv2[1])
    )  # on prend le 2ème élément de standOff -> annotations

    # niv3 a 5 éléments : annotationsGrp type Unit, annotationsGrp type Schema, et 3 div type unit-fs, relation-fs et schema-fs
    # ici on s'intéresse au 1er élément : annotationsGrp type Unit
    u = etree.XML(etree.tostring(niv3[0]))
    for m in u:
        ident = int(m.get("id").split("-")[-1])
        start = int(m.get("from").split("_")[-1])
        end = int(m.get("to").split("_")[-1])
        if start <= end:
            d[ident] = [i for i in range(start, end + 1)]

            # ne pas tenir copte des erreurs from > to
            # ex : <span id="u-MENTION-111" from="text:w_FC_NAR_EXT_181Pauline_brut_PRIS_PAR_MARINE_499" to="text:w_FC_NAR_EXT_181Pauline_brut_PRIS_PAR_MARINE_488" ana="#u-MENTION-111-fs"></span>

    return d
    # return({27:[119,120,121,122],29:[128,129,130,131]})


# -----------------------------------------------
"""def get_chaines(ursroot):
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
"""


# -----------------------------------------------
def get_ann(d1, d2, d3):
    # d1= dico des chaines; d2 = dico des mentions; d3 = dico des w id
    texte = ""
    # Pour chaque chaine, si elle n'a qu'une mention(on commence par le cas simple) et si elle est de type PER, GPE, FAC, LOC
    ind_T = 1  # indice pour le TAG T
    for c in sorted(d1.keys()):
        if len(d1[c]["mentions"]) == 1 and d1[c]["type"] in [
            "PER",
            "GPE",
            "FAC",
            "LOC",
            "ORG",
        ]:
            # la chaine ne contient qu'une mention
            text_mention = ""
            m = d1[c]["mentions"][0]
            start_offset = d3[d2[m][0]][
                "start"
            ]  # on récupère la position de départ du premier w id de la seule mention de la chaine
            for w in d2[m]:
                text_mention += "{} ".format(d3[w]["texte"])
            end_offset = d3[w]["end"]
            texte += (
                "T"
                + str(ind_T)
                + "\t"
                + d1[c]["type"]
                + " "
                + str(start_offset)
                + " "
                + str(end_offset)
                + " "
                + text_mention
                + "\n"
            )
            ind_T += 1
    return texte
    # return("T1    Organization 0 4    Sony\nT2  MERGE-ORG 14 27 joint venture\nT3  Organization 33 41  Ericsson\nE1  MERGE-ORG:T2 Org1:T1 Org2:T3\n")


# -----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Passage du format Democrat au format Brat"
    )
    parser.add_argument(
        "xml", type=str, help="le fichier .xml",
    )
    parser.add_argument(
        "urs", type=str, help="le fichier -urs.xml",
    )
    parser.add_argument(
        "--out_dir", default="../brat/", type=str, help="répertoire de sortie",
    )

    args = parser.parse_args()

    # Fichiers d'entrée
    f_xml = args.xml
    f_ursxml = args.urs

    xml_tree = etree.parse(f_xml)
    xml_root = xml_tree.getroot()
    title = get_doc_title(xml_root)

    ursxml_tree = etree.parse(f_ursxml)
    urs_root = ursxml_tree.getroot()

    # fichiers de sortie
    f_brat_txt = args.out_dir + title + ".txt"
    f_brat_ann = args.out_dir + title + ".ann"

    ## Stockage des phrases (éléments 's' et des mots 'w')
    ## Écriture du fichier txt
    sentences = get_sentences(xml_root)
    with open(f_brat_txt, "w") as txt:
        for s in sentences:
            print(s, end="", file=txt)

    ## Stockage des chaînes
    # on a besoin d'un dictionnaire de tous les objets 'Word' du texte
    # dictionnaire en compréhension à partir des objets 'Sentence'
    words = {w.id: w for s in sentences for w in s.content if isinstance(w, Word)}
    chaines = get_chaines(urs_root, words)
    i = 1
    with open(f_brat_ann, "w") as ann:
        for chaine in chaines:
            for mention in chaine.mentions:
                if mention.is_entity():
                    print(
                        f"T{i}\t{chaine.type_referent} {mention.words[0].start} {mention.words[-1].get_end()}\t{str(mention)}",
                        file=ann,
                    )
                    i = i + 1

    """
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
    
    brat_ann=get_ann(chaines, mentions_w_id,w_id_text_offsets)
    

    # Création des fichiers de sortie

    f_txt=open(f_brat_txt,"w")
    f_txt.write(brat_txt)
    f_txt.close()
    f_ann=open(f_brat_ann,"w")
    f_ann.write(brat_ann)
    f_ann.close()
    """
