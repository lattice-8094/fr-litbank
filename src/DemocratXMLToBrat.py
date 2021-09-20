# -*- coding: utf-8 -*-

## Passage du format XML Democrat au format BRAT


import re
import os
from shutil import copyfile
from lxml import etree
import argparse

from democratXMLToBrat_class import Sentence, Word, Chaine, Mention, Event


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
    """
    Finds and returns the title of the document
    remove space, punctuation and diacriticals marks
    Parameters
    ----------
    xmlroot: node
        the root node

    Returns
    -------
    String
        title of the document (normalized as a filename)
    """
    title = root.find(".//tei:titleStmt/tei:title", namespaces=nsd)
    file_name = title.text.strip()
    file_name = re.sub(r' ?\((\d+)\)', r'-\1', file_name) # "fifi (1)" -> "fifi-1"
    file_name = re.sub('\s', '_', file_name) # "Mademoiselle Fifi" -> "Mademoiselle_Fifi"
    file_name = re.sub('[,;.]', '', file_name) # "Mademoiselle Fifi, nouveaux contes" -> "Mademoiselle_fifi_nouveaux_contes"
    file_name = re.sub('[Ééèêë]', 'e', file_name) # "Pécuchet" -> "Pecuchet"
    file_name = re.sub('[àâ]', 'a', file_name)
    file_name = re.sub('[ù]', 'u', file_name)
    file_name = re.sub('[ç]', 'c', file_name)
    file_name = re.sub('[ï]', 'i', file_name)
    return file_name


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
        list of Mention objects
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

def get_events(urs_root, words):
    """
    Finds and returns the 'event mentions (u-BAMMAN)' in the given urs xml node

    Parameters
    ----------
    urs_root: node
        the urs root node
    words: dict
        dict of all Word objects in text, index by word id (w.id: w)

    Returns
    -------
    List
        list of Event objects
    """
    events = []
    event_elems = urs_root.findall('.//annotationGrp[@subtype="BAMMAN"]/span', namespaces=nsd)
    for event_elem in event_elems:
        # dans une mention de type event on peut avoir plusieurs mots
        # ils sont indiqués par 'span' (id du début, id de fin) : il faut retrouver tous les mots entre les bornes
        event_words = []
        event_id = event_elem.get("id")
        span_from = event_elem.get("from")[5:]
        span_to = event_elem.get("to")[5:]
        from_int = get_mention_num_id(span_from)
        to_int = get_mention_num_id(span_to)
        for i in range(from_int, to_int + 1):
            w_id = re.sub(r"(.+)_\d+", rf"\1_{i}", span_from)
            event_words.append(words[w_id])
        events.append(Event(event_id, event_words))
        #print(event_id, ','.join([w.form for w in event_words]))
    return events

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
        list of Chaine objects
    """
    chaines = []
    schemas = urs_root.findall('.//div[@type="schema-fs"]/fs', namespaces=nsd)
    for schema in schemas:
        type_elem = schema.find('./f[@name="TYPE REFERENT"]/string', namespaces=nsd)
        if type_elem is not None:
            if type_elem.text == "EVENT":
                continue
            id = schema.get("id")[:-3]
            nb_maillons_elem = schema.find('./f[@name="NB MAILLONS"]/string', namespaces=nsd)
            if nb_maillons_elem is None:
                nb_maillons = ""
            else:
                nb_maillons = nb_maillons_elem.text
            ref_elem = schema.find('./f[@name="REF"]/string', namespaces=nsd)
            if ref_elem is None:
                ref = ""
            else:
                ref = ref_elem.text
            chaine = Chaine(id, ref, nb_maillons, type_elem.text)
            mentions = get_mentions(chaine, urs_root, words)
            chaine.mentions = mentions
            chaines.append(chaine)
    return chaines




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

    #####
    # Traitement des 'entities'
    #####
    # fichiers de sortie pour 'entities'
    entities_dir = os.path.join(args.out_dir, 'entities')
    if not os.path.exists(entities_dir):
        os.makedirs(entities_dir)
    f_brat_entities_txt = os.path.join(entities_dir, title + ".txt")
    f_brat_entities_ann = os.path.join(entities_dir, title + ".ann")

    ## Stockage des phrases (éléments 's' et des mots 'w')
    ## Écriture du fichier txt
    sentences = get_sentences(xml_root)
    with open(f_brat_entities_txt, "w") as txt:
        for s in sentences:
            print(s, end="", file=txt)

    ## Stockage des chaînes
    # on a besoin d'un dictionnaire de tous les objets 'Word' du texte
    # dictionnaire en compréhension à partir des objets 'Sentence'
    words = {w.id: w for s in sentences for w in s.content if isinstance(w, Word)}
    chaines = get_chaines(urs_root, words)
    i = 1
    with open(f_brat_entities_ann, "w") as ann:
        for chaine in chaines:
            for mention in chaine.mentions:
                if mention.is_entity():
                    print(
                        f"T{i}\t{chaine.type_referent} {mention.words[0].start} {mention.words[-1].get_end()}\t{str(mention)}",
                        file=ann,
                    )
                    i = i + 1
    
    #####
    # Traitement des 'events'
    #####
    # création du répertoire 'brat/events' si besoin
    events_dir = os.path.join(args.out_dir, 'events')
    if not os.path.exists(events_dir):
        os.makedirs(events_dir)
    # fichiers de sortie pour 'events'
    f_brat_events_txt = os.path.join(events_dir, title + ".txt")
    f_brat_events_ann = os.path.join(events_dir, title + ".ann")

    # simple copie du fichier txt des 'entities'
    copyfile(f_brat_entities_txt, f_brat_events_txt)
    ## Récupération des annotations de type 'event'
    events = get_events(urs_root, words)
    i = 1
    with open(f_brat_events_ann, "w") as ann:
        for event in events:
            print(
                        f"T{i}\tEVENT {event.words[0].start} {event.words[-1].get_end()}\t{str(event)}",
                        file=ann,
                    )
            i = i + 1

    