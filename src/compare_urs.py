# -*- coding: utf-8 -*-

## Comparaison des types de référents dans deux fichiers urs

import argparse
import pathlib
import re
import os
from lxml import etree
from dataclasses import dataclass
from jinja2 import Template


nsd = {'tei': 'http://www.tei-c.org/ns/1.0',
        'txm':  'http://textometrie.org/1.0'}

@dataclass
class TypeAnnotation:
    """
    Class to store a type annotation in booknlp fr style
    """
    id: str
    ref: str
    mention: str
    type_annotator_1: str = ""
    type_annotator_2: str = ""
    display: str = ""
    id_num: int = 0

    def __post_init__(self):
        """
        set the num attribute for sorting purpose
        """
        match = re.search('.+-(\d+)-fs$', self.id)
        if match:
            self.id_num = int(match.group(1))

    def compare(self):
        """
        Compare the annotations and sets the display char
        """
        if self.type_annotator_1 == "" or self.type_annotator_2 == "":
            self.display = "⭕"
        elif self.type_annotator_1 != self.type_annotator_2:
            self.display = "❌"
        elif self.type_annotator_1 == self.type_annotator_2:
            self.display = "✔️"

def generate_html(annotator_1, annotator_2, annotations, scores, out_file, templ_file="template.html"):
    """
    Generate the html files with scores and list of annotations
    html comes from a template file, uses jinja2 to operate with the template
    """
    templ_str = ""
    with open(templ_file) as templ:
        templ_str = templ.read()
    template = Template(templ_str)
    template.stream(annotator_1=annotator_1, annotator_2=annotator_2, annotations=annotations, scores=scores).dump(out_file)


def get_mention_num_id(mention_id):
    search = re.search('.+_(\d+)$', mention_id)
    if search:
        return int(search.group(1))

def get_doc_title(root):
    title = root.find(".//titleStmt/title", namespaces=nsd)
    return title.text.strip()

def get_annotator(root):
    name = root.find(".//soHeader//name", namespaces=nsd)
    return name.get('id')

def find_mention(schema, root, xml_root):
    """
    find and return the string value of the first mention in the schema
    """
    mention = ""
    schema_id = schema.get('id')[:-3]
    link = root.find('.//link[@id="' + schema_id + '"]', namespaces=nsd)
    targets = link.get('target')
    first_mention = targets.split(' ')[0][1:]
    first_mention_span = root.find('.//span[@id="'+ first_mention +'"]', namespaces=nsd)
    span_from = first_mention_span.get('from')[5:]
    span_to = first_mention_span.get('to')[5:]
    
    from_int = get_mention_num_id(span_from)
    to_int = get_mention_num_id(span_to)
    for i in range(from_int, to_int+1):
        id = re.sub(r'(.+)_\d+', rf"\1_{i}", span_from)
        w = xml_root.find('.//tei:w[@id="'+ id +'"]/txm:form', namespaces=nsd)
        mention += w.text + " "
    return mention

    
def get_type_annotations(annotations, root, urs_path, annotator="annotator_1"):
    """
    Find the type annotations ('REF') in the given urs document and add them
    as TypeAnnotation object in givent annotations dict
    """
    prefix_def = root.find('.//prefixDef', namespaces=nsd)
    xml_path = prefix_def.get('replacementPattern')[:-3]
    xml_tree = etree.parse(str(os.path.dirname(pathlib.Path(urs_path))) + '/' + xml_path)
    xml_root = xml_tree.getroot()

    schemas = root.findall('.//div[@type="schema-fs"]/fs', namespaces=nsd)
    for schema in schemas:
        type = schema.find('./f[@name="TYPE REFERENT"]/string', namespaces=nsd)
        if type is not None:
            id = schema.get('id')
            # si l'annotation existe déjà pour cette chaîne
            # on ajoute le type donné par l'autre annotateur
            if id in annotations:
                if annotator == "annotator_1":
                    annotations[id].type_annotator_1 = type.text
                else:
                    annotations[id].type_annotator_2 = type.text           
            else:
                ref = schema.find('./f[@name="REF"]/string', namespaces=nsd)
                if ref is None:
                    ref = etree.Element('none')
                    ref.text = ""
                mention = find_mention(schema, root, xml_root)
                if annotator == "annotator_1":
                    annotations[id] = TypeAnnotation(id, ref.text, mention, type_annotator_1=type.text)
                else:
                    annotations[id] = TypeAnnotation(id, ref.text, mention, type_annotator_2=type.text)                   
    return annotations

@dataclass
class Score:
    """
    """
    cat: str
    annotator_1: int=0
    annotator_2: int=0
    ok: int=0
    nok: int=0
    void: int=0

    def add(self, annotation):
        if annotation.type_annotator_1 == self.cat or (self.cat == 'all' and annotation.type_annotator_1 != ""):
            self.annotator_1 += 1
        if annotation.type_annotator_2 == self.cat or (self.cat == 'all' and annotation.type_annotator_2 != ""):
            self.annotator_2 += 1
        if annotation.display == '⭕':
            self.void += 1
        elif annotation.display == '❌':
            self.nok += 1
        elif annotation.display == '✔️':
            self.ok += 1
        

def compute_scores(annotations):
    scores = {}
    scores['all'] = Score('all')
    for i in annotations:
        scores['all'].add(annotations[i])
        if annotations[i].display == '✔️':
            cat = annotations[i].type_annotator_1
            if cat in scores:
                scores[cat].add(annotations[i])
            else:
                scores[cat] = Score(cat)
                print(f"add {cat} in dict")
                scores[cat].add(annotations[i])
        else:
            cat = annotations[i].type_annotator_1
            if cat != "":
                if cat in scores:
                    scores[cat].add(annotations[i])
                else:
                    scores[cat] = Score(cat)
                    print(f"add {cat} in dict")
                    scores[cat].add(annotations[i])
            cat = annotations[i].type_annotator_2
            if cat != "":
                if cat in scores:
                    scores[cat].add(annotations[i])
                else:
                    scores[cat] = Score(cat)
                    print(f"add {cat} in dict")
                    scores[cat].add(annotations[i])                
    return scores

def main():
    parser = argparse.ArgumentParser(description="Comparaison des types de référents dans deux fichiers urs")
    parser.add_argument(
        "urs_1",
        metavar="urs_1/myfile-urs.xml",
        type=str,
        help="le fichier -urs.xml de l'annotateur 1",
    )
    parser.add_argument(
        "urs_2",
        metavar="urs_2/myfile-urs.xml",
        type=str,
        help="le fichier -urs.xml de l'annotateur 2",
    )

    args = parser.parse_args()
   
    xml_tree_1 = etree.parse(args.urs_1)
    xml_root_1 = xml_tree_1.getroot()

    xml_tree_2 = etree.parse(args.urs_2)
    xml_root_2 = xml_tree_2.getroot()

    annotator_1 = get_annotator(xml_root_1)
    annotator_2 = get_annotator(xml_root_2)

    title = get_doc_title(xml_root_1)
    out_file = f"{title}_{annotator_1}_{annotator_2}.html"
    annotations = {}
    annotations = get_type_annotations(annotations, xml_root_1, args.urs_1, annotator="annotator_1")
    print(f"annotateur 1: {len(annotations)}")
    annotations = get_type_annotations(annotations, xml_root_2, args.urs_2, annotator="annotator_2")
    print(f"annotateur 2: {len(annotations)}")
    for i in annotations:
        annotations[i].compare()

    scores = compute_scores(annotations)
    generate_html(annotator_1, annotator_2, annotations.values(), scores.values(), out_file)

if __name__ == "__main__":
    main()
