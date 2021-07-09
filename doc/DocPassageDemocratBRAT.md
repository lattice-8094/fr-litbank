# Passage du format d'annotations DEMOCRAT au format BRAT

## Formats de départ (democrat)

Les formats annotés récupérés par l'export dans le logiciel TXM sont de la forme :

### TOTO.xml

```
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:txm="http://textometrie.org/1.0">
<teiHeader xml:lang="fr" xmlns="http://www.tei-c.org/ns/1.0">
....
</teiHeader>
<text id="TOTO">
...
<lb n="n°"></lb>
...
<s n="n°">
...
<w id="w_TOTO_nbw" n="nbw">
<txm:form>text</txm:form>
<txm:ana type="#frpos" resp="#txm">NAM</txm:ana>
<txm:ana type="#frlemma" resp="#txm">lemma</txm:ana>
</w>
...
<w id="w_TOTO_nbw" n="nbw">
<txm:form>text</txm:form>
<txm:ana type="#frpos" resp="#txm">NAM</txm:ana>
<txm:ana type="#frlemma" resp="#txm">lemma</txm:ana>
</w>
...
</text></TEI>
```

<?xml version="1.0" encoding="UTF-8"?>

## TOTO-URS.XML

```
<tei:TEI xmlns:tei="http://www.tei-c.org/ns/1.0">
<teiHeader>
...
</teiHeader>
...

<standOff>
...
<annotations type="coreference">
<annotationGrp type="Unit" subtype="MENTION">
...
<span id="u-MENTION-nbmention" from="text:w_TOTO_nbw" to="text:w_TOTO_nbw" ana="#u-MENTION-1-fs"></span>
...
</annotationGrp>
<annotationGrp type="Schema" subtype="CHAINE">
...
<link id="s-CHAINE-nbchaine" target="#u-MENTION-nbmention
 #u-MENTION-nbmention #u-MENTION-nbmention" ana="#s-CHAINE-nbchaine-fs"></link>

...
</annotationGrp>
<div type="unit-fs">

<fs id="u-MENTION-nbmention-fs">
<f name="REF"><string>le narrateur</string></f>
</fs>
...
</div>
</annotations>
</standOff>
</tei:TEI>


```

<?xml version="1.0" encoding="UTF-8"?>

## Formats d'arrivée (Brat)

Le détail est ici : [Standoff format - brat rapid annotation tool](https://brat.nlplab.org/standoff.html)

### Text file (.txt)

Texte simple du roman ou partie du roman retenu.

### Annotation file (.ann)

T -> texte

E -> Event



<div>
Tid<tab>type<espace>start-offset<espace>end-offset<tab>texte


</div>

A compléter avec les autres types d'annotations ...

## Algorithme

### Réflexions et tests

- Pour recréer le fichier texte à partir du fichier toto.xml de départ, il faudrait retrouver la feuille de style (.XSL ?) utilisée dans le logiciel TXM. On pourrait utiliser un programme simple qui concanète chaque morceau de texte (cf dans toto.xml balises w et txm:form) en ajoutant un espace entre chaque systématiquent mais on aura, par exemple des espaces avant les virgules, les points, les tirets, après les apostrophes et les dialogues seront mal retranscris, etc ...

- 

## Liens

Lire fichiers BRAT : https://brat.nlplab.org/ installé au Lattice : https://apps.lattice.cnrs.fr/brat/index.xhtml#/litbank/entities/










