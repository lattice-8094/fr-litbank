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
    </s>
  ...
  <s n="n°">
  ...
    <w id="w_TOTO_nbw" n="nbw">
      <txm:form>text</txm:form>
      <txm:ana type="#frpos" resp="#txm">NAM</txm:ana>
      <txm:ana type="#frlemma" resp="#txm">lemma</txm:ana>
    </w>
  </s>
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
      <div type="schema-fs"></div>
        ...
        <fs id="s-CHAINE-nbchaine-fs">
          <f name="REF"><string>texteREF</string></f>
          <f name="NB MAILLONS"><string>nbmaillons</string></f>
          <f name="TYPE REFERENT"><string>[]PER,FAC,VEH...]</string></f>
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

T -> Entity

    Named entities are phrases that contain the names of persons,
organizations, locations, times and quantities.
Example (https://www.clips.uantwerpen.be/conll2002/ner/)

> [PER Wolff ] , currently a journalist in [LOC Argentina ] , played with [PER Del Bosque ] in the final years of the seventies in [ORG Real Madrid ] .

E -> Event ...

```
Tid<tab>type<espace>start-offset<espace>end-offset<tab>texte
```

A compléter avec les autres types d'annotations ...

## Algorithme

### Feuilles de style pour l'import dans TXM

Pour recréer le fichier texte à partir du fichier toto.xml de départ, il faudrait retrouver la feuille de style (.XSL ?) utilisée dans le logiciel TXM. On pourrait utiliser un programme simple qui concanète chaque morceau de texte (cf dans toto.xml balises w et txm:form) en ajoutant un espace entre chaque systématiquent mais on aura, par exemple des espaces avant les virgules, les points, les tirets, après les apostrophes et les dialogues seront mal retranscris, etc ...

[Bibliothèque XSLT](https://txm.gitpages.huma-num.fr/textometrie/files/library/xsl/#feuilles-de-style-de-base-pour-filtrer-les-sources-xml)

**A REGARDER**

### Réflexions et tests

- 

- Une solution serait de construire une structure de tableau ( avec pandas ) avec :
  
  | texte    | nbw  | nbmention | nbchaine | TYPE REFERENT | REF      |
  | -------- | ---- | --------- | -------- | ------------- | -------- |
  | la       | 1354 | 316       | 57       | PER           | Laurence |
  | personne | 1355 | 317       | 57       | PER           | Laurence |
  | qu'      | 1362 | 318       | 57       | PER           | Laurence |
  | elle     | 1384 | 324       | 57       | PER           | Laurence |
  | ...      |      |           |          |               |          |
  | Laurence | 9399 | 2531      | 57       | PER           | Laurence |
  | ...      |      |           |          |               |          |

- Il peut y avoir plusieurs nbw pour 1 nbmention

- Il peut y avoir plusieurs nbmention pour 1 nbchaine

- TYPE REFERENT et REF sont liés à la chaine (nbchaine)

## Liens

Lire fichiers BRAT : https://brat.nlplab.org/ installé au Lattice : https://apps.lattice.cnrs.fr/brat/index.xhtml#/litbank/entities/