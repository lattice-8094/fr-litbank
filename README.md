# fr-litbank
## A french LitBank corpus

We use a subset of the files from [Democrat](https://www.ortolang.fr/market/corpora/democrat/) project to establish a corpus of french litterature from the XIX and XX centuries. In addition to this selection from Democrat,
we added two short stories by Balzac (\*).The files are annotated following the [LitBank](https://github.com/dbamman/litbank) guidelines.


## Corpus

|Date|Author|Title|
|---|---|---|
|1830|Honoré de Balzac|La maison du chat qui pelote*|
|1830|Honoré de Balzac|Sarrasine|
|1836|Théophile Gautier|La morte amoureuse|
|1837|Honoré de Balzac|La maison Nucingen*|
|1841|George Sand|Pauline|
|1856|Victor Cousin|Madame de Hautefort|ok|ok|			
|1863|Théophile Gautier|Le capitaine Fracasse|
|1873|Émile Zola|Le ventre de Paris|
|1881|Gustave Flaubert|Bouvard et Pécuchet|
|1882-1883|Guy de Maupassant|Mademoiselle Fifi, nouveaux contes (1)|
|1882-1883|Guy de Maupassant|Mademoiselle Fifi, nouveaux contes (2)|
|1882-1883|Guy de Maupassant|Mademoiselle Fifi, nouveaux contes (3)|
|1901|Lucie Achard|Rosalie de Constant, sa famille et ses amis|
|1903|Laure Conan|Élisabeth Seton|
|1904-1912|Romain Rolland|Jean-Christophe (1)|
|1904-1912|Romain Rolland|Jean-Christophe (2)|
|1917|Adèle Bourgeois|Némoville|
|1923|Raymond Radiguet|Le diable au corps|
|1926|Marguerite Audoux|De la ville au moulin|
|1937|Marguerite Audoux|Douce Lumière|

## Project structure

Note that annotation guidelines can be found in <i>[Manuel_Annotation.pdf](./doc/Manuel_Annotation.pdf)</i> in the <i>doc</i> folder.


```                                            
   .
   ├── brat
   │   ├── citations
   │   │   ├── Bouvard_et_Pecuchet.ann
   │   │   ├── Bouvard_et_Pecuchet.txt
   │   │   ├── Le_capitaine_Fracasse.ann
   │   │   ├── Le_capitaine_Fracasse.txt
   │   │   ├── ...
   │   ├── coref
   │   │   ├── Bouvard_et_Pecuchet.ann
   │   │   ├── Bouvard_et_Pecuchet.txt
   │   │   ├── Le_capitaine_Fracasse.ann
   │   │   ├── Le_capitaine_Fracasse.txt
   │   │   ├── ...
   │   ├── entities
   │   │   ├── Bouvard_et_Pecuchet.ann
   │   │   ├── Bouvard_et_Pecuchet.txt
   │   │   ├── Le_capitaine_Fracasse.ann
   │   │   ├── Le_capitaine_Fracasse.txt
   │   │   ├── ...
   │   ├── events
   │   │   ├── Bouvard_et_Pecuchet.ann
   │   │   ├── Bouvard_et_Pecuchet.txt
   │   │   ├── Le_capitaine_Fracasse.ann
   │   │   ├── Le_capitaine_Fracasse.txt
   │   │   ├── ...
   │   └── events_tsv
   │       ├── Bouvard_et_Pecuchet.tsv
   │       ├── Le_capitaine_Fracasse.tsv
   │       ├── ...
   ├── doc
   │   ├── Manuel_Annotation.pdf
   │   ├── ...
   ├── MODEL_NLP-schema
   │   ├── BOUVARDETPECUCHET_NLP.aam
   │   ├── ...
   ├── sacr
   │   ├── Bouvard_et_Pecuchet.sacr
   │   ├── Le_capitaine_Fracasse.sacr
   │   ├── ...
   ├── src
   │   ├── nameFile.py
   ├── urs
   │   ├── BOUVARDETPECUCHET.urs
   │   ├── CAPITAINEFRACASSE.urs
   │   ├── ...
   ├── urs-xml
   │   ├── bouvardetpecuchet-urs.xml
   │   ├── capitainefracasse-urs.xml
   │   ├── ...
   └── xml
       ├── bouvardetpecuchet.xml
       ├── capitainefracasse.xml
       ├── ...
```

<a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>
<br/>fr-litBank is licensed under a <a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0/">Attribution-ShareAlike 2.0 France (CC BY-SA 2.0 FR)</a>.
