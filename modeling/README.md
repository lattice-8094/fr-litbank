## Présentation
Ce code permet de produire et utiliser des modèles d'étiquetage de texte (reconnaissance d'entités nommées, identification de discours de direct) et de liaison d'entités (coréférence, détection de source pour le discours direct). Plus généralement, ces modèles pourraient apprendre à étiqueter et à typer n'importe quelle mention qui puisse être d'intérêt littéraire, syntaxique ou grammatical. Ici, on appelle ces mentions "entités", mais il pourrait s'agir de n'importe quel ensemble de sous-parties du texte. De plus, ces modèles pourraient apprendre à lier ces mentions entre elles. Ici, on appelle ces liens "coréférence", mais il pourrait s'agir de n'importe quel lien entre deux "entités", qu'elles aient le même type ou non.

Ce code fonctionne sous 3 modes :\
1- **Entraînement** : Avoir à disposition des données annotées (sous format brat), et les utiliser pour *fine-tuner* un modèle CamemBERT capable de "comprendre" ces annotations pour pouvoir ensuite les reproduire. Ce mode est accessible par défaut.\
2- **Evaluation** : Avoir à disposition un modèle déjà entraîné ET des données annotées (sous format brat), et les utiliser pour évaluer ce modèle en comparant les prédiction du modèle aux annotations (*gold*). Ce mode est accessible via l'option ```--test```.\
3- **Inférence** : Avoir à disposition un modèle déjà entraîné et des fichiers ```.txt```, et utiliser le modèle pour prédire les entités et les coréférences qu'il a été entraîné à prédire. Ce mode est accessible via l'option ```--inference```.

## Installation :
```pip install -r requirements.txt```

## Entraînement :
Utilisation de base :\
```python run_lm.py --data_dir <dossier contenant les fichiers brat> --output_dir <dossier de sortie>```

Autres options :
- ```--coref_pred```: Prédire également la coréférence
- ```--max_seq_length```: Indique la longueur des fenêtres (256 mots par défaut), des fenêtres longues améliorent la performance en coréférence mais sont plus lourdes pour les calculs
- ```--chunk-int```: Indique la longueur de chevauchement entres les fenêtres (16 mots par défaut), des fenêtres trop chevauchées améliorent la performance en coréférence mais sont plus lourdes pour les calculs
- ```--bio```: Utiliser un schéma BIO plutôt que BIOES, meilleur pour les entités qui ne peuvent pas être emboîtées, comme les évènements par exemple.
- ```--per_device_train_batch_size```: Taille de batch en entraînement (8 par défaut)
- ```--per_device_eval_batch_size```: Taille de batch en évaluation (8 par défaut)
- ```--learning_rate```: Taux d'apprentissage (4e-5 par défaut)
- ```--num_train_epochs```: Nombre d'epochs d'entraînement (10 par défaut)
- ```--debug```: Pour débugger rapidement, prendre seulement un petite partie des données d'entraînement et d'évaluation
- ```--use_cache```: Si l'entraînement a déjà été lancé avant pour les mêmes données, utiliser la cache pour accélerer le chargement des données
- ```--model_name_or_path```: Identifiant HuggingFace du modèle à charger (camembert-base par défaut) par exemple "camembert/camembert-large"
- ```--ignore_labels```: Labels à ignorer, séparés par une virgule
- ```--replace_labels```: Labels à renommer, séparés par une virgule, par exemple AA:BB,CC:BB,DD:EE remplace AA par BB, CC par BB et DD par EE.

Exemples :\
```python modeling/run_lm.py --data_dir brat/entities --output_dir modele_test --ignore_labels None,TO_DISCUSS,OTHER,X --replace_labels NO_PER:PER,HIST:TIME,METALEPSE:PER```\
```python modeling/run_lm.py  --data_dir brat/coref --coref_pred --ignore_labels None,TO_DISCUSS,OTHER,X --replace_labels NO_PER:PER,HIST:TIME,METALEPSE:PER --output_dir modele_test```

Evidemment, il faut bien noter les paramètres d'entraînement qu'on a précisés (tels que ```ignore_labels```, ```replace_labels```, ```coref_pred```), et les remettre également en utilisant le modèle pour l'évaluation ou l'inférence.

## Evaluation d'un modèle existant :
Utilisation de base :\
```python run_lm.py --data_dir <dossier contenant les fichiers brat de test> --output_dir <dossier de sortie> --test --model_name_or_path <dossier contenant le modele>```\
Exemples :\
```python modeling/run_lm.py --data_dir brat/entities --output_dir output_test --test --model_name_or_path modele_test --ignore_labels None,TO_DISCUSS,OTHER,X --replace_labels NO_PER:PER,HIST:TIME,METALEPSE:PER```\
```python modeling/run_lm.py  --data_dir brat/coref --coref_pred --ignore_labels None,TO_DISCUSS,OTHER,X --replace_labels NO_PER:PER,HIST:TIME,METALEPSE:PER --output_dir output_test --test --model_name_or_path modele_test ```


## Prédiction à l'aide d'un modèle existant :
Utilisation de base :\
```python run_lm.py --data_dir <dossier contenant les fichiers txt> --output_dir <dossier de sortie> --inference --model_name_or_path <dossier contenant le modele>```\
Exemples :\
```python modeling/run_lm.py --data_dir brat/entities --output_dir output_test --inference --model_name_or_path modele_test --ignore_labels None,TO_DISCUSS,OTHER,X --replace_labels NO_PER:PER,HIST:TIME,METALEPSE:PER```\
```python modeling/run_lm.py  --data_dir brat/coref --coref_pred --ignore_labels None,TO_DISCUSS,OTHER,X --replace_labels NO_PER:PER,HIST:TIME,METALEPSE:PER --output_dir output_test --inference --model_name_or_path modele_test ```

## Architecture
Nous utilisons un modèle CamemBERT pré-entraîné et procédons à un *fine-tuning* de ses paramètres pour reconnaître les mentions et résoudre la coréférence au sein d'un *chunk* (fenêtre glissante) de $n=256$ tokens ; CamemBERT, comme la plupart des modèles basés sur une architecture *Transformer*, a une complexité quadratique en fonction de la longueur de séquence d'entrée et est donc plus adapté à des entrées de taille fixe relativement petite.

Pour pouvoir détecter des mentions imbriquées, nous utilisons un schéma d'étiquetage *BIOES* (**B**eginning, **I**nside, **O**utside, **E**nding, **S**ingle-word). L'étiquette attribuée à chaque mot dépend donc du type de la mention, mais également de la position du mot dans celle-ci. L'ensemble des étiquettes possibles est M =  \{O, B-PER, I-PER, S-PER, E-PER, B-LOC, ...\}.

Concrètement, pour chaque token $w_i$, le modèle est entraîné à prédire deux étiquettes :
- $m_i \in M$ correspondant à la mention la plus courte contenant le mot, et à sa position dans celle-ci
- $r_i \in \{0, 1, ... n\}$ indiquant l'indice du premier mot qui coréfère avec $w_i$ dans la fenêtre glissante, si un tel mot existe, et $0$ sinon.

Pour cela, nous entraînons le modèle à attribuer à $w_i$ trois représentations $a(w_i)$, $q(w_i)$ et $k(w_i)$ telles que :
- $a(w_i)$ représente le mot en tant que mention, on modélise ainsi la distribution de probabilité sur l'ensemble des étiquettes possibles comme $$P\big(m_i=M_s | a(w_i)\big) = \sigma\left[f(a(w_i))\right]_s$$ où $f$ est une projection linéaire apprise dans l'ensemble des étiquettes et $\sigma$ la fonction \textit{softmax}.
- $q(w_i)$ représente le mot en tant que référence et $k(w_i)$ en tant que référent, de façon à ce que plus $w_i$ est susceptible de référer à $w_j$, plus $q(w_i)$ est proche de $k(w_j)$. On modélise ainsi la distribution de probabilité sur l'ensemble des étiquettes comme
    $$P\big(r_i=t | \left(q(w_i),k(w_t)\right)\big) = \sigma\left[q(w_i) \cdot k(w_t)\right]_t$$ où $\cdot$ désigne le produit scalaire.

Pour pouvoir utiliser le modèle dans notre cas avec des textes longs, ceux-ci sont découpés en $chunks$ chevauchés chacun de taille $n=256$ tokens. Un nouveau $chunk$ commence tous les $l=16$ tokens. La plupart des mots du texte sont donc présents dans $\frac{n}{l} = 16$ \textit{chunks}.

Une fois les prédictions du modèle calculées, on sélectionne parmi les $16$ prédictions attribuées à chaque mot en fonction du nombre d'occurrences et la position du mot dans chaque \textit{chunk}, puis un parcours par profondeur est utilisé pour récupérer les chaînes de coréférence au niveau global. Nous obtenons ainsi, comme montré dans l'exemple figure \ref{fig:exemple}, pour une entrée de taille quelconque, une suite d'étiquettes identifiant les mentions, et une suite d'étiquettes identifiant les entités distinctes mentionnées.

## Performances pour la coréférence
Deux textes sont séparés comme jeu de validation (Douce Lumière et De la ville au moulin). Nous entraînons un modèle CamemBERT-Large pendant $10$ *epochs* avec un *batch size* de 8 *chunks*, en démarrant l'apprentissage avec un *learning rate* de $5\times 10^{-4}$. Le tableau suivant détaille les performances du modèle sur le jeu de validation pour la détection de mentions et la résolution de coréférence selon différents scores.


|          | précision | rappel | $F_1$ |
|----------|-----------|--------|-------|
| Mentions |    90,65  |  90,08 |  90,37|
| $MUC$    |    85,06  |  85,10 |  85,08|
| $B^3$    |    82,66  |  56,49 |  67,11|
| $CEAFe$  |    28,50  |  91,89 |  43,50|
| $BLANC$  |    85,81  |  62,99 |  69,22|
| $LEA$    |    64,73  |  62,47 |  63,58|

Cet article https://hal.archives-ouvertes.fr/hal-03701468/ explique davantage ce modèle et en montre une application directe dans la recherche en littérature.
