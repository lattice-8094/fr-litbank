### Présentation
Ce code permet de produire et utiliser des modèles d'étiquetage de texte (reconnaissance d'entités nommées, identification de discours de direct) et de liaison d'entités (coréférence, détection de source pour le discours direct). Plus généralement, ces modèles pourraient apprendre à étiqueter et à typer n'importe quelle mention qui puisse être d'intérêt littéraire, syntaxique ou grammatical. Ici, on appelle ces mentions "entités", mais il pourrait s'agir de n'importe quel ensemble de sous-parties du texte. De plus, ces modèles pourraient apprendre à lier ces mentions entre elles. Ici, on appelle ces liens "coréférence", mais il pourrait s'agir de n'importe quel lien entre deux "entités", qu'elles aient le même type ou non.

### Installation :
```pip install -r requirements.txt```

### Entraînement :
```python run_lm.py --data_dir <dossier contenant les fichiers brat> --output_dir <dossier de sortie>```

Autres options :
- ```--coref_pred```: Prédire également la coréférence
- ```--max_seq_length```: Indique la longueur des fenêtres (256 mots par défaut), des fenêtres longues améliorent la performance en coréférence mais sont plus lourdes pour les calculs
- ```--chunk-int```: Indique la longueur de chevauchement entres les fenêtres (16 mots par défaut), des fenêtres trop chevauchées améliorent la performance en coréférence mais sont plus lourdes pour les calculs
- ```--bio```: Utiliser un schéma BIO plutôt que BIOES, meilleur pour les entités qui ne peuvent pas être emboîtées, comme les évènements par exemple.
- ```--per_device_train_batch_size```: Taille de batch en entraînement (8 par défaut)
- ```--per_device_eval_batch_size```: Taille de batch en évaluation (8 par défaut)
- ```--learning_rate```: Taux d'apprentissage (4e-5 par défaut)
- ```--num_train_epochs```: Nombre d'epochs d'entraînement
- ```--debug```: Pour débugger rapidement, prendre seulement un petite partie des données d'entraînement et d'évaluation
- ```--use_cache```: Si l'entraînement a déjà été lancé avant pour les mêmes données, utiliser la cache pour accélerer le chargement des données
- ```--model_name_or_path```: Identifiant HuggingFace du modèle à charger (camembert-base par défaut) par exemple "camembert/camembert-large"
- ```--ignore_labels```: Labels à ignorer, séparés par une virgule
- ```--replace_labels```: Labels à renommer, séparés par une virgule, par exemple AA:BB,CC:BB,DD:EE remplace AA par BB, CC par BB et DD par EE.


### Evaluation d'un modèle existant :
```python run_lm.py --data_dir <dossier contenant les fichiers brat de test> --output_dir <dossier de sortie> --test --model_name_or_path <dossier contenant le modele>```

### Prédiction à l'aide d'un modèle existant :
```python run_lm.py --data_dir <dossier contenant les fichiers txt> --output_dir <dossier de sortie> --inference --model_name_or_path <dossier contenant le modele>```
