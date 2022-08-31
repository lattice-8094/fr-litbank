Un readme plus propre arrive prochainement.

### Installation :
```pip install -r requirements.txt```

### Entraînement :
```python run_lm.py --data_dir <dossier contenant les fichiers brat> --output_dir <dossier de sortie>```

### Evaluation d'un modèle existant :
```python run_lm.py --data_dir <dossier contenant les fichiers brat de test> --output_dir <dossier de sortie> --test --model_name_or_path <dossier contenant le modele>```

### Prédiction à l'aide d'un modèle existant :
```python run_lm.py --data_dir <dossier contenant les fichiers txt> --output_dir <dossier de sortie> --inference --model_name_or_path <dossier contenant le modele>```
