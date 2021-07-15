import os
import re
# prend en entrée les fichiers model .aam
# et ajoute les étiquettes d'annotations pour le schema

# 2 dossiers : dossier avec MODEL de départ et dossier où seront modifiés les fichiers
dossier_source = "MODEL/"
dossier_cible = "MODEL_NLP-schema/"

ajout = '\
<feature name="TYPE REFERENT">\n\
<possibleValues default = "">\n\
<value>None</value>\n\
<value>OTHER</value>\n\
<value>NO_PER</value>\n\
<value>LOC</value>\n\
<value>ORG</value>\n\
<value>VEH</value>\n\
<value>METALEPSE</value>\n\
<value>FAC</value>\n\
<value>TIME</value>\n\
<value>PER</value>\n\
<value>GPE</value>\n\
<value>TO_DISCUSS</value>\n\
<value>EVENT</value>\n\
</possibleValues>\n\
</feature>\n'

listFile = list()
for path, dirs, files in os.walk(dossier_source):
	for filename in files:
		listFile.append(filename)

for file in listFile:
	# ouvrir le fichier source
	fileOut = re.sub("\.aam", "_NLP.aam", file)
	correc = open(dossier_cible+fileOut, "w")

	# ouvrir un fichier cible
	pathFile = open(dossier_source+file)
	lignes = pathFile.readlines()
	for ligne in lignes:
		#if '<feature name="REF">' in ligne:
		if '<feature name="NB MAILLONS">' in ligne:
			ligne=ajout+ligne
		correc.writelines(ligne)
	correc.close()
