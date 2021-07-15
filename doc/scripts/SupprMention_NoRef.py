from bs4 import BeautifulSoup
import re
import os
import sys

# script qui prend en entrée un fichier *.xml (file de type *-urs.xml)
# 1/ il supprime toutes les mentions annotées qui ne sont pas dans une chaine
# 2/ il reporte sur la mention dans le champ "UNIT REF" la valeur du "TYPE DE REF"
# en sortie un fichier *-REF.xml" est créé

file = sys.argv[1]
title = re.sub(".xml", "", file)

indata=open(file,"r", encoding="utf-8", errors="ignore") # UTF-8 encoding errors are ignored
contents = indata.read()
soup = BeautifulSoup(contents,'xml')

# Supprimer les chaînes non retenues comme étant 'EN'
myDicSch = dict()
sch = soup.find("div", {"type": "schema-fs"})
for sch_fs in sch.find_all('fs'):
	myId = str(sch_fs['id'])
	myId = re.sub("-fs", "", myId)
	# si pas de type -> "none"
	typeRef = sch_fs.find("f", attrs={"name":"TYPE REFERENT"})
	try :
		myVal = typeRef.get_text()
	except:
		myVal = "None"
		sch_fs.extract()
	# mon id était : s-CHAINE-1781-fs / il devient : s-CHAINE-1781
	myDicSch[myId]=myVal

print("nombre de chaines supprimées (sans annotation TYPE REFERENCE) :"+str(len(myDicSch)))

# annotationgrp -> recupérer les mentions/unit qui composent un schema
myDicMention = dict()
annot = soup.find("annotationGrp", {"type": "Schema"})

for annot_link in annot.find_all('link'):
	if annot_link['id'] in myDicSch :
		# les mentions sont dans l'argument "target"
		# ... elles ont ce format : #u-MENTION-11
		lstMention = annot_link['target'].split(" #")
		for mention in lstMention:
			# ... format utile pour la suite : u-MENTION-11-fs
			mention = re.sub("#", "", mention)
			mention = re.sub("$", "-fs", mention)
			myDicMention[mention]=myDicSch[annot_link['id']]
#print(myDicMention)

# Récupérer les id des mentions annotées non retenues dans une chaîne :
# <span id="u-MENTION-119"...
myLstSupprMention = list()
# DIV unit-fs -> parcourir les mentions/unités
unitAnnot = soup.find("div", {"type": "unit-fs"})
for unit_fs in unitAnnot.find_all('fs'):
	unitId = str(unit_fs['id'])

	# soit value soit None
	if unitId in myDicMention:
		if myDicMention[unitId] == "None":
			unit_fs.extract()
			unitId = re.sub("-fs", "", unitId)
			myLstSupprMention.append(unitId)
		else:
			#create a new tag
			new_tag = soup.new_tag("f")
			new_tag.attrs['name'] = 'UNIT REF'

			#encadrer la valeur de la balise string
			tagStr = soup.new_tag("string")
			tagStr.append(myDicMention[unitId])
			new_tag.append(tagStr)

			for unit_f in unit_fs.findChildren('f'):
				unit_f.insert_after(new_tag)
print("nombre de mentions supprimées (car non utilisées dans une chaine) :"+str(len(myLstSupprMention)))	

# Supprimer les mentions inutiles/non annotées
#<annotationGrp type="Unit" subtype="MENTION">
annotMention = soup.find("annotationGrp", {"subtype": "MENTION"})
for annot_span in annotMention.find_all('span'):
	if annot_span['id'] in myLstSupprMention :
		annot_span.extract()

f = open(title+'-REF.xml','wb')
f.write(soup.encode('utf-8'))