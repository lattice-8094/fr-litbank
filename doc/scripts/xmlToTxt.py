import os
import re
from bs4 import BeautifulSoup

# 2 dossiers : dossier avec XML de départ et dossier où seront modifiés les fichiers
dossier_source = "XML/"
dossier_cible = "TXT/"

listFile = list()
for path, dirs, files in os.walk(dossier_source):
	for filename in files:
		listFile.append(filename)

# liste trouvée grâce à : set([t.parent.name for t in text])
blacklist=['publicationstmt', 'projectdesc', 'orgname', 'profiledesc', 'titlestmt', 'p', 'body', 'text', 's', 'date', 'editorialdecl', 'txm:ana', 'editionstmt', 'teiheader', 'respstmt', 'note', 'availability', 'author', 'funder', 'samplingdecl', 'name', 'term', 'idno', 'sourcedesc', 'ref', 'licence', 'tei', 'w', 'textclass', 'encodingdesc', 'org', 'extent', 'edition', '[document]', 'keywords', 'filedesc', 'principal', 'resp', 'creation', 'distributor', 'bibl', 'title', 'publisher']

## parcourir chacun des fichiers
for file in listFile:

	output = ''

	# ouvrir le fichier source
	fileOut = re.sub("\.xml", ".txt", file)
	correc = open(dossier_cible+fileOut, "w")

	# ouvrir un fichier cible
	pathFile = open(dossier_source+file)
	content = pathFile.read()

	soup = BeautifulSoup(content,features='lxml')

	for b in soup.find_all('lb'):
		b.string = "\n"

	text = soup.find_all(text=True)

	#print(set([t.parent.name for t in text]))
	for t in text:
		if t.parent.name not in blacklist:
			output += '{} '.format(t)
	
	correc.writelines(output)
	correc.close()
