#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AnnToSacr.py: from Brat format to Sacr. The input files are in the folder 'brat/coref', the output files (the files and the schema) are in the folder 'sacr'. The output files can be opened with  Sacr (https://github.com/boberle/sacr). """

__author__ = "Frédérique Mélanie-Becquet & Martial Pastor"
__copyright__ = "Copyright 2022, BookNLP Project"
__license__ = "CC BY-SA 4.0"
__email__ = "frederique.melanie@ens.psl.eu"

import os
import re

def listerFile (myRep):
	listFile = list()
	for path, dirs, files in os.walk(folder_path):
		for filename in files:
			if ".txt" in filename:
				filename = re.sub(".txt","",filename)
				listFile.append(filename)
	listFile = sorted(listFile)
	#print(listFile)
	return listFile

def annotToDic (myFile):

	myFileAnnot = myFile+".ann"

	pathAnnot = open(folder_path+myFileAnnot)
	myAnnot = pathAnnot.readlines()

	myDicAnnot = dict()
	myDicCoref = dict()

	for annot in myAnnot:

		myLine = annot.strip().split('\t')

		myRef = myLine[1].split(' ')

		# les entités
		#if annot.startswith("T"):
		if len(myLine) == 3:

			print(myRef[0])

			if myRef[0]	in myListFilter :

				myAnnotValue = myRef[0]
				borneInit = int(myRef[1])
				borneFinale = int(myRef[2])
				
				#mylist = Tnum + EN + borneInit + borneFinale + texte
				myList = (myLine[0],myAnnotValue,borneInit,borneFinale,myLine[2])
				final_key = str(borneInit) +'-'+ str(borneFinale)
				myDicAnnot[int(myRef[1])]=myList
			
		# les relations
		# R61	Coreference Arg1:T552 Arg2:T534
		# elif annot.startswith("R"):
		else:
			myRef1 = re.sub('Arg1:','',myRef[1])
			myRef2 = re.sub('Arg2:','',myRef[2])
			if myRef1 in myDicCoref or myRef2 in myDicCoref :
				try :
					myDicCoref[myRef2].append(myRef1)					
				# A PRIORI INUTILE :
				except :
					myDicCoref[myRef1].append(myRef2)
					
			else:
				myListRef = list()
				myListRef.append(myRef1)
				myDicCoref[myRef2]=myListRef
	
	return myDicCoref,myDicAnnot

def myTexteBrut(myFile):
	myFileTxt = myFile+".txt"
	pathTxt = open(folder_path+myFileTxt)
	myTxt = pathTxt.read()
	return myTxt

def mySchema(myEntities,myPathSchema):
	# créer le template / schema d'annotation
	mySchema = open(myPathSchema+"schema.sacr", "w")
	mySchema.write("PROP:name=EN\n$$$\n")
	for keyEntity,myEntity in myEntities.items():
		mySchema.write(myEntity+"\n")
	mySchema.close()

# inverser clef et valeur
# toutes les ref sont en clef et ont pour valeur une ref unique
def inverseKeyValue(myDicCoref):	
	myDicCorefInvers=dict()
	for myCoref, myLstRef in myDicCoref.items():
		for eachRef in myLstRef:
			myDicCorefInvers[eachRef]=myCoref
	return myDicCorefInvers

def ordonnKey(myDicAnnot):
	# Ordonner les clefs du dico
	sorted_keys = sorted(myDicAnnot)
	# créer un dico ordonné
	sorted_annot = {}
	for key in sorted_keys:
	  sorted_annot[key] = list(myDicAnnot[key])

	for keyOrd in sorted_annot:
		# print(sorted_annot[keyOrd][0])
		if sorted_annot[keyOrd][0] in myDicCorefInvers:
			sorted_annot[keyOrd].append(myDicCorefInvers[sorted_annot[keyOrd][0]])
		else :
			sorted_annot[keyOrd].append(sorted_annot[keyOrd][0])

	return sorted_annot

def createFileSacr(sorted_annot,myTxt):
	
	nested_annot = {}
	bornPreced = 0

	#print(sorted_annot)

	i=0
	for en in sorted_annot:
		for en2 in sorted_annot:
			if en == en2 : continue
			if sorted_annot[en][2] < sorted_annot[en2][2] < sorted_annot[en][3]:
				nested_annot.setdefault(en2,{'set':i})
				nested_annot.setdefault(en,{'set':i})
				nested_annot[en2] = nested_annot[en]
			i+=1
	nested_annot_sets = {}
	for en in nested_annot:
		nested_annot_sets.setdefault(nested_annot[en]['set'],{'ens':{}})
		beg = int(sorted_annot[en][2])
		end = int(sorted_annot[en][3])
		nested_annot_sets[nested_annot[en]['set']]['ens'][end-beg] = en
	
	#print(nested_annot_sets)

	written_nested = {}
	for x,y in sorted_annot.items():
	 	#['T1', 'PER', 0, 19, 'Bouvard et Pécuchet', 'T1']
		# PER FAC LOC
		myAnnot = sorted_annot[x][1]

		if myAnnot in myEntities:
			myAnnot=myEntities[myAnnot]
		# une entité d'annotation n'est pas présente dans le schema
		else:
			print("!!! A AJOUTER dans le schema : " + str(myAnnot))

		borneInit = int(sorted_annot[x][2])
		borneFinale = int(sorted_annot[x][3])
		mySeq = sorted_annot[x][4]
		myCoref = sorted_annot[x][5]

		# texte sans annotation
		if borneInit != bornPreced :
			#print(myTxt[bornPreced:borneInit-1])
			myFileOut.write(myTxt[bornPreced:borneInit-1])
		
		# texte annoté
		# exemple : format du template d'annotation 
		# {M1:EN="p PER" Bouvard et Pécuchet}

		if x in written_nested:continue
		mySacr = mySeq
		if x in nested_annot and x not in written_nested:
			ens_dic = nested_annot_sets[nested_annot[x]['set']]['ens']
			for en in sorted(ens_dic, key=ens_dic.get):
				written_nested.setdefault(ens_dic[en],'')
				annotatedSeq = '{'+sorted_annot[ens_dic[en]][5]+':EN="'+myEntities[sorted_annot[ens_dic[en]][1]]+'" '+ sorted_annot[ens_dic[en]][4] +'}'
				mySacr = mySacr.replace(sorted_annot[ens_dic[en]][4], annotatedSeq)
			#print(mySacr)
		else:
			mySacr = '{'+myCoref+':EN="'+myAnnot+'" '+ mySeq +'}'

		#mySacr = '{'+myCoref+':EN="'+myAnnot+'" '+ mySeq +'}'
		# print(mySacr)
		myFileOut.write(mySacr)

		bornPreced=borneFinale

	myFileOut.close()

def applyCoref(sorted_annot, myDicCorefInvers):

	new_sorted_annot = {}
	for key in sorted_annot:
		mdci_key = str(sorted_annot[key][0]) + ','
		if mdci_key  in myDicCorefInvers:
			sorted_annot[key][5] =  myDicCorefInvers[mdci_key]
			#print(sorted_annot[key])
			new_sorted_annot[key] = sorted_annot[key]
		else:
			new_sorted_annot[key] = sorted_annot[key]

	return new_sorted_annot


# exécution des fonctions
if __name__ == '__main__':

	# Récupérer le nom des fichiers annotés: brut (txt) et annotés (ann)
	# en créer une liste ordonnée
	# folder_path = "brat/coref/"
	# folder_path = "Manon_Lescaut/"
	folder_path = "coref/"
	listFile = listerFile(folder_path)

	# les entitées présentes dans les fichiers COREF/*.ann
	myEntities={"None":"x None","X":"xx X","OTHER":"xxx OTHER","TO_DISCUSS":"xxxx TO_DISCUSS","METALEPSE":"m METALEPSE","NO_PER":"n NO_PER","PER":"p PER","LOC":"l LOC","FAC":"f FAC","TIME":"t TIME","ORG":"o ORG","VEH":"v VEH","GPE":"g GPE","HIST":"h HIST"}

	myListFilter = myEntities

	############## FILTRER ###########
	# Decommenter les 2 lignes ci-dessous pour n'avoir que les PERS 
	# myListFilter = {"METALEPSE":"m METALEPSE","NO_PER":"n NO_PER","PER":"p PER"}
	#myEntities = myListFilter
	
	# créer le schema (commun à tous les fichiers)
	# myPathOut = "sacr/Pers_entites"
	myPathOut = "sacr/All_Entites/"
	mySchema(myEntities,myPathOut)

	for myFile in listFile:

		myFileOut = open(myPathOut+myFile+".sacr", "w")

		# récuperer les annot
		#print("---------------")
		myDicCoref = annotToDic(myFile)[0]
		#print(annotToDic(myFile)[0])
		myDicAnnot = annotToDic(myFile)[1]
		#print(annotToDic(myFile)[1])

		# récuperer le texte brut
		myTxt = myTexteBrut(myFile)

		myDicCorefInvers = inverseKeyValue(myDicCoref)
	
		sorted_annot = ordonnKey(myDicAnnot)
		sorted_annot = applyCoref(sorted_annot, myDicCorefInvers)

		createFileSacr(sorted_annot,myTxt)
		
