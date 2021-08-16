# Sentence (object)
Sentence (élement s dans le fichier XML TEI)

## Properties
 - `id` (int)
 - `content` (List) # on doit conserver l'ordre des mots (éléments `w` et des `lb`), on peut stocker ça dans une liste contenant les id des `w` et la chaîne "lb" 


# Word (object)
Word (élément w dans le fichier XML TEI)

## Properties
 - `id` (str)
 - `start` (int) # début offset de car.
 - `end` (int) # début offset de car.
 - `pos` (str)
 - `lemma` (str)


# Mention (object)
Mention (dans le fichier URS)

## Properties
 - `id` (str)
 - `words` (List) # la liste des id des mots qui composent une mention (dans l'ordre évidemment)
 - `ref` (str) # dans la `div type="unit-fs"`, l'élément `f name="REF"`

# Chaine (object)
Chaine (dans le fichier URS)

## Properties
 - `id` (str)
 - `mentions` (List) # la list des id des mentions faisant partie de la chaîne (dans l'ordre of course)
 - `ref` (str) # dans la `div type="schema-fs"`
 - `nb_maillons` (int) # dans la `div type="schema-fs"`
 - `type_referent` (str) # dans la `div type="schema-fs"`