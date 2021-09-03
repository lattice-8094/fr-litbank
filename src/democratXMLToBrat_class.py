# -*- coding: utf-8 -*-

"""
classes for fr-litbank
"""


class Sentence:
    """ """

    # from https://forge.cbp.ens-lyon.fr/redmine/projects/txm/repository/entry/tmp/org.txm.utils/src/org/txm/utils/i18n/LangFormater.java
    __fr_nospace_before = [",", ".", ")", "]", "}", "°", "-", "-"]
    __fr_nospace_after = ["’", "'", "(", "[", "{"]

    def __init__(self, id, start=0):
        self.id = id
        self.start = start
        self.content = []

    def __str__(self):
        res = ""
        for item in self.content:
            if isinstance(item, Word):
                res += item.printable
            else:
                res += item
        return res

    def set_content(self, content):
        """
        Sets sentence content, i.e. a list of Word objects and '\n'
        This method sets the 'start', 'end' and 'printable' attributes of each word
        """
        current_offset = self.start  #  la position du caractère courant
        for i, item in enumerate(content):
            if isinstance(item, Word):
                item.start = current_offset
                try:
                    # si l'item suivant est un objet Word et qu'il commence par un caractère 'no space before' : pas d'espace
                    if (
                        isinstance(content[i + 1], Word)
                        and content[i + 1].form[0] in Sentence.__fr_nospace_before
                    ):
                        item.printable = item.form
                    # si le dernier caractère de l'item courant est 'no space after' : pas d'espace
                    elif item.form[-1] in Sentence.__fr_nospace_after:
                        item.printable = item.form
                    # si l'item suivant n'est pas un objet Word : pas d'espace
                    elif not(isinstance(content[i + 1], Word)):
                        item.printable = item.form
                    # sinon un espace après
                    else:
                        item.printable = item.form + " "
                # exception quand pas de mot suivant : espace
                except IndexError:
                    if item.pos == 'NOM' or item.pos == 'NAM':
                        item.printable = item.form
                    else:
                        item.printable = item.form + " "
                self.content.append(item)
                current_offset = item.get_end()
            else:
                self.content.append(item)
                current_offset += 1

    def get_end(self):
        """
        Computes and returns the last char offset
        """
        if self.content:
            self.end = 0
            for item in self.content[
                ::-1
            ]:  # on parcourt le contenu en partant de la fin
                if isinstance(
                    item, Word
                ):  # si l'item courant est un mot : on renvoie self.end plus l'offset de fin du mot courant
                    self.end += item.get_end()
                    return self.end
                else:  # sinon c'est un 'lb' on ajoute 1 à self.end
                    self.end += 1


class Word:
    """ """

    def __init__(self, id, form, pos, lemma):
        self.id = id
        self.form = form
        self.pos = pos
        self.lemma = lemma
        self.printable = ""

    def __str__(self):
        return self.printable

    def get_end(self):
        return self.start + len(self.printable)


class Mention:
    """ """

    def __init__(self, id, ref, words):
        self.id = id
        self.ref = ref
        self.words = words

    def __str__(self):
        return "".join([str(word) for word in self.words])

    def is_entity(self):
        """
        Tells wether a mention is an entity (in litbank way) or not
        If one of the word in mention is a noun (proper or common), the 
        mention is an entity
        """
        for word in self.words:
            if word.pos == "NAM" or word.pos == "NOM":
                return True
        return False

class Event:
    """
    An event annotation in the litbank way (see https://github.com/dbamman/litbank#event-annotations)
    i.e. like a mention in the urs format but not related to a 'chaine'
    the event annotation is not herited from the Democrat corpus, it has been made for the fr-litbank
    """

    def __init__(self, id, words):
        self.id = id
        self.words = words
    
    def __str__(self):
        return "".join([str(word) for word in self.words])

class Chaine:
    """ """

    def __init__(self, id, ref, nb_maillons, type_referent):
        self.id = id
        self.ref = ref
        self.nb_maillons = nb_maillons
        self.type_referent = type_referent
