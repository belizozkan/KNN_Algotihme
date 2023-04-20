################################################################# !----SCRIPT----! #########################################################################

from IPython.testing import test
from IPython.utils import text
import json
import math
import os
import re
import copy

# on définit une expression régulière pour tokeniser des textes
tok_grm = re.compile(r"""
    (?:etc\.|p\.ex\.|cf\.|M\.)|
    \w+(?=(?:-(?:je|tu|ils?|elles?|nous|vous|leur|lui|les?|ce|t-|même|ci|là)))|
    [\w\-]+'?|
    .
""", re.X)

class TextVect :
    def __init__(self, vector={}) :
        
        """
        Initialise la classe TextVect.

        argument 1 (optionnel) - > vector (dict):  Un dictionnaire vide.
        valeur de retour - > (none) : Cette mèthode ne retourne rien.
        """

        # on stocke le dictionnaire vector 
        self.vector = vector
        # on initialise un ensemble vide pour la stoplist
        self.stoplist = set()

    def read_dict(self, stoplist_filename) :
        
        """
        Lit un fichier (stoplist) qui contient une liste de mots à exclure, puis les stocke dans l'attribut self.stoplist.

        argument 1 - > stoplist_filename (str): Le nom du fichier contenant la liste de mots à exclure.
        valeur de retour - > (set) : L'ensemble de mots stocké dans self.stoplist.
        """

        # on ouvre le fichier pour le lire
        try :   
          dict_file = open(stoplist_filename, "r", encoding="utf-8")
        except Exception as err :
        # ce bloc s'exécute en cas d'erreur
          print("Impossible d'ouvrir ", stoplist_filename,f" Erreur={err}")
          # on ne retourne rien et un message d'erreur s'affiche
          return None
        # on lit le contenu du fichier 
        dict_content = dict_file.read()
        # et après on ferme le fichier
        dict_file.close()
        # on sépare le dict_content(string) avec la saut de ligne et renvoie un ensemble
        self.stoplist = set(dict_content.split("\n"))
        return self.stoplist

    
    def filter(self, stoplist:set, documents, non_hapax) :
        
        """
        Filtre les documents fournis en enlevant les mots appartenant à la stoplist et les mots (les hapax) qui n'apparaissent qu'une seule fois.

        argument 1 - > stoplist (set) : L'ensemble de mots à exclure.
        argument 2 - > documents (list) : Une liste de dictionnaires contenant les documents à filtrer. 
        argument 3 - > non_hapax (bool) : Si True, élimine également les mots (les hapax) qui n'apparaissent qu'une seule fois.
        valeur de retour - > (list) : Une nouvelle liste de dictionnaires contenant les documents filtrés. Chaque dictionnaire représente un document et contient deux clés : "label" (une chaîne de caractères qui donne un nom au document) et "vect" (un dictionnaire qui contient les tokens du document et leur fréquence)
        """

        # on crée une liste vide pour stocker les documents filtrés
        documents_filtre = []
        # on parcourt chaque dictionnaire dans les dictionnaires de liste et on crée un nouveau dictionnaire filtré pour chaque document
        for document in documents:
            document_filtre = {}
            # on met la valeur de clé "label" dans dict document à celle de document_filtre et aussi la valeur de vect "un dict" au token
            document_filtre["label"] = document["label"]
            tokens = document["vect"]
            # on crée un nouveau dictionnaire token_filtre
            tokens_filtre = {}
            # on parcourt chaque token dans les clés de dict tokens
            for token in tokens.keys():
            # selon le choix de l'utilisateur, si l'on veut éliminer les hapax, on exécute les codes ci-desous
                if token.lower() not in stoplist and (not non_hapax or tokens[token]>1):
                  tokens_filtre[token] = tokens[token]
              
            # après avoir terminé l'ajout des tokens dans le dictionnaire des tokens filtrés, on l'initialise dans le dictionnaire document_filtre
            document_filtre["vect"] = tokens_filtre
            # finalement, on rajoute le dictionnaire document_filtre dans la liste des ensembles de documents filtrés (la liste documents_filtre)
            documents_filtre.append(document_filtre)
        return documents_filtre

    def tokenize(self, text, tok_grm) :
        
        """
        Tokenise un texte en utilisant une expression régulière.

        argument 1 - > text (str) : Le texte à tokeniser.
        argument 2 - > tok_grm (regex) : L'expression régulière utilisée pour la tokenisation. 
        valeur de retour - > (list) : Une liste de tokens.
        """

        # on utilise la fonction findall pour diviser le texte en tokens
        return tok_grm.findall(text)

    def vectorize(self, tokens) :
        
        """
        Crée un vecteur à partir d'une liste de tokens, en calculant la fréquence de chaque token.

        argument 1 - > tokens (list) : Une liste de chaînes de caractères représentant les tokens.
        valeur de retour - > (dict) : Un dictionnaire associant chaque token à sa fréquence dans la liste.
        """

        # initialisation du hachage
        token_freq = {}
        # on parcourt des tokens
        for token in tokens:
            # si le token n'est pas dans le dictionnaire, on l'ajoute avec une fréquence de 0
            if token not in token_freq:
                token_freq[token] = 0
            # on incrémente la fréquence du token
            token_freq[token] += 1
        return token_freq

    def read_texts(self, file_names, tok_grm) :
        
        """
        Lit une liste de fichiers contenant des textes, puis tokenise chaque texte et crée un vecteur pour chaque fichier.

        argument 1 - > file_names (list) : La liste des noms de fichiers à lire.
        argument 2 - > tok_grm (regex) : L'expression régulière utilisée pour la tokenisation.
        valeur de retour - > (list) : Une liste de dictionnaires contenant les vecteurs associés à chaque fichier.
        """

        # on initialise la liste de vecteurs qui contiendra le resultat
        vectors = []
        for file_name in file_names:
          try :
            input_file = open(file_name, mode="r", encoding="utf8")
          except Exception as err :
            # ce bloc s'exécute en cas d'erreur
            print("Impossible d'ouvrir ", file_name,f" Erreur={err}")
            # on ne retourne rien et un message d'erreur s'affiche
            return None
          
          # initialisation de la liste  
          tokens = []
          # on parcourt pour chaque ligne dans le fichier d'entrée
          for line in input_file:
              # on supprime les retours à la ligne
              line = line.strip()
              # on tokenise les lignes
              toks = self.tokenize(line, tok_grm)
              # on ajoute toks à la liste tokens  
              tokens.extend(toks)
          # on ferme le fichier
          input_file.close()

          # on lance vectorize sur tokens 
          vector = self.vectorize(tokens)
          # on ajoute le vecteur obtenu à la liste
          vectors.append({'label': file_name, 'vect': vector})
        # on retourne la liste
        return vectors

    def tf_idf(self, documents) :
        
        """
        Calcule le poids TF-IDF de chaque token dans chaque document.

        argument 1 - > documents (list) : Une liste de dictionnaires contenant les documents à traiter. 
        valeur de retour - > (list) : Une nouvelle liste de dictionnaires contenant les documents avec leurs vecteurs TF-IDF associés.
        """

        documents_new = copy.deepcopy(documents)
        # on crée un ensemble contenant tous les mots de tous les docs
        mots = set()
         
        # on parcourt les documents
        for doc in documents:
            # on parcourt chaque mot du doc étant dans notre vecteur doc
            for word in doc["vect"]:
                # on ajoute word à l'ensemble mots
                mots.add(word)
        
        # on crée un dict contenant les fréquences de doc de chacun
        freq_doc = {}
        for word in mots:
            # on parcourt les documents pour vérifier si le mot est présent
            for doc in documents:
                # si le mot est présent, on incrémente la fréquence
                if word in doc["vect"]:
                    freq_doc[word] = freq_doc.get(word, 0) + 1

        # on parcourt les documents et pour chaque mot, on met à jour sa fréquence
        for doc in documents_new:
            for word in doc["vect"]:
                doc["vect"][word] = doc["vect"][word] / math.log(1 + freq_doc[word])

        # on retourne la liste de documents mise à jour
        return documents_new

    def scalar(self, vector1, vector2) :
        
        """
        Calcule le produit scalaire entre deux vecteurs.

        argument 1 - > vector1 (dict) : Dictionnaire représentant le premier vecteur.
        argument 2 - > vector2 (dict) : Dictionnaire représentant le deuxième vecteur.
        valeur de retour - > (float) : Le produit scalaire entre les deux vecteurs.
        """

        # on récupère les valeurs des vecteurs
        val1 = vector1.values()
        val2 = vector2.values()
        # on calcule le produit scalaire en utilisant la fonction sum() et la fonction zip()
        scalar_product = sum([i*j for (i, j) in zip(val1, val2)])
        return scalar_product

    def norm(self, vector) :
        
        """
        Calcule la norme d'un vecteur.

        argument 1 - > vector (dict) : Dictionnaire représentant le vecteur.
        valeur de retour - > (float) : La norme du vecteur.
        """

        norm_carre = 0
        # on parcourt chaque clé du vecteur et on calcule la norme au carré
        for key in vector:
            norm_carre += vector[key] * vector[key]
        # on calcule la racine carrée de la somme des carrés de chaque composante du vecteur
        norm = math.sqrt(norm_carre)
        return norm

    def cosine(self, vector1, vector2) :
        
        """
        Calcule la similarité cosinus entre deux vecteurs.

        argument 1 - > vector1 (dict) : Dictionnaire représentant le premier vecteur.
        argument 2 - > vector2 (dict) : Dictionnaire représentant le deuxième vecteur.
        valeur de retour - > (float) : La similarité cosinus entre les deux vecteurs.
        """

        # on calcule la norme de chaque vecteur
        norme1 = self.norm(vector1)
        norme2 = self.norm(vector2)
        # on calcule le produit scalaire entre les deux vecteurs
        scal = self.scalar(vector1,vector2)
        # on calcule la similarité cosinus en utilisant les valeurs précédentes
        cosinus = (scal/(norme1*norme2))
        return cosinus

##########################################################################################################################################

class KNNClassifier(TextVect) :
    def __init__(self) :
        
        """
        Initialise une instance de la classe KNNClassifier avec une description vide et une liste vide pour stocker les données.
        Appelle également le constructeur de la classe parente TextVect.

        valeur de retour - > (none) : Cette mèthode ne retourne rien.
        """

        # on initialise pour stocker une description du classifier
        self.description = ""
        # on initialise une liste pour stocker les données
        self.data = []
        # appel du constructeur de la classe parente TextVect
        TextVect().__init__()


    def add_class(self, label, vectors) :
        
        """
        Ajoute une nouvelle classe avec l'étiquette donnée et les vecteurs donnés à la liste de données.

        argument 1 - > label (str) : L'étiquette de la classe à ajouter.
        argument 2 - > vectors (liste) : Une liste de vecteurs à associer à la classe.
        valeur de retour - > (none) : Cette mèthode ne retourne rien.
        """

        # on ajoute une nouvelle classe à la liste de données 
        self.data.append({
            # on stocke l'étiquette
            "label": label,
            # on stocke les vecteurs
            "vectors": vectors
        })

    def add_vector(self, label, vector) :
        
        """
        Ajoute un vecteur à la liste de vecteurs de la classe correspondant à l'étiquette donnée.
        Si la classe n'existe pas encore, on la crée avec l'étiquette donnée et le vecteur donné.

        argument 1 - > label (str) : L'étiquette de la classe à laquelle on ajoute le vecteur.
        argument 2 - > vector (dict) : Le vecteur à ajouter à la classe. 
        valeur de retour - > (none) : Cette mèthode ne retourne rien.
        """

        # on parcourt chaque classe dans les données
        for cls in self.data:
            # si l'étiquette correspond à l'argument label
            if cls["label"] == label:
                # on ajoute le vecteur à la liste de vecteurs de la classe
                cls["vectors"].append(vector)
                return
        # si l'étiquette n'existe pas encore, on ajoute une nouvelle classe avec le label et le vecteur
        self.add_class(label, vector)

    def del_class(self, label) :
        
        """
        Supprime une classe de données correspondant à l'étiquette donnée.

        argument 1 - > label (str) : L'étiquette de la classe à supprimer.
        valeur de retour - > (none) : Cette mèthode ne retourne rien.
        """

        # on parcourt chaque classe dans les données
        for cls in self.data:
            # si l'étiquette correspond à l'argument label
            if cls["label"] == label:
                # on supprime la classe
                self.data.remove(cls)
                return

    def save_as_json(self, filename) :
        
        """
        Enregistre la description et les données de l'objet au format JSON dans un fichier.

        argument 1 - > filename (str) : Le nom du fichier à créer.
        valeur de retour - > (none) : Cette mèthode ne retourne rien.
        """

        # on ouvre le fichier pour écrire dedans au format JSON
        with open(filename, 'w') as input_file:
            # on écrit la description et les données au format JSON dans le fichier
            json.dump({
                "description": self.description,
                "data": self.data
            }, input_file)
        # le fichier se ferme automatiquement grace à with open
            

    def load_as_json(self, filename) :
        
        """
        Charge les données et la description de l'objet depuis un fichier JSON.

        argument 1 - > filename (str) : Le nom du fichier à lire.
        valeur de retour - > (none) : Cette mèthode ne retourne rien.
        """

        # on ouvre le fichier pour le lire au format JSON
        with open(filename, 'r') as input_file:
            # on lit les données au format JSON dans data
            data = json.load(input_file)
        # le fichier se ferme automatiquement grace à with open
        
        # on stocke la description et les données 
        self.description = data["description"]
        self.data = data["data"]

    def classify(self, vector, k, sim_func=TextVect.cosine) :
        
        """
        Classe les données selon la similarité entre le vecteur d'entrée et les vecteurs des classes.

        argument 1 - > vector (list(float)) : Le vecteur à classifier.
        argument 2 - >  k (int) : Le nombre de voisins les plus proches à utiliser pour le vote majoritaire.
        argument 3 - > sim_func (fonction) : La fonction de similarité à utiliser pour calculer la distance entre les vecteurs.
        valeur de retour - > (list(tuple(str,float))) : Une liste de k tuples contenant la classe et sa similarité moyenne avec le vecteur d'entrée.
        """

        # on initialise une liste de candidats
        list_candidats=[]
        # on récupére toutes les classes
        classes = self.data
        # on parcourt les classes
        for cla in classes:
            # on initialise une liste de similarité pour les vecteurs de la classe
            sim_vect = []
            # on parcourt chaque vecteur dans la classe
            for vec in cla["vectors"]:
                # on calcule la similarité entre le vecteur courant et le vecteur de la classe
                sim_vect.append(sim_func(vec,vector))
            # on trie les valeurs de similarité en ordre décroissant
            sim_vect.sort(reverse=True)
            # on calcule la moyenne des k valeurs de similarité les plus élevées
            avr = 0
            for i in range(min(k,len(sim_vect))):
                avr += sim_vect[i]
            # on ajoute la classe et sa similarité moyenne à la liste de candidats
            list_candidats.append((cla["label"],avr/k))
        # on trie la liste de candidats en ordre décroissant de similarité moyenne
        list_candidats.sort(key = lambda x: x[1], reverse = True)
        return list_candidats 

################################################################# !----MAIN----! #########################################################################

def main() :
  text_vect = TextVect()
  print("TextVect tests :")
  #Test pour read_dict()
  print("")

  stoplist = text_vect.read_dict("drive/MyDrive/Colab Notebooks/Data/test.txt")
  stoplist_expected = {'abord', 'à', 'â', 'a'}
  print(stoplist)
  print("read_dict() method test ->", end =" ")
  if stoplist == stoplist_expected:
    print('Success!')
  else:
    print('Failure')
  
  # Test pour filter()
  print("")
 
  documents3 = [{"label":"doc1","vect":{"test1":2,"test2":1,"test1":2}}, {"label":"doc2","vect":{"test3":3,"test2":1,"test3":3,"test3":3}}]
  stoplist = {'test1', 'à', 'â', 'a'} 
  res = text_vect.filter(stoplist,documents3,non_hapax=False)
  documents4 = [{"label":"doc1","vect":{"test2":1}}, {"label":"doc2","vect":{"test3":3,"test2":1,"test3":3,"test3":3}}]
  print("stoplist  -> ",stoplist)
  print("document => ", documents3 )
  print("filtered (non_hapax = false) => ", res )
  ress = text_vect.filter(stoplist,documents3,non_hapax=True)
  documents5 = [{"label":"doc1","vect":{}}, {"label":"doc2","vect":{"test3":3,"test3":3,"test3":3}}]
  print("filtered (non_hapax = true) => ", ress )
  print("filter() method test ->", end =" ")
  if documents5 == ress and documents4 == res:
    print('Success!')
  else:
    print('Failure')

  # Test pour vectorize()
  print("")

  vectored=text_vect.vectorize((["tst","Test","teest","Test","test"]))
  print("vector to vertorize = >",["tst","Test","teest","Test","test"])
  print("vectorised = > ", vectored)
  vect_res = {'tst': 1, 'Test': 2, 'teest': 1, 'test': 1}
  print("vectorize() method test ->", end =" ")
  if vectored == vect_res:
    print('Success!')
  else:
    print('Failure')

  # Test pour read_texts()
  print("")

  read_test = [{"label":"drive/MyDrive/Colab Notebooks/Data/test.txt","vect":{'abord':1, 'à':1, 'â':1, 'a':1}}]
  res = text_vect.read_texts( ["drive/MyDrive/Colab Notebooks/Data/test.txt"],tok_grm)
  print("file content= >",res)
  print("read_texts() method test ->", end =" ")
  if read_test == res:
    print('Success!')
  else:
    print('Failure')

  # Test pour tf_idf()
  print("")

  document_idf = [{"label":"doc1","vect":{"test1":2,"test2":1,"test1":2}}, {"label":"doc2","vect":{"test3":3,"test2":1,"test3":3,"test3":3}}]
  idf = text_vect.tf_idf(document_idf)
  expected_result = [{'label': 'doc1', 'vect': {'test1': 2.8853900817779268, 'test2': 0.9102392266268373}}, {'label': 'doc2', 'vect': {'test3': 4.328085122666891, 'test2': 0.9102392266268373}}]
  print("initial doc = >",document_idf)
  print("after term frequency-inverse calculation", idf)
  print("tf_idf() method test ->", end =" ")
  if idf == expected_result:
    print('Success!')
  else:
    print('Failure')

  # Test pour cosine()
  print("")

  testcosinvect = {"A":2,"O":1,"v":2,"r":3,"g":2,"n":9}
  testcosinvect2 = {"B":3,"F":4,"x":2,"g":4,"p":5,"t":5}
  expected_value = 0.8188504723485274

  cosi = text_vect.cosine(testcosinvect,testcosinvect2)
  print("vect A = >",testcosinvect )
  print("vect B = >",testcosinvect2 )
  print("cosine calculation => ",cosi)
  print("cosine() method test ->", end =" ")
  if cosi == expected_value:
    print('Success!')
  else:
    print('Failure')

  ##########################################################################################################################################

  KNN = KNNClassifier()
  print("\nKNNClassifier tests :")

  # Test pour add_class()
  print("")

  vect = {"test":2,"test2":1,"test":2}
  print("KNN data =>", KNN.data)
  KNN.add_class("test",[vect])
  res = [{'label': 'test', 'vectors': [{'test': 2, 'test2': 1}]}]
  print("after adding the class", KNN.data)
  print("add_class() method test ->", end =" ")
  if KNN.data == res:
    print('Success!')
  else:
    print('Failure')

  # Test pour add_vector()
  print("")

  print("KNN data =>", KNN.data)
  KNN.add_vector("test",vect)
  
  res = [{'label': 'test', 'vectors': [{'test': 2, 'test2': 1}, {'test': 2, 'test2': 1}]}]
  print("after adding the vector", KNN.data)
  print("add_vector() method test ->", end =" ")
  if KNN.data == res:
    print('Success!')
  else:
    print('Failure')
 
  # Test pour del_class()
  print("")
  
  print("KNN data =>", KNN.data)
  KNN.del_class("test")
  res = []
  print("after deleting the class", KNN.data)
  print("del_class() method test ->", end =" ")
  if KNN.data == res:
    print('Success!')
  else:
    print('Failure')

  # Test pour save_as_json() + load_as_json()
  print("")

  KNN.add_class("test",[vect,testcosinvect])
  KNN.description = "Grenoble"
  KNN.save_as_json("fichier_test")
  KNNt = KNNClassifier()
  KNNt.data = list(KNN.data)
  KNNt.description = KNN.description
  KNN.load_as_json("fichier_test")
  print("save_as_json() + load_as_json() methods test ->", end =" ")
  if KNN.description == KNNt.description and KNN.data == KNNt.data:
    print('Success!')
  else:
    print('Failure')

  
  # Test pour classify()
  print("")

  veclist = [{"test":2,"test2":1,"test":2},{"te":1,"test2":1,"test":2,"test":2},{"A":2,"O":1,"v":2,"r":3,"g":2,"n":9} ]
  veclist2 = [{"test":3,"test2":1,"test":3, "test":3},{"tef":1,"test2":1,"test":2,"test":2},{"A":2,"O":1,"v":2,"r":3,"g":2,"n":9},{"A":2,"O":6,"v":2,"r":6,"g":3,"n":6,"vrf":10,"A":2,"O":6,"v":2,"r":6,"g":3,"n":6,"vrf":10} ]
  veclist3 = [{"test":5,"test":5,"test":5,"test":5,"test":5,"yi":1},{"tyu":1,"gr":1,"b":2,"b":2}]
  veclist4 = [{"test":5,"test":5,"test":5,"test":5,"test":5,"yi":1},{"test":2,"test2":1,"test":2},{"tyu":1,"gr":1,"b":2,"b":2},{"A":2,"O":1,"v":2,"r":3,"g":2,"n":9}]
  KNN.add_class("A",veclist)
  KNN.add_class("B",veclist2)
  KNN.add_class("C",veclist3)
  KNN.add_class("D",veclist4)
  vect = {"A":2,"O":6,"v":2,"r":6,"g":3,"n":6,"vrf":10,"A":2,"O":6,"v":2,"r":6,"g":3,"n":6,"vrf":10}
  print("Document = >", KNN.description, KNN.data)
  listres = KNN.classify(vect,3,text_vect.cosine)
  print("list couple (class label, similarity) decreasing order =>",listres)
  expected_result = [('B', 0.6436446409928411), ('A', 0.4096921066594985), ('D', 0.4096921066594985), ('test', 0.30082589586913505), ('C', 0.17859639217282894)]
  print("classify() method test ->", end =" ")
  if listres == expected_result:
    print('Success!')
  else:
    print('Failure')
  
if __name__ == "__main__":
    main()
    


