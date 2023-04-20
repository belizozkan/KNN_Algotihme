# KNN_Algorithme

## README.md

C'est un fichier qui informe sur le contenu de tous les autres fichiers déposés sur GitLab.

## Script + API + Main.py

Le code est écrit en Python et utilise les modules re, copy, os, json, math, IPython.testing et IPython.utils. 
Les instructions pour l'installation de ces modules sont disponibles en ligne. Le code utilise également des expressions régulières pour tokeniser des textes.
Les règles d'expression régulière sont définies dans la variable tok_grm.

Ce code Python contient deux classes : TextVect et KNNClassifier.

------> SCRIPT <------ 

### TextVect ###

La classe TextVect permet de traiter des données textuelles et fournit des méthodes de tokenisation, de filtrage et de vectorisation du texte. Elle comprend également des méthodes de calcul de la similarité tf-idf et cosinus. Elle contient les méthodes suivantes :

# Méthode read_dict #
Cette méthode permet de lire un fichier de stoplist (une liste de mots à ignorer) et de la stocker dans TextVect. Elle prend en argument le nom du fichier stoplist à lire et renvoie la liste des mots à ignorer.

# Méthode filter #
Cette méthode permet de filtrer une liste de documents (représentés sous forme de dictionnaires avec une clé "label" et une clé "vect") en enlevant les mots présents dans la stoplist. Elle prend en argument la stoplist (une liste de mots), la liste de documents à filtrer et un booléen non_hapax indiquant si l'on veut enlever les mots qui n'apparaissent qu'une seule fois dans l'ensemble des documents ou pas. Elle renvoie la liste de documents filtrée.

# Méthode tokenize #
Cette méthode permet de tokeniser un texte à partir d'une expression régulière. Elle prend en argument le texte à tokeniser et une expression régulière tok_grm. Elle renvoie une liste de tokens.

# Méthode vectorize #
Cette méthode permet de créer un vecteur à partir d'une liste de tokens. Elle prend en argument une liste de tokens et renvoie un dictionnaire avec les fréquences des tokens.

# Méthode read_texts #
Cette méthode permet de lire des fichiers de textes et de les transformer en vecteurs. Elle prend en argument une liste de noms de fichiers à lire et une expression régulière tok_grm. Elle renvoie une liste de dictionnaires, chacun représentant un fichier avec une clé "label" et une clé "vect".

# Méthode tf_idf #
Cette méthode permet de calculer le score tf-idf pour chaque mot dans chaque document. Elle prend en argument la liste de documents (représentés sous forme de dictionnaires avec une clé "label" et une clé "vect"). Elle renvoie la liste de documents avec les scores tf-idf.

# Méthode scalar #
Cette méthode permet de calculer le produit scalaire entre deux vecteurs. Elle prend en argument deux dictionnaires représentant des vecteurs.

# Méthode norm #
Cette méthode permet de calculer la norme d'un vecteur. Elle prend en argument un dictionnaire représentant un vecteur.

# Méthode cosine #
Cette méthode permet de calculer la similarité cosinus entre deux vecteurs. Elle prend en argument deux dictionnaires représentant des vecteurs.

### KNNClassifier ###

La classe KNNClassifier permet de classifier des vecteurs de textes en utilisant l'algorithme des k plus proches voisins (KNN). Elle hérite de la classe TextVect, ce qui lui permet d'utiliser les méthodes définies dans cette dernière pour effectuer la classification.

# Méthode add_class #
Cette méthode permet d'ajouter une classe de documents à la base de données. Elle prend en argument une étiquette pour la classe et une liste de dictionnaires représentant les vecteurs de cette classe.

# Méthode add_vector #
Cette méthode permet d'ajouter un vecteur à une classe existante. Elle prend en argument une étiquette pour la classe et un dictionnaire représentant le vecteur à ajouter.

# Méthode del_class #
Cette méthode permet de supprimer une classe de notre ensemble de données. Elle prend en paramètre l'étiquette de la classe à supprimer.

# Méthode save_as_json #
Cette méthode permet de sauvegarder notre ensemble de données au format JSON. Elle prend en paramètre le nom du fichier dans lequel nous souhaitons enregistrer les données.

# Méthode load_as_json #
Cette méthode permet de charger notre ensemble de données depuis un fichier JSON. Elle prend en paramètre le nom du fichier à charger.

# Méthode classify #
Cette méthode permet de classer un vecteur donné en utilisant l'algorithme KNN. Elle prend en paramètre le vecteur à classer, le nombre de voisins à considérer (k) et la fonction de similarité à utiliser (par défaut, la fonction de similarité cosinus de la classe TextVect est utilisée).

------> MAIN <------ 

Le main est un exemple d'utilisation des classes TextVect et KNNClassifier. Il montre comment on peut créer une instance de chaque classe et utiliser les différentes méthodes de ces classes.

NOTE : Veuillez cliquer sur le lien dessous pour accéder aux résultats des tests. Veillez à ce que Google Colab ait accès à vos fichiers stockés sur Drive avant d'éxécuter le code. 

https://colab.research.google.com/drive/1WmTsgq4uATvTxnh9QzIt0W2d-8GVl5u1#scrollTo=pv7wB0wE_Bc2

------> TESTS <------

Tous les tests sont réussis.

## TextVect tests

# Read_dict() method test #

Test.txt ne contient que quatre mots à éviter qui sont les suivants : {'â', 'à', 'abord', 'a'}.

# Filter() method test # 

Ces résultats montrent l'effet de l'utilisation d'une stoplist pour filtrer des mots dans un document. Dans ce cas, la stoplist contient les mots "â", "a", "à" et "test1". Le document d'origine contient deux documents (doc1 et doc2) avec des vecteurs représentant des mots et leur fréquence dans chaque document.

Après l'application de la stoplist, les mots "â", "a", "à" et "test1" ont été supprimés, ce qui a entraîné la suppression complète du vecteur de doc1 car il ne contenait que des mots de la stoplist. 

Lorsque l'option "non_hapax" est activée, seuls les mots qui apparaissent dans plus d'un document sont conservés. Cela a pour effet de supprimer le mot "test2" du vecteur de doc1 ainsi que celui du doc2 car il n'apparaît que dans ces documents et est donc considéré comme un hapax.

# Vectorize() method test # 

Les résultats montrent une opération de vectorisation qui a été appliquée à une liste de mots: "tst", "Test", "teest", "Test" et "test". La vectorisation a été effectuée en comptant le nombre d'occurrences de chaque mot dans la liste et en créant un vecteur qui reflète ces fréquences. Ainsi, le résultat de la vectorisation est un dictionnaire avec les mots en tant que clés et leur nombre d'occurrences en tant que valeurs correspondantes: {'tst': 1, 'Test': 2, 'teest': 1, 'test': 1}. On remarque que les mots "Test" et "test" sont traités comme des mots différents en raison de leur casse.

# Read_texts() method test #

Les résultats indiquent que le fichier "drive/MyDrive/Colab Notebooks/Data/test.txt" a été lu et qu'un dictionnaire de vecteurs a été créé pour représenter le contenu du fichier. Le dictionnaire de vecteurs montre que le fichier contient les mots "a", "à", "â" et "abord".

# Tf_idf() method test #

Ces résultats montrent le processus de calcul de la fréquence des mots inverses (TF-IDF) pour deux documents. Le calcul de l'IDF prend en compte le nombre total de documents et le nombre de documents contenant le mot considéré. On peut voir que les valeurs de TF-IDF sont plus grandes pour les mots qui sont moins fréquents dans l'ensemble des documents, car ils ont une plus grande valeur IDF.

# Cosine() method test #

La première étape est de représenter les vecteurs A et B sous forme de dictionnaires de mots et de fréquences d'apparition respectives. Dans le vecteur A, les mots sont 'A', 'O', 'v', 'r', 'g', et 'n' et leurs fréquences d'apparition respectives sont 2, 1, 2, 3, 2 et 9. Dans le vecteur B, les mots sont 'B', 'F', 'x', 'g', 'p', et 't' et leurs fréquences d'apparition respectives sont 3, 4, 2, 4, 5 et 5.

Ensuite, nous utilisons la formule de la similarité cosinus pour calculer la similarité entre les deux vecteurs. Le résultat de ce calcul est 0,8188504723485274.

## KNNClassifier tests

# Add_class() method test #

Les résultats montrent une liste de données KNN initialement vide. Ensuite, une classe "test" avec un vecteur est ajoutée à la liste. Cela signifie que la classe "test" est maintenant considérée comme faisant partie des données KNN et peut être utilisée pour la classification des nouvelles instances à l'aide de l'algorithme KNN.

# Add_vector() method test #

Le résultat indique une mise à jour des données KNN. Avant la mise à jour, il existait une seule classe "test" avec un vecteur contenant deux mots "test" et "test2". Après la mise à jour, un nouveau vecteur avec les mêmes mots "test" et "test2" a été ajouté à la classe "test", résultant en un vecteur de la classe avec deux vecteurs de même taille.

# Del_class() method test #

Le résultat indique simplement que la liste contenant les classes de KNN a été vidée après la suppression de la seule classe existante dans cette liste.

# Save_as_json() + load_as_json() method test #

Le resultat est un fichier généré automatiquement et nommé "fichier_test" contenant la description et les données au format JSON.

# Classify() method test #

Les résultats montrent que l'algorithme a analysé le document "Grenoble" et a attribué des étiquettes (labels) à différentes parties du document, représentées par des vecteurs.

Il y a cinq étiquettes au total, appelées A, B, C, D et test. 

La liste de couples (étiquette de classe, similarité) en ordre décroissant indique que l'algorithme a trouvé que le vecteur associé à l'étiquette "B" était le plus similaire au document analysé, suivi par les vecteurs associés aux étiquettes "A" et "D", qui ont la même similitude. L'étiquette "test" est la quatrième plus similaire, suivie de l'étiquette "C" qui est la moins similaire.

Ces résultats suggèrent que l'algorithme a identifié certains thèmes ou sujets dans le document qui sont plus étroitement liés à l'étiquette "B" que aux autres étiquettes. 

## Compte-rendu des bogues et améliorations possibles pour le code 

1.  Dans la fonction read_dict, si une erreur se produit à l'ouverture du fichier, le bloc try-except ne fait que renvoyer None. Cela peut causer des problèmes si la fonction est appelée plus tard et que self.stoplist n'a pas été initialisé correctement.

2.  Le code utilise des expressions régulières pour le traitement du texte, mais il serait plus efficace d'utiliser une librairie spécialisée comme NLTK, Spacy ou même des packages Python tels que nltk.tokenize pour la tokenization et le prétraitement de texte.

3.  Le code utilise des listes et des dictionnaires Python standard pour stocker les vecteurs et les tokens. Cependant, il serait plus performant d'utiliser des structures de données telles que des tableaux Numpy ou des dictionnaires Python modifiés tels que les defaultdict pour stocker des vecteurs ou des tokens.

4.  Il serait possible d'accélèrer le traitement de grandes quantités de données grace à la programmation parallèle. En utilisant des bibliothèques telles que multiprocessing ou dask, on peut paralléliser certaines tâches telles que la tokenization et la vectorisation, ce qui peut accélérer considérablement l'exécution du code.

5.  La fonction filter effectue actuellement une boucle sur chaque document et chaque token, ce qui peut être très lent pour les grands ensembles de données. En utilisant des opérations vectorielles, la fonction pourrait être optimisée pour être beaucoup plus rapide.

6.  Le code ne fournit pas actuellement de fonctions pour effectuer des tâches telles que la normalisation, la lemmatisation ou la suppression de la ponctuation, qui peuvent être utiles dans le cadre du prétraitement des données.

7.  Le code utilise actuellement la pondération TF-IDF pour la vectorisation des documents. Cependant, il pourrait être intéressant d'ajouter d'autres méthodes de pondération telles que BM25 ou Okapi BM25, qui peuvent mieux fonctionner pour certains types de données.

8.  Le code ne dispose pas de mèthodes privées. En python, il n'est pas possible de rendre les mèthodes privées.Pourtant les mèthodes privées peuvent etre indiquées par la symbole "_" par laquelle serait suivi le nom de la mèthode.

9.  Le code ne dispose de qu'une seule mèthode de similarité (cosinus). Il serait intéressant d'ajouter également d'autres mèthodes de similarité telles que la distance euclidienne, la distance de Manhattan, la similarité de Jaccard, la distance de Chebyshev et la similarité de Pearson.

10. Il serait intéressant d'utiliser une méthode différente pour calculer la moyenne des scores de similarité telle que la moyenne pondérée. Cette méthode donnerait plus de poids aux vecteurs les plus similaires, ce qui pourrait améliorer la précision de la classification.

11. Il est possible d'utiliser la mèthoe TDF-IDF pour améliorer le résultat de la mèthode classify.


## text.txt

C'est la stoplist utilisée pour la mèthode Read_dict().


