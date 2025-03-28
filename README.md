# CAJOU - Classification Automatique de Jouets
**CAJOU** est une application développée dans le cadre d'un projet de fin d'année de Master 2 Informatique 
à l'Université de Bordeaux.

L'application permet de classifier automatiquement des jouets dans différentes catégories
(par défaut : Jouet d'éveil, Jouet d'imitation, Playmobil, Poupée ou Véhicule). Elle offre 
également la possibilité de superviser manuellement l'entraînement de l'IA. Une fois la prédiction 
affichée, l'utilisateur peut sélectionner la bonne catégorie afin d'améliorer l'apprentissage du 
modèle et ainsi augmenter les performances de **CAJOU**. Toutes les fonctionnalités sont décrites 
plus en détail ci-dessous.

## Installation de l'application
Pour que **CAJOU** fonctionne correctement, il est nécessaire d'avoir **Python** et certains modules 
installés. Heureusement, tout cela est géré automatiquement lors du premier lancement. Il vous suffit 
alors de **double-cliquer** sur le fichier correspondant à votre système d'exploitation : 
- **Windows** : `CAJOU-Windows.bat`  
- **Linux/macOS** : `CAJOU-Linux-Darwin.sh`

Si l'installation est interrompue, volontairement ou non, il faut supprimer le dossier `venv/` et 
relancer l’installation en exécutant à nouveau le fichier `CAJOU-Windows.bat` ou `CAJOU-Linux-Darwin.sh`.

Si besoin, l’installation des modules peut aussi être effectuée manuellement avec la commande suivante :  

```bash
python -m pip install -r src/requirements.txt
```

## Dropbox
Notre application utilise **Dropbox** et, à son lancement, elle vous demandera de vous connecter à votre compte. Dropbox sera utilisé pour stocker certaines images ainsi que le modèle (IA) dans un dossier distant (**remote**). Ainsi, si plusieurs personnes partagent le même compte Dropbox, elles pourront toutes accéder aux mêmes images pour entraîner le modèle, ainsi qu'à d'autres fonctionnalités détaillées plus bas. Il est recommandé de vous connecter au préalable à votre compte commun Dropbox avant de lancer l'application. 

### Installation
Avant de lancer l'application, quelques manipulations sont nécessaires pour son bon fonctionnement. Dans le fichier `variables.yaml` du dossier `src/`, les variables **"APP_KEY"** et **"APP_SECRET"** (lignes 8 et 9) doivent être renseignées.

Pour obtenir la clé et le mot de passe de l'application, il y a deux possibilités :  
- Soit quelqu'un vous les partage
- Soit vous devez créer votre propre application Dropbox

Seule une personne a besoin de créer l'application Dropbox et partager sa clé et son mot de passe. Ces informations sont générées aléatoirement lors de la création de l'application Dropbox (détaillée ci-dessous), il n'y a donc pas de risque à les partager entre vous. Toutefois, il est important de **ne pas partager ces informations publiquement**, car cela permettrait à d'autres personnes d'accéder à votre application Dropbox.

Une application Dropbox n'est pas vraiment une application classique, mais plutôt un élément lié à notre application grâce à la clé et au mot de passe. Cela permet à notre application de communiquer et de partager des informations sur les fichiers Dropbox des utilisateurs se connectant à **CAJOU**.

#### Création d'une application Dropbox  
Pour créer une application Dropbox, veuillez suivre les étapes suivantes :  

1. Allez sur le site : [dropbox.com/developers/apps](https://www.dropbox.com/developers/apps)
2. Cliquez sur **"Create app"**.
3. Choisissez l'API : Sélectionnez la seule option disponible **"Scoped access"**.
4. Choisissez le type d'accès : Sélectionnez **"Full Dropbox"**, la deuxième option.
5. Nommez l'application : Vous pouvez la nommer comme bon vous semble, par exemple **CAJOU**. Cela n'aura pas d'impact sur le reste. 
6. Cliquez sur **"Create app"**.
7. Une nouvelle page s'ouvrira, vous serez dans l'onglet **Settings**. Allez dans l'onglet **"Permissions"**  
8. Cochez toutes les cases dans la partie **"Files and folders"** et **"Collaborations"**.
9. Retournez dans **Settings**.
10. Dans **"Development users"**, cliquez sur **"Enable additional users"** si vous souhaitez utiliser plusieurs comptes Dropbox ou un autre compte que celui avec lequel vous êtes en train de créer l'application.  
11. Vous trouverez votre clé et votre mot de passe dans les sections **"App key"** et **"App secret"**. Vous pouvez les copier et les coller dans le fichier `variables.yaml` de notre application (dossier `src/`).

### Une fois l'installation terminée  
Lors du lancement de l'application, un **token** vous sera demandé. Une page web s'ouvrira avec le token correspondant au compte Dropbox connecté sur votre appareil. Il vous suffira de **copier-coller** ce token, et l'application pourra fonctionner normalement. Ce système permet de sécuriser l'accès à l'application uniquement aux personnes 
ayant accès au compte commun de votre application Dropbox. **CAJOU** demandera un nouveau token à chaque lancement. 

## Utilisation
Lorsque vous lancerez **CAJOU** pour la première fois, une seule fonctionnalité sera disponible : **"Entraîner CAJOU"**.  Cela signifie qu'aucun modèle d'IA n'est disponible localement. Vous pouvez cliquer sur **"Entraîner CAJOU"** qui ouvrira la page pour entraîner un modèle ou en récupérer un, ce qui débloquera les autres fonctionnalités. 
Chacun des boutons est expliqué ci-dessous. 

### 'Entraîner CAJOU'
Lorsque vous cliquez sur **"Entraîner CAJOU"**, une nouvelle page de l'application s'ouvre et vous propose quatre options : **Lancer l'entraînement**, **Arrêter l'entraînement**, **Uploader le modèle**, **Télécharger le modèle.**  

- **Lancer l'entraînement** : L'entraînement ne peut être lancé uniquement si des images sont enregistrées dans le dossier `saved_images/` de votre Dropbox.  
  - Le nombre d'images disponibles est indiqué dans l'application.
  - L'entraînement peut durer plus ou moins longtemps, allant de 3 heures à toute la nuit pour environ 100 images par classe, en fonction des performances de votre ordinateur.
  - Durant l'entraînement, vous pouvez utiliser les autres fonctionnalités de l'application sans problème. Cependant, il ne faut pas fermer l'application pour que l'entraînement continue. Les détails sur l'entraînement seront affichés dans le terminal.
  - Une fois l'entraînement terminé, le nouveau modèle remplacera **l'ancien modèle** sur votre appareil (s'il en existe un).
  - Lors du **premier lancement**, si aucune image n'est présente dans Dropbox, vous devrez les ajouter **manuellement**. Pour cela, créez un dossier `saved_images/` dans votre Dropbox. À l'intérieur, créez un sous-dossier pour **chaque catégorie** de prédiction, en respectant strictement les noms indiqués dans la variable **"CLASS_NAMES"** (ligne 2) du fichier `variables.yaml` (dossier `src/`). Ajoutez ensuite les images correspondantes dans chaque dossier. **Aucun dossier ne doit rester vide.** Vous pouvez ajouter soit 
  manuellement les images ou directement à l'aide de l'application. Une fois ces étapes effectuées, vous pourrez lancer l'entraînement.
- **Arrêter l'entraînement** : Si un entraînement est en cours, ce bouton permet de l'arrêter. 

- **Uploader le modèle** : Une fois l'entraînement terminé ou après avoir téléchargé un modèle, vous pouvez l'uploader sur Dropbox.  
  - Il sera stocké dans le dossier `model/`.  
  - Les utilisateurs connectés au même compte Dropbox pourront alors le télécharger.

- **Télécharger le modèle** : Si un modèle est disponible sur Dropbox, vous pouvez le télécharger.  
  - Il remplacera **l'ancien modèle** sur votre appareil (s'il en existe un).  

### 'Choisir une image sur cet appareil' et 'Prendre une photo'
Une fois que vous avez un modèle sur votre appareil, vous pourrez appuyer sur le bouton **"Choisir une image sur cet appareil"** ou **"Prendre une photo"**.  

- **"Choisir une image sur cet appareil"** ouvrira votre explorateur de fichiers et vous permettra de sélectionner une image depuis vos dossiers.  
- **"Prendre une photo"** allumera votre webcam (si vous en avez une) et vous permettra de prendre une photo.

Une fois l'image sélectionnée ou la photo prise, une nouvelle page s'affichera : **la page de la prédiction**.

Sur cette page, les probabilités selon l'IA que le jouet dans l'image appartienne à chaque catégorie seront d'abord affichées. En dessous, plusieurs boutons seront disponibles : un pour chaque classe, plus un bouton **"Hors Catégorie"** que vous pouvez sélectionner.  

Vous pourrez ensuite appuyer sur **"Sauvegarder"**, ce qui sauvegardera l'image dans **Dropbox**. L'image sera alors prise en compte lors du prochain entraînement. **"Hors Catégorie"** n'est pas une classe, les images sauvegardées sous cette catégorie ne seront pas comptabilisées pour l'entraînement.  

Si vous appuyez sur **"Suivant"**, vous pourrez sélectionner une autre image ou prendre une nouvelle photo, selon le mode que vous avez choisi au départ. Si vous souhaitez changer de mode, il vous faudra retourner au **Menu**.

## Ajouter une catégorie
CAJOU permet d'ajouter de nouvelles catégories pour faire évoluer le modèle de classification d'images. Pour ce faire, il suffit de modifier les lignes 2, 3 et 4 du fichier `src/variables.yaml` en ajoutant le nom et le symbole de la nouvelle catégorie.

/!\ Attention, ces points sont essentiels pour garantir le bon fonctionnement de l'application : 
- "CLASS_NAMES" :
  - Les catégories doivent être classées par ordre alphabétique.
  - Elles ne doivent pas contenir d’accents ou de caractères spéciaux.
- "CLASS_NAMES_FR" : 
  - Les catégories doivent être dans le même ordre que dans "CLASS_NAMES"
- "CLASS_SYMBOLS" :
  - Doit respecter le même ordre que "CLASS_NAMES".
  - Chaque symbole doit être un seul caractère.
  - Il ne peut pas être un caractère spécial ou comporter un accent.

Par exemple, voici à quoi ressemblent actuellement les lignes 2, 3 et 4 du fichier : 
```
"CLASS_NAMES" : ["Jouet_Eveil", "Jouet_Imitation", "Playmobil", "Poupee", "Vehicule"],
"CLASS_NAMES_FR" : ["Jouet d'éveil", "Jouet d'imitation", "Playmobil", "Poupée", "Véhicule"],
"CLASS_SYMBOLS" : ["E", "I", "Y", "P", "V"],
```
Pour ajouter une catégorie, par exemple, "LEGO", il suffirait de modifier ces deux lignes comme suit : 
```
"CLASS_NAMES" : ["Jouet_Eveil", "Jouet_Imitation", "LEGO", "Playmobil", "Poupee", "Vehicule"],
"CLASS_NAMES_FR" : ["Jouet d'éveil", "Jouet d'imitation", "LEGO", "Playmobil", "Poupée", "Véhicule"],
"CLASS_SYMBOLS" : ["E", "I", "L", "Y", "P", "V"],
```

## Architecture
Le dossier `src/` regroupe l'ensemble des fichiers Python. Dans ce dossier, nous retrouvons quatre sous-dossiers :
- `application/` : contient les fichiers relatifs à l'interface graphique de l'application.
- `classifier/` : contient les fichiers pour la classification des images et l'entraînement du modèle.
- `tests/` : contient les fichiers de test pour les différents modèles.
- `data/` : contient les données d'entraînement, de test ainsi que les poids des modèles.

◆ `application/` : L'intégralité de l'application est contenue dans le fichier `appli.py`, qui est lui-même divisé en plusieurs sections correspondant aux différentes pages de l'application.

◆ `classifier/` : Les fonctionnalités de classification des images et d'entraînement du modèle sont contenues dans le dossier. Plusieurs fichiers sont présents dans ce dossier : 
- `model.py` : contient les modèles de réseaux de neurones utilisés pour la classification des images.
- `train.py` : contient les fonctions pour l'entraînement du modèle.
- `dataloader.py` : contient les fonctions pour le chargement et la préparation des données.
- `tools.py` : contient des fonctions utiles utilisées dans différentes parties du projet.

## Commandes utiles

La liste des commandes utiles est contenue dans le fichier `src/main.py`. Pour lancer une commande, il suffit d'exécuter la commande suivante depuis le dossier `src/` : 

"python main.py {nom_de_la_commande}"

Par exemple : "python main.py" pour lancer l'application ou "python main.py test" pour tester les performances du modèle.

## Entraîner et tester manuellement le modèle

### Jeux de données
Il est nécessaire d'entraîner manuellement le modèle une fois pour pouvoir accéder aux fonctionnalités en découlant comme la prédiction des probabilités. Il est donc nécessaire de vous constituer vos propres jeux de données que vous pourrez ajouter directement depuis l'application comme expliqué plus haut. 

/!\ Dans l'optique d'une exploitation commerciale du modèle, nous vous recommandons de vous constituer votre propre dataset en collectant des images sous licences adaptées 
ou en générant vos propres données en conformité avec la réglementation en vigueur. 

Le fichier `src/variables.yaml` (lignes 20, 26 et 27) précise le chemin des jeux de données à ajouter. Si les dossiers ne sont pas présents, il faudra les créer manuellement. De la même manière que sur Dropbox, il est nécessaire de créer les sous-dossiers correspondant aux catégories ligne 2 du même fichier (il suffit de copier-coller les noms sans les guillemets). Il suffit ensuite d'ajouter les images de vos jeux de données dans les catégories correspondant (un jeu de données pour l'entraînement et un autre pour les tests). 

/!\ Une image ne doit apparaître que dans un seul jeu de données et que dans une catégorie unique. Ceci permet de garantir la qualité des performances du modèle entraîné ainsi que 
le fait que les résultats des tests n'aient pas de biais. Evitez les images pouvant appartenir à plusieurs catégories pour les données d'entraînement. Cela favoriserait une mauvaise généralisation du modèle.

### Activer l'environnement virtuel
Si vous avez installé les dépendances à l'aide des fichiers .bat ou .sh présents à la racine du projet alors un environnement virtuel sera créé. Il faut alors activer cet environnement virtuel avant de lancer l'entraînement. 

### Lancer l'entraînement
Pour lancer l'entraînement (qui se fait uniquement sur les données stockées localement) il suffit de lancer la commande suivante depuis le dossier `src/` : 

"python main.py train"

Si vous posséder un GPU, l'entraînement durera aux alentours de 40 mins et aux alentours de 2h si vous n'en possédez pas un.

### Tester le modèle
Pour tester le modèle, il faut lancer les commandes suivantes : 

- "python main.py unitest" : Calcule l'*accuracy*, la TCP moyenne, la matrice de confusion et montre les résultats des prédictions sur 10 exemples. Ce test est effectué sur des images ne pouvant avoir qu'une seule classe. 

- "python main.py multitest" : Calcule l'*accuracy*, la TCP moyenne, la matrice de confusion et montre les résultats des prédictions sur 10 exemples. Ce test est effectué sur des images pouvant avoir plusieurs classes. 

- "python main.py test" : Lance les commandes "unitest" et "multitest".

- "python main.py modeltest" : Calcule l'*accuracy* moyenne et la TCP moyenne pour tous les modèles contenus dans le dossier décrit ligne 28 du fichier `src/variables.yaml` sur un nombre d'itérations donné. Permet de comparer les modèles.

L'ensemble des résultats des tests sont contenus dans le terminal et dans le dossier `/src/tests/test_results/`.

### Format des Fichiers pour le Multitest
Dans la grande majorité de nos projets, les datasets sont organisés avec un dossier par classe, comme décrit précédemment.

Ainsi, pour réaliser les tests unitaires (unittest), vous devrez :
- Ajouter un dossier pour chaque catégorie à prédire.
- Insérer les images correspondantes dans le dossier indiqué à la ligne 26 du fichier variables.yaml, sous la clé `"TEST_UNICLASS_DATASET"`.

Cependant, le format de données est différent pour le multitest, car une image peut appartenir à plusieurs classes simultanément. Contrairement aux tests unitaires, il n'est pas nécessaire de créer un dossier par classe. Les images doivent simplement être nommées selon le format suivant :
```
{SYMBOLE}_{NOM}_{NUMERO}.png
```

- {SYMBOLE} → Liste des symboles des classes associées à l’image. Les symboles pour chaque classe sont indiqués dans la ligne 4 du fichier `variables.yaml`, sous la clé `"CLASS_SYMBOLS"`.
- {NOM} → Nom de l’image.
- {NUMERO} → Identifiant unique pour éviter les doublons.

Par exemple si nous avons deux classes : "V" pour "Véhicule" et "Y" pour "Playmobil", Une image appartenant aux deux classes pourrait être nommée :

```
VY_Voiture_Playmobil_1.png
```
