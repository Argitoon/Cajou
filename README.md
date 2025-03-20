# CAJOU - Classification Automatique de Jouets
CAJOU est une application développée dans le cadre d'un projet de fin d'année de Master 2 Informatique 
à l'Université de Bordeaux.

L'application permet de classifier automatiquement des jouets dans différentes catégories
(par défaut : Jouet d'éveil, Jouet d'imitation, Playmobil, Poupée ou Véhicule). Elle offre 
également la possibilité de superviser manuellement l'entraînement de l'IA. Une fois la prédiction 
affichée, l'utilisateur peut sélectionner la bonne catégorie afin d'améliorer l'apprentissage du 
modèle et ainsi augmenter les performances de CAJOU. Toutes les fonctionnalités sont décrites 
plus en détail ci-dessous.

## Instalation (Application)
Pour que **CAJOU** fonctionne correctement, il est nécessaire d'avoir **Python** et certains modules 
installés. Heureusement, tout cela est géré automatiquement lors du premier lancement.  Il vous suffit 
alors de **double-cliquer** sur le fichier correspondant à votre système d'exploitation :  
- **Windows** : `CAJOU-Windows.bat`  
- **Linux/macOS** : `CAJOU-Linux-Darwin`

Si l'installation est interrompue, volontairement ou non, il faut supprimer le dossier `venv/` et 
relancer l’installation en exécutant à nouveau le fichier `CAJOU-Windows.bat` ou `CAJOU-Linux-Darwin`.

Si besoin, l’installation des modules peut aussi être effectuée manuellement avec la commande suivante :  

```bash
python -m pip install -r src/requirements.txt
```

## Dropbox
Notre application utilise **Dropbox** et, à son lancement, elle vous demandera de vous connecter à votre compte. Dropbox sera utilisé pour stocker certaines images ainsi que le modèle (IA) dans un dossier distant (**remote**). Ainsi, si plusieurs personnes partagent le même compte Dropbox, elles pourront toutes accéder aux mêmes images pour entraîner le modèle, ainsi qu'à d'autres fonctionnalités détaillées plus bas.  

### Instalation
Avant de lancer l'application, quelques manipulations sont nécessaires pour son bon fonctionnement. Dans le fichier `variables.yaml` du dossier `src/`, les variables **"APP_KEY"** et **"APP_SECRET"** (lignes 7 et 8) doivent être renseignées.

Pour obtenir la clé et le mot de passe de l'application, il y a deux possibilités :  
- Soit quelqu'un vous les partage,  
- Soit vous devez créer votre propre application Dropbox.

Seule une personne a besoin de créer l'application Dropbox et partager sa clé et son mot de passe. Ces informations sont générées aléatoirement lors de la création de l'application Dropbox (détaillée ci-dessous), il n'y a donc pas de risque à les partager entre vous. Toutefois, il est important de **ne pas partager ces informations publiquement**, car cela permettrait à d'autres personnes d'accéder à votre application Dropbox.

Une application Dropbox n'est pas vraiment une application classique, mais plutôt un élément lié à notre application grâce à la clé et au mot de passe. Cela permet à notre application de communiquer et de partager des informations sur les fichiers Dropbox des utilisateurs se connectant à **CAJOU**.

#### Création d'une application Dropbox  
Pour créer une application Dropbox, suivez les étapes suivantes :  

1. Allez sur le site : [dropbox.com/developers/apps](https://www.dropbox.com/developers/apps)
2. Cliquez sur **"Create app"**.
3. Choisissez l'API : Sélectionnez la seule option disponible.
4. Choisissez le type d'accès : Sélectionnez **"Full Dropbox"**, la deuxième option.
5. Nommez l'application : Vous pouvez la nommer comme bon vous semble, par exemple **CAJOU**. Cela n'aura pas d'impact sur le reste. 
6. Cliquez sur **"Create app"**.
7. Une nouvelle page s'ouvrira, vous serez dans l'onglet **Settings**. Allez dans l'onglet **"Permissions"**  
8. Cochez toutes les cases dans la partie **"Files and folders"** et **"Collaborations"**.
9. Retournez dans **Settings**.
10. Dans **"Development users"**, cliquez sur **"Enable additional users"** si vous souhaitez utiliser plusieurs comptes Dropbox ou un autre compte que celui avec lequel vous êtes en train de créer l'application.  
11. Vous trouverez votre clé et votre mot de passe dans les sections **"App key"** et **"App secret"**. Vous pouvez les copier et les coller dans le fichier `variables.yaml` de notre application.

### Une fois l'installation terminée  
Lors du lancement de l'application, un **token** vous sera demandé. Une page web s'ouvrira avec le token correspondant au compte Dropbox connecté sur votre appareil. Il vous suffira de **copier-coller** ce token, et l'application pourra fonctionner normalement.  

L'application demandera un nouveau token à chaque lancement. 

## Utilisation
Lorsque vous lancerez **CAJOU** pour la première fois, une seule fonctionnalité sera disponible : **"Entraîner CAJOU"**.  Cela signifie qu'aucun modèle d'IA n'est disponible localement. Vous pouvez cliquer sur **"Entraîner CAJOU"** pour entraîner un modèle ou en récupérer un, ce qui débloquera les autres fonctionnalités. Chacun des boutons est expliqué ci-dessous.

### 'Entraîner CAJOU'
Lorsque vous cliquez sur **"Entraîner CAJOU"**, une nouvelle page de l'application s'ouvre et vous propose quatre options : **Lancer l'entraînement**, **Arrêter l'entraînement**, **Uploader le modèle**, **Télécharger le modèle.**  

- **Lancer l'entraînement** : L'entraînement ne peut être lancé uniquement si des images sont enregistrées dans le dossier `saved_images/` de votre Dropbox.  
  - Le nombre d'images disponibles est indiqué dans l'application.
  - L'entraînement peut durer plus ou moins longtemps, allant de 3 heures à toute la nuit pour environ 100 images par classe, en fonction des performances de votre ordinateur.
  - Durant l'entraînement, vous pouvez utiliser les autres fonctionnalités de l'application sans problème. Cependant, il ne faut pas fermer l'application pour que l'entraînement continue. Les détails sur l'entraînement seront affichés dans le terminal.
  - Une fois l'entraînement terminé, le nouveau modèle remplacera **l'ancien modèle** sur votre appareil(s'il en existe un).
  - Lors du **premier lancement**, si aucune image n'est présente dans Dropbox, vous devrez les ajouter **manuellement**. Pour cela, créez un dossier `saved_images/` dans votre Dropbox. À l'intérieur, créez un sous-dossier pour **chaque catégorie** de prédiction, en respectant strictement les noms indiqués dans la variable **"CLASS_NAMES"** (ligne 2) du fichier `variables.yaml` (dossier `src/`). Ajoutez ensuite les images correspondantes dans chaque dossier. **Aucun dossier ne doit rester vide.** Une fois ces étapes effectuées, vous pourrez lancer l'entraînement.
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

## Architecture
