#!/bin/bash

# Aller dans le dossier src/ (le répertoire où se trouve le script .sh)
cd "./src"

# Vérifier si l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "Installation en cours..."
    
    # Vérifier si Python est installé
    if ! command -v python3 &> /dev/null; then
        echo "Python non détecté, téléchargement en cours..."
        # Installation de Python (avec Homebrew ou en téléchargeant directement)
        if command -v brew &> /dev/null; then
            brew install python
        else
            echo "Homebrew n'est pas installé. Téléchargez Python manuellement à partir de https://www.python.org"
            exit 1
        fi
    fi

    # Affichage de la barre de progression
    ProgressBar 10

    # Création de l'environnement virtuel
    echo "Création de l'environnement virtuel..."
    python3 -m venv venv
    ProgressBar 15

    # Activation de l'environnement virtuel
    source venv/bin/activate

    # Mise à jour de pip et installation des dépendances
    echo "Mise à jour de pip..."
    python3 -m ensurepip
    python3 -m pip install --upgrade pip
    ProgressBar 20

    if [ -f "requirements.txt" ]; then
        echo "Installation des dépendances..."
        python3 -m pip install --no-cache-dir -r requirements.txt
    else
        echo "requirements.txt introuvable."
    fi

    ProgressBar 70
else
    echo "L'environnement virtuel existe déjà. Activation..."
fi

# Activation de l'environnement virtuel
source venv/bin/activate
ProgressBar 100

# Lancer main.py
if [ -f "main.py" ]; then
    echo "Lancement de main.py..."
    python3 main.py
else
    echo "main.py introuvable."
fi

# Pause pour garder la fenêtre ouverte
read -p "Appuyez sur [Entrée] pour quitter..."

# Fonction de barre de progression
ProgressBar() {
    local progress=$1
    for i in $(seq 1 $progress); do
        echo -n "#"
        sleep 0.05
    done
    echo " ($progress%)"
}