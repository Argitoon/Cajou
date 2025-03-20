@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Aller dans le dossier src/
cd /d "./src"

:: Vérifier si l'environnement virtuel existe
IF NOT EXIST venv (
    echo Installation en cours...
    
    :: Vérifier si Python est installé
    python --version >nul 2>&1
    IF %ERRORLEVEL% NEQ 0 (
        echo Python non détecté, téléchargement en cours...
        curl -o python_installer.exe https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe
        start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    )
    CALL :ProgressBar 10

    :: Création de l'environnement virtuel avec barre de progression
    echo Création de l'environnement virtuel...
    python -m venv venv
    CALL :ProgressBar 15

    :: Activer l'environnement virtuel
    CALL venv\Scripts\activate.bat

    :: Installation de pip et des dépendances
    echo Mise à jour de pip...
    python -m ensurepip
    python -m pip install --upgrade pip
    CALL :ProgressBar 20

    IF EXIST requirements.txt (
        echo Installation des dépendances...
        python -m pip install --no-cache-dir -r requirements.txt
    ) ELSE (
        echo requirements.txt introuvable.
    )

    CALL :ProgressBar 70
) ELSE (
    echo L'environnement virtuel existe déjà. Activation...
)

:: Activer l'environnement virtuel
CALL venv\Scripts\activate.bat
CALL :ProgressBar 100

:: Lancer main.py avec barre de progression
IF EXIST main.py (
    echo Lancement de main.py...
    python main.py
) ELSE (
    echo main.py introuvable.
)

pause
exit /b

:: === Fonction de barre de progression ===
:ProgressBar
SET /A progress=%1
for /L %%A in (1,1,%progress%) do (
    <nul set /p "=#"
    ping -n 1 -w 50 127.0.0.1 >nul
)
echo  (%progress%%%)
exit /b