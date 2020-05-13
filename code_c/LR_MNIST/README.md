# Projet CLANU - INSA GE #

Ce document donne les informations pour l'utilisation du canevas.

L'aide et le détails des fonctions fournies sont disponibles dans le fichier ./html/index.html


## PRE-REQUIS ##

Avant de commencer, l'environnement QtCreator doit etre fonctionnel sur votre PC. 
On recommande l'utilisation d'un compilateur g++ > 5.0
La bibliothèque Qt n'est pas nécessaire.


## CMAKE ## 

Ce projet s'appuie sur l'outil CMAKE qui permet de gérer plusieurs executables dans le même projet.

1. La premiere étape sera d'installer CMAKE sur votre machine.

	* Windows : https://cmake.org/files/v3.11/cmake-3.11.0-rc1-win64-x64.msi
	* Linux : utiliser votre gestionnaire de paquets : 
		ubuntu/debian    : sudo apt-get install cmake cmake-curses-gui cmake-qt-gui 
		fedora	         : sudo dnf install cmake cmake-curses-gui cmake-qt-gui
		centos/fedora/RH : sudo yum install cmake cmake-curses-gui cmake-qt-gui

2. Deuxièmement, il faudra créer un répertoire de compilation. Ce répertoire *ne doit pas etre dans le répertoire des sources*.
Aussi, les noms de répertoire ne doivent pas comporter de caractères spéciaux (dont : accents, espaces, caractères chinois, ...).
Exemple, si vos sources sont dans /home/tgrenier/clanu/LR_MNIST
	créer un répertoire pour la compilation /home/tgrenier/clanu/LR_MNIST-Build 
	
3. Troisièmement, initialiser le projet en exécutant cmake.

+	Sous windows et sous linux, une interface graphique est disponible. Il faudra que le chemin vers le "source code" soit le chemin du répertoire contenant le CMakeLists.txt du projet.
	Dans l'exemple précédent : /home/tgrenier/clanu/LR_MNIST
	Puis le répertoire "where to build the binaries" sera le répertoire créé précédement: 
	/home/tgrenier/clanu/LR_MNIST-Build
	Il faut ensuite cliquer sur **Configure** et renseigner les informations CMAKE_BUILD_TYPE: 
	
	+ taper "Debug" pour une version debogable (moins rapide mais où l'on peut mettre des breakpoint)
	+ taper "Release" pour une version optimisée et parallèle
	
	Sous windows, si Cmake demande quel "generator" utiliser (une fenetre sugit lors du clic sur Configure) choisir MinGW Makefiles.

+	Si une option a été modifiée, il faudra relancer l'étape **Configure** (re-cliquer).
	
+	Si tout va bien, l'option **Generate** est disponible. Cette étape génére les fichiers compilations. 
	Pour celà, juste cliquer sur Generate.

+	Il est aussi possible de faire ceci en mode console:
	
	> cd /home/tgrenier/clanu/LR_MNIST-Build
	> ccmake /home/tgrenier/clanu/LR_MNIST
	taper sur la touche 'c' pour configurer le projet
	completer la valeur CMAKE_BUILD_TYPE en tapant "Debug" ou "Release" (sans les guillemets)
	taper sur la touche 'g' pour générer les fichiers de compilation


4. Quatrièment étape : ouvrir le projet avec QTCreator.

Ouvrir le CMakeFile.txt du répertoire LR_MNIST comme projet.
Dans l'onglet "Projets" de Qtcreator, changer le répertoire de compilation à LR_MNIST-Build.
Il sera possible de passer de Debug à Release directement dans QtCreator.
