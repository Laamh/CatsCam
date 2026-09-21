# CatsCam — Détection et identification d'animaux domestiques en temps réel

## Présentation

CatsCam est un projet expérimental de vision par ordinateur visant à surveiller une pièce du domicile, détecter la présence d'animaux domestiques, les identifier individuellement, et protéger la vie privée des occupants humains en floutant automatiquement leur présence à l'image.

Le projet repose sur un flux vidéo en direct, un modèle de détection généraliste pour repérer les sujets dans l'image, et un second modèle entraîné spécifiquement pour reconnaître les cinq animaux du foyer : Bom, Broutch, Titus, Toundra et Yuki.

Il s'agit d'un prototype personnel, développé comme terrain d'expérimentation autour des architectures de détection et de classification en temps réel, avant d'envisager une version plus aboutie.

## Fonctionnement général

Le pipeline se déroule en trois étapes :

1. **Acquisition du flux vidéo** : une caméra iPad diffuse un flux MJPEG sur le réseau local via l'application DroidCam, récupéré directement en HTTP sans matériel ni pilote supplémentaire.
2. **Détection généraliste** : un modèle YOLOv8 pré-entraîné identifie les sujets présents dans chaque frame (personne, chat, chien) et fournit leurs coordonnées.
3. **Traitement conditionnel** :
   - Si un humain est détecté, la zone correspondante est pixelisée avant affichage.
   - Si un chat ou un chien est détecté, le crop correspondant est transmis à un second modèle, entraîné spécifiquement sur les cinq animaux du foyer, qui tente de l'identifier individuellement avec un taux de confiance associé.

## Choix techniques

### La caméra : un iPad plutôt qu'une caméra IP dédiée

Le projet n'utilise pas de caméra de surveillance classique, mais un iPad existant, transformé en caméra IP via l'application DroidCam. Ce choix a été fait par pragmatisme parce que j'ai juste pas de matériel de ce genre a ma disposition... mais il introduit des contraintes réelles :

- Le flux transite en MJPEG (succession d'images JPEG), un format plus simple qu'un vrai flux compressé H.264, ce qui pèse davantage sur la bande passante et pénalise la fluidité à haute résolution.
- L'iPad doit rester allumé, à l'écran actif, et l'application au premier plan — iOS suspend agressivement les applications en arrière-plan, ce qui coupe le flux si l'appareil se verrouille.
- La latence dépend directement de la qualité du réseau Wi-Fi local, sans possibilité d'optimisation applicative si ce n'est de la configuration de base, résolution et fréquence d'image

### Détection : YOLOv8

YOLOv8 a été retenu pour la détection généraliste (personne, chat, chien) car il offre un bon compromis vitesse/précision, tourne correctement en temps réel sur GPU grand public, et dispose d'un écosystème Python mature (bibliothèque Ultralytics) facilitant l'intégration.

D'autres approches ont été envisagées et écartées à ce stade : MediaPipe, plus léger mais moins précis, adapté à des contraintes matérielles plus faibles (Raspberry Pi par exemple), mais encore une fois ici je ne possède pas de raspberry pi (malheureusement) ; Faster R-CNN, plus précis mais nettement plus lourd, pertinent seulement si la précision prime sur la latence, j'ai essayé et pour parler de manière plus famillière il as PLIÉ ma carte graphique et en plus c'était nettement plus compliqué à mettre en place et je voulais gagner le plus de temps possible

### Identification individuelle : classification par transfert

L'identification des animaux repose sur un second modèle, YOLOv8 en mode classification, entraîné sur un jeu de données constitué manuellement : des crops extraits automatiquement du flux vidéo lors des détections, puis triés à la main par animal via un petit outil développé par une IA ou j'appuie sur une touche en fonction de l'animal que j'identifie pour me faciliter le travail.

### Environnement d'exécution

Le projet tourne sur GPU (RTX 4070 Super) via CUDA, ce qui permet d'utiliser des variantes de modèle plus lourdes (YOLOv8x) sans sacrifier la fluidité. L'utilisation du GPU impose Python 3.12, les wheels PyTorch compatibles CUDA n'étant pas encore disponibles pour les versions plus récentes de Python au moment du développement, j'ai donc du revenir à une version plus ancienne de python et me battre avec les incompatibilités de librairies

## À propos des taux de détection observés

Les captures d'écran jointes à ce projet montrent parfois des taux de confiance de 100 % sur l'identification individuelle des animaux. Ce chiffre doit être interprété avec prudence et ne reflète pas une fiabilité réelle à ce niveau.

Le jeu de données d'entraînement actuel est restreint (quelques centaines d'images par animal, collectées sur une courte période, dans un environnement quasi identique à chaque capture : même pièce, même éclairage, angles de caméra similaires). Dans ces conditions, le modèle apprend à reconnaître des caractéristiques très spécifiques au contexte de capture plutôt qu'à généraliser sur l'apparence réelle de l'animal ce qu'on pourrai qualifier d'overfitting. Un taux de confiance élevé dans ce cadre traduit la facilité du cas particulier testé, pas une performance représentative en conditions variées (autre pièce, autre saison, animal dans une posture inhabituelle, éclairage différent et même possiblement animaux étrangers de famille/amis).

Ce point constitue précisément la limite principale du prototype actuel, détaillée ci-dessous.

## Axes d'amélioration pour une version professionnelle

Le prototype actuel valide une approche technique, mais nécessiterait un travail conséquent avant d'être considéré comme une solution robuste et présentable en contexte professionnel.

1. **Élargir et diversifier le jeu de données** : multiplier les sessions de capture sur plusieurs semaines, dans différentes pièces, avec des variations d'éclairage naturel et artificiel, pour réduire le risque de surapprentissage contextuel.
2. **Mettre en place une validation croisée rigoureuse** : au-delà d'un simple split train/val, utiliser une validation croisée (k-fold) pour obtenir une mesure de performance plus fiable et moins dépendante du hasard du découpage.
3. **Ajouter un jeu de test totalement disjoint des sessions d'entraînement** : capturé à une date différente, pour mesurer la vraie capacité de généralisation du modèle plutôt qu'une performance en circuit fermé.
4. **Remplacer ou compléter la caméra iPad par du matériel dédié** : une caméra IP fixe avec un vrai flux compressé (RTSP/H.264) réduirait la latence, stabiliserait la qualité d'image, et supprimerait la dépendance à un appareil ABSOLUMENT pas conçu pour cet usage.
5. **Industrialiser le pipeline d'entraînement** : automatiser le ré-entraînement périodique du modèle de classification à mesure que le dataset s'enrichit, avec suivi de version des modèles et des jeux de données (par exemple avec DVC ou MLflow) pour l'instant je dois traiter les images à la main comme expliqué précédemment.
6. **Ajouter une détection de mouvement en amont** : ne déclencher l'inférence YOLO que lors d'un changement significatif dans l'image, pour réduire la charge GPU et permettre un déploiement sur du matériel plus modeste, même si en essayant avec d'autres modèles la charge GPU peux également être plus légère, les CPU sont à oublier ou alors vraiment un très solide.
7. **Renforcer la gestion des erreurs et la résilience du flux** : reconnexion automatique en cas de coupure réseau ou de perte du flux DroidCam, sans intervention manuelle, impossible encore une fois à cause de l'ipad qui attends mon mdp/face id pour revenir à lui.
8. **Historiser et journaliser les détections** : conserver un registre horodaté des passages de chaque animal (et des floutages humains), utile pour des statistiques d'usage ou de comportement.
9. **Étendre la protection de la vie privée** : envisager un floutage plus robuste que le pixelisé simple (par exemple, floutage gaussien fort combiné à un masque de forme plutôt qu'une simple boîte rectangulaire), et réfléchir à la conservation ou non des frames contenant des humains, même floutées. Je sais que certains outils permettent la depixelisation, cette protection vie privée pourrait être "desarmée" la nuit par exemple à des fins de surveillance et là ou on cherche à identifier un intru potentiellement humain
10. **Documenter et tester formellement les performances du modèle** : mesurer précision, rappel et F1-score par classe animale plutôt que de se fier à un taux de confiance ponctuel affiché à l'écran, et présenter ces métriques de façon transparente plutôt que les meilleurs cas observés. Cette partie est assez chronophage même si automatisable avec de l'IA, mais ça me dérangeais un peu que le travail d'analyse soit moins personnel, également j'ai seulement passé une journée sur ce projet je pense que les métriques seront pas très pertinentes ni représentatives d'un vrai projet.

## Limites connues du prototype actuel (résumé)

- Le jeu de données est encore restreint et peu diversifié, ce qui limite la fiabilité de l'identification individuelle hors du contexte de capture.
- Le flux vidéo dépend d'un iPad maintenu allumé en continu, une contrainte peu adaptée à un usage prolongé sans matériel dédié.
- Aucune persistance des détections n'est mise en place à ce stade ; le système fonctionne en flux continu sans historique.
- Le floutage des humains repose sur la détection YOLO générique, qui peut manquer une détection dans certaines conditions (occlusion partielle, angle inhabituel), sans garantie absolue de confidentialité.

## Conclusion

CatsCam, en l'état, démontre la faisabilité technique d'un pipeline de détection et d'identification d'animaux en temps réel à partir de matériel grand public, avec une attention portée à la vie privée des occupants humains. Il s'agit d'une preuve de concept, pas d'un produit fini : les axes d'amélioration listés ci-dessus tracent le chemin vers une version plus robuste, généralisable et exploitable dans un cadre professionnel.

## POST-SCRIPTUM

Même si ma candidature n'est pas retenue, je vous remercie d'avoir proposé ce petit exercice, je me suis en réalité bien amusé avec mes animaux à la maison et je continuerai surement à développer ce petit outil pour faire des statistiques de leur temps de sommeil, et leur temps passé dans une pièce ou une autre.
