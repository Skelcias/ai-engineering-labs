# 04.1 · Neurones et réseaux

## Objectif
Coder en NumPy pur le passage avant (forward) d'un MLP 784 → 16 → 16 → 10,
et vérifier les formes et le nombre de paramètres.

## Ce que fait ex01_mlp_forward.py
- Entrée a0 : vecteur (784, 1). Ici un dégradé de 0 à 1 (`linspace`) pour
  tester les formes ; un vrai chiffre MNIST viendra plus tard.
- Couche cachée 1 : W1 (16, 784), b1 (16, 1), activation ReLU
- Couche cachée 2 : W2 (16, 16), b2 (16, 1), activation ReLU
- Sortie : W3 (10, 16), b3 (10, 1), sans activation (une softmax viendra en 04.2)

Chaque couche calcule z = W @ a + b, puis a = ReLU(z).

## Choix
- Vecteurs colonnes (n, 1) et convention W @ a + b :
  une ligne de W = les poids d'un neurone.
- Biais initialisés à 0 : les poids aléatoires brisent déjà la symétrie.
- Poids initialisés selon He : loi normale de moyenne 0 et d'écart-type
  √(2/n), avec n le nombre d'entrées de la couche. Objectif : garder
  l'amplitude des activations stable d'une couche à l'autre.
  Détail approfondi en 04.5.

## Ce que j'ai observé
- **Poids uniformes entre −1 et 1, biais entre −4 et 4** : z allait de −19 à +21.
  Avec une sigmoïde, seuls 2 neurones sur 16 restaient dans la zone utile,
  les autres étaient collés à 0 ou à 1 (saturation : pente quasi nulle,
  donc apprentissage bloqué). Cause : z additionne 784 termes, sa taille
  grandit comme √n.
- **Même échelle √(2/784) pour toutes les couches** : erreur corrigée.
  W2 et W3 n'ont que 16 entrées, leurs poids auraient été environ 7 fois
  trop petits, et le signal se serait éteint couche après couche.
- **Avec He** : les sorties de la couche 1 restent autour de 1 après ReLU,
  avec environ la moitié des neurones à 0, comme attendu.

## Vérifications
- Forme de chaque sortie : z1 (16, 1), z2 (16, 1), z3 (10, 1)
- Total des paramètres = 13 002, recompté à partir de `.size`,
  conforme au calcul à la main

## Notions couvertes
Neurone, poids, biais, activation (ReLU, sigmoïde), forme matricielle,
initialisation He, saturation, explosion et extinction du signal.

