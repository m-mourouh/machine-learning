

<div class="cell code" execution_count="110">

``` python
#librairires
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
```

</div>

<div class="cell markdown">

<h1>Projet de régression</h1>

</div>

<div class="cell markdown">

#### a. Générer un dataset synthètique.

</div>

<div class="cell code" execution_count="12">

``` python
# Définir la graine aléatoire pour garantir que le dataset est toujours le même
np.random.seed(42)

# Générer un dataset synthétique de 1000 lignes et 5 colonnes avec des valeurs positives
rows = 1000
cols = 5
data = np.abs(np.random.randn(rows, cols))  # Utiliser la valeur absolue pour garantir des valeurs positives

# Créer un DataFrame pandas à partir des données générées
columns = [f'feature_{i+1}' for i in range(cols)]
df = pd.DataFrame(data, columns=columns)

# Générer une variable cible continue avec du bruit et garantir des valeurs positives
weights = np.random.randn(cols)
noise = np.random.randn(rows) * 0.5  # Ajouter du bruit
target = np.dot(df[columns], weights) + noise
target = np.abs(target)  # Utiliser la valeur absolue pour garantir des valeurs positives
df['target'] = target

# Afficher les premières lignes du DataFrame
df.head()
```

<div class="output execute_result" execution_count="12">

       feature_1  feature_2  feature_3  feature_4  feature_5    target
    0   0.496714   0.138264   0.647689   1.523030   0.234153  2.404455
    1   0.234137   1.579213   0.767435   0.469474   0.542560  1.426418
    2   0.463418   0.465730   0.241962   1.913280   1.724918  0.034378
    3   0.562288   1.012831   0.314247   0.908024   1.412304  0.893652
    4   1.465649   0.225776   0.067528   1.424748   0.544383  0.986830

</div>

</div>

<div class="cell markdown">

# Partie 1 : Création du modèle à l’aide de sklearn

</div>

<div class="cell markdown">

### 1. Dataset

</div>

<div class="cell markdown">

#### b. Afficher les dimensions du dataset (nombre de lignes, nombre de colonnes)

</div>

<div class="cell code" execution_count="13">

``` python
print("Dimensions du dataset :", df.shape)
```

<div class="output stream stdout">

    Dimensions du dataset : (1000, 6)

</div>

</div>

<div class="cell markdown">

#### c. Afficher le type de données des colonnes du dataset

</div>

<div class="cell code" execution_count="14">

``` python
print("Type de données des colonnes :\n", df.dtypes)
```

<div class="output stream stdout">

    Type de données des colonnes :
     feature_1    float64
    feature_2    float64
    feature_3    float64
    feature_4    float64
    feature_5    float64
    target       float64
    dtype: object

</div>

</div>

<div class="cell markdown">

#### d. Afficher les dimensions du dataset (nombre de lignes, nombre de colonnes)

</div>

<div class="cell code" execution_count="16">

``` python
print(df.shape)
```

<div class="output stream stdout">

    (1000, 6)

</div>

</div>

<div class="cell markdown">

### 2. Type de problème

</div>

<div class="cell markdown">

#### a. Afficher que target est de type numérique

</div>

<div class="cell code" execution_count="17">

``` python
print(df['target'].dtype)
```

<div class="output stream stdout">

    float64

</div>

</div>

<div class="cell markdown">

#### b. Pourquoi il s’agit d’un problème de régression ?

</div>

<div class="cell markdown">

C'est un problème de régression car la variable cible (target) est
continue et numérique.

</div>

<div class="cell markdown">

### 3. Features selection

</div>

<div class="cell markdown">

#### a. Combien de features existe dans ce dataset ?

</div>

<div class="cell code" execution_count="18">

``` python
print("Nombre de features :", len(columns))
```

<div class="output stream stdout">

    Nombre de features : 5

</div>

</div>

<div class="cell markdown">

#### b. Extraire la partie features : x

</div>

<div class="cell code" execution_count="19">

``` python
x = df[columns]
x
```

<div class="output execute_result" execution_count="19">

         feature_1  feature_2  feature_3  feature_4  feature_5
    0     0.496714   0.138264   0.647689   1.523030   0.234153
    1     0.234137   1.579213   0.767435   0.469474   0.542560
    2     0.463418   0.465730   0.241962   1.913280   1.724918
    3     0.562288   1.012831   0.314247   0.908024   1.412304
    4     1.465649   0.225776   0.067528   1.424748   0.544383
    ..         ...        ...        ...        ...        ...
    995   1.373835   1.378470   0.115825   0.389605   2.220421
    996   1.197966   0.887080   0.286774   0.147205   0.564842
    997   1.635798   0.221042   0.069370   0.192597   2.392110
    998   2.099356   0.683223   0.114802   0.566772   0.657373
    999   0.048965   0.711411   3.112910   0.808036   0.848066

    [1000 rows x 5 columns]

</div>

</div>

<div class="cell markdown">

#### c. Extraire la target : y

</div>

<div class="cell code" execution_count="20">

``` python
y = df['target']
y
```

<div class="output execute_result" execution_count="20">

    0      2.404455
    1      1.426418
    2      0.034378
    3      0.893652
    4      0.986830
             ...   
    995    0.473629
    996    1.374932
    997    0.300431
    998    1.384891
    999    5.685302
    Name: target, Length: 1000, dtype: float64

</div>

</div>

<div class="cell markdown">

### 4. Encodage des variables catégorielles

</div>

<div class="cell markdown">

#### a. Pourquoi la vérification de type de variable est une phase importante ?

</div>

<div class="cell code">

``` python
```

</div>

<div class="cell markdown">

#### b. Écrire le code source qui permet d’afficher le type de chaque feature

</div>

<div class="cell code" execution_count="21">

``` python
print("Type de chaque feature :\n", x.dtypes)
```

<div class="output stream stdout">

    Type de chaque feature :
     feature_1    float64
    feature_2    float64
    feature_3    float64
    feature_4    float64
    feature_5    float64
    dtype: object

</div>

</div>

<div class="cell markdown">

#### c. Pour ce dataset, est ce qu’il y a des variables à encoder ?

</div>

<div class="cell markdown">

Toutes les features sont numériques, donc il n'y a pas de variables
catégorielles à encoder

</div>

<div class="cell markdown">

### 5. Split le dataset

</div>

<div class="cell markdown">

#### a. Pourquoi est-il nécessaire de diviser le dataset en training dataset et test dataset ?

</div>

<div class="cell markdown">

Pour évaluer la performance du modèle sur des données non vues durant
l'entraînement et éviter l'overfitting.

</div>

<div class="cell markdown">

#### b. Quelle fonction utiliser pour diviser le dataset en train (X_train,y_train) et en test (X_test,y_test) ? quels sont ces principaux paramètres ?

</div>

<div class="cell markdown">

Pour diviser le dataset en train et test, on utilise la fonction
`train_test_split` de la bibliothèque `sklearn.model_selection`. Les
principaux paramètres de cette fonction sont les suivants :

1.  **`X`** : Les données d'entrée (features).
2.  **`y`** : Les données de sortie (target).
3.  **`test_size`** : La proportion du dataset à inclure dans le split
    de test (par exemple, 0.2 pour 20%).
4.  **`train_size`** : La proportion du dataset à inclure dans le split
    d'entraînement (optionnel, complémentaire de `test_size`).
5.  **`random_state`** : Un entier qui fixe l'état aléatoire pour rendre
    les résultats reproductibles.
6.  **`shuffle`** : Booléen indiquant si les données doivent être
    mélangées avant d'être divisées (par défaut à `True`).

</div>

<div class="cell markdown">

#### c. Pourquoi il est important de fixer le paramètre random_state ? utiliser random_state=23

</div>

<div class="cell markdown">

Le paramètre `random_state` est crucial pour garantir la
reproductibilité des résultats. Fixer ce paramètre à une valeur
spécifique permet d'obtenir les mêmes résultats chaque fois qu'on
exécute le code.

</div>

<div class="cell markdown">

#### d. Diviser le data set en training (80%) et en test (20%). Quelle est la taille de X_test, X_train ?

</div>

<div class="cell code" execution_count="22">

``` python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# Afficher les dimensions des datasets d'entraînement et de test
print("Dimensions de X_train :", x_train.shape)
print("Dimensions de X_test :", x_test.shape)
print("Dimensions de y_train :", y_train.shape)
print("Dimensions de y_test :", y_test.shape)
```

<div class="output stream stdout">

    Dimensions de X_train : (800, 5)
    Dimensions de X_test : (200, 5)
    Dimensions de y_train : (800,)
    Dimensions de y_test : (200,)

</div>

</div>

<div class="cell markdown">

### 6. Model

</div>

<div class="cell markdown">

#### a. Quelle est la forme mathématique du modèle qui correspond à ce dataset ?

</div>

<div class="cell markdown">

Le modèle mathématique correspondant à un problème de classification
binaire comme celui-ci est la **régression logistique**. La régression
logistique permet de prédire la probabilité qu'un échantillon
appartienne à une classe donnée (par exemple, 0 ou 1) en fonction des
features.

### Forme mathématique de la régression logistique

1.  **Modèle linéaire** : Le modèle commence par une combinaison
    linéaire des features d'entrée ( x ) : \[ z = \\theta_0 + \\theta_1
    x_1 + \\theta_2 x_2 + \\ldots + \\theta_n x_n \] où :

    -   ( z ) est la somme pondérée des features.
    -   ( \\theta_0 ) est le biais (intercept).
    -   ( \\theta_1, \\theta_2, \\ldots, \\theta_n ) sont les
        coefficients (poids) associés à chaque feature ( x_1, x_2,
        \\ldots, x_n ).

2.  **Fonction sigmoïde** : La régression logistique utilise la fonction
    sigmoïde pour convertir cette combinaison linéaire en une
    probabilité comprise entre 0 et 1 : \[ h\_\\theta(x) = \\sigma(z) =
    \\frac{1}{1 + e^{-z}} \] où :

    -   ( \\sigma(z) ) est la fonction sigmoïde.
    -   ( e ) est la base du logarithme naturel (environ 2.71828).

3.  **Hypothèse du modèle** : L'hypothèse de la régression logistique
    est donc : \[ h\_\\theta(x) = \\frac{1}{1 + e^{-(\\theta_0 +
    \\theta_1 x_1 + \\theta_2 x_2 + \\ldots + \\theta_n x_n)}} \]

4.  **Décision de classification** : Pour prendre une décision de
    classification binaire, on applique un seuil (souvent 0.5) à la
    probabilité prédite : \[ \\hat{y} = \\begin{cases} 1 & \\text{si }
    h\_\\theta(x) \\geq 0.5 \\ 0 & \\text{sinon} \\end{cases} \] où :

    -   ( \\hat{y} ) est la classe prédite.

En résumé, la régression logistique modélise la relation entre les
features ( x ) et la probabilité que la sortie ( y ) soit 1 en utilisant
une combinaison linéaire des features transformée par une fonction
sigmoïde.

</div>

<div class="cell markdown">

#### b. En se basant sur LinearRegression, créer le modèle

</div>

<div class="cell code" execution_count="23">

``` python
# Créer le modèle
model = LinearRegression()
```

</div>

<div class="cell markdown">

#### c. Entrainer le modèle

</div>

<div class="cell code" execution_count="24">

``` python
model.fit(x_train, y_train)
```

<div class="output execute_result" execution_count="24">

    LinearRegression()

</div>

</div>

<div class="cell markdown">

#### d. Afficher les paramètres du modèle

</div>

<div class="cell code" execution_count="25">

``` python
print("Coefficients du modèle :", model.coef_)
print("Intercept du modèle :", model.intercept_)
```

<div class="output stream stdout">

    Coefficients du modèle : [ 0.37613121  0.39358175  1.68422795  0.31757578 -0.59116717]
    Intercept du modèle : 0.14943275948075407

</div>

</div>

<div class="cell markdown">

#### e. Sur la base des paramètres du modèle trouvé, créer une fonction predict1 (X) capable de faire des prédictions

</div>

<div class="cell code" execution_count="26">

``` python
def predict1(X):
    return np.dot(X, model.coef_) + model.intercept_
```

</div>

<div class="cell markdown">

#### f. Afficher le résultat retourné par la fonction predict1 si on lui passe le X_test

</div>

<div class="cell code" execution_count="27">

``` python
print("Prédictions pour X_test :", predict1(x_test))
```

<div class="output stream stdout">

    Prédictions pour X_test : [ 1.67951346  2.51017042  2.69034117  2.05832336  1.30786681  1.30508514
     -0.05946982  0.61654762  4.74787845  1.73643975  2.39091293  2.71532649
      3.65266604  1.31016353  2.22533505  1.14621503  4.18804496  0.46890358
      1.19269934  3.12849293  4.111894    1.21030411  0.20838048  2.3407911
      0.57028639  1.69439183  2.10373487  2.56511903  0.95333111  0.89060304
      0.58275466  1.50361433  1.57815626  1.65916761  1.72354053  1.26139691
      2.6854412   3.23031714  1.11297342  1.07498046  1.60523945  0.85604296
      1.00929017  1.17904043  0.93824251  3.438833    0.14325467  0.27240571
      2.1619295   1.46372917  1.99719376  5.17400983  3.37576631  1.93763281
      2.40723577  3.68223081  5.36715953  2.01228774  3.72475412  4.07682853
      2.36495432  1.13631304  2.08662288  2.25699756  0.52260311  1.30928201
      1.29027389  0.45198357  2.70879695  3.06581346  1.89858894  3.4950801
      1.05349247  1.47101114  3.1539173   1.35755171  2.17798183  2.51450548
      3.14448032  0.69709718  2.16530786  1.39509706  3.5401662   1.33133112
      4.31356942  1.23041453  1.75974277  2.22560774  4.25188539  2.40258525
      2.02623309  1.36918507  2.22202684  1.13211776  0.636194    4.3443471
      3.01398462  2.53777158  1.00868366  1.4958471   3.35288793  1.21999822
      1.13002825  2.3141254   0.99268204  1.7650236   0.54043324  3.26577156
      2.77323883  1.80972523  1.93585753 -0.19945202  1.27921218  1.33404255
      0.19538691 -0.13052459  0.38146605  0.97430823  2.9312185   3.65857707
      1.92628037  0.68575403  1.87558045  1.16474069  0.61433996  3.08640528
      2.68552383  1.85913575  3.73239228  2.08235374  1.93881545  2.30541512
      1.61252987  3.58775134  1.45806376  2.36129555 -0.14110055  1.19057019
      1.16722578  0.82483309  1.57089099  1.67281623  1.06021687  0.7144808
      1.12291154  1.37768458  2.05948192  2.12749941 -0.35068626  3.39896206
      3.17073583  4.39283411  1.92423265  2.80541314  1.46238603  3.73256707
      1.37558787  2.97892434  3.14250873 -0.17007269  3.04666071  0.27197041
      0.82266199  1.18590859  1.30826416  1.40935262  1.93293013  1.6434019
      2.34456103  1.11907423  1.85748292  0.88678546  2.59248496  1.35578912
      0.54155787 -0.42518401  2.3864558   0.74822935  3.99650566  0.96645319
      2.23703183  0.66366297  2.79162463  0.25190772  1.7459688   1.99938342
      1.90632935  0.59922052  0.95062969  1.72260728  1.79760245  0.52213375
      0.87237436  2.05097156  0.53945294  2.23498413  2.14143632  1.2972228
      1.84203409  2.42411204]

</div>

</div>

<div class="cell markdown">

### 7. Evaluation du modèle

</div>

<div class="cell markdown">

#### a. En utilisant sklearn, quelle est la valeur de mse du modèle ? Interpréter le résultat.

</div>

<div class="cell code" execution_count="28">

``` python
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Valeur de MSE du modèle :", mse)
```

<div class="output stream stdout">

    Valeur de MSE du modèle : 0.2299318893415373

</div>

</div>

<div class="cell markdown">

#### b. En utilisant sklearn, quelle est la valeur de r2 du modèle ? Interpréter le résultat.

</div>

<div class="cell code" execution_count="29">

``` python
r2 = r2_score(y_test, y_pred)
print("Valeur de R2 du modèle :", r2)
```

<div class="output stream stdout">

    Valeur de R2 du modèle : 0.8382989618935877

</div>

</div>

<div class="cell markdown">

#### c. Sans utiliser sklearn, créer une fonction evaluer (y_hat,y_test) capable de calculer r2 et mse et les retourner

</div>

<div class="cell code" execution_count="31">

``` python
def evaluer(y_hat, y_test):
    mse = np.mean((y_hat - y_test) ** 2)
    r2 = 1 - (np.sum((y_hat - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    return mse, r2

# Utiliser la fonction evaluer pour calculer r2 et mse
mse_manual, r2_manual = evaluer(predict1(x_test), y_test)
print("MSE (manuel) :", mse_manual)
print("R2 (manuel) :", r2_manual)
```

<div class="output stream stdout">

    MSE (manuel) : 0.2299318893415373
    R2 (manuel) : 0.8382989618935877

</div>

</div>

<div class="cell markdown">

# Partie 2 : création du modèle sans utiliser sklearn

</div>

<div class="cell markdown">

### 1. L’objectif est de créer notre propre fonction fit(X,y) qui se base sur l’algorithme de gradient descent

</div>

<div class="cell markdown">

#### a. Ce que c’est l’algorithme de gradient descent ?

</div>

<div class="cell markdown">

L'algorithme de Gradient Descent (Descente de Gradient) est une méthode
d'optimisation utilisée pour minimiser une fonction objectif, souvent
une fonction de coût, en ajustant les paramètres du modèle. Dans le
contexte des régressions linéaires, il est utilisé pour minimiser
l'erreur entre les prédictions du modèle et les valeurs réelles.

</div>

<div class="cell markdown">

#### b. Quelles sont les principales étapes de l’algorithme ?

</div>

<div class="cell markdown">

1.  **Initialisation** : Commencer avec des valeurs initiales pour les
    paramètres ( \\theta ) (par exemple, des zéros).
2.  **Calcul du Gradient** : Calculer le gradient de la fonction de coût
    par rapport à chaque paramètre ( \\theta ).
3.  **Mise à Jour des Paramètres** : Mettre à jour les paramètres (
    \\theta ) en les ajustant dans la direction opposée au gradient.
4.  **Répétition** : Répéter les étapes de calcul du gradient et de mise
    à jour des paramètres jusqu'à ce que la fonction de coût converge
    (les changements deviennent très petits) ou qu'un nombre maximal
    d'itérations soit atteint.

</div>

<div class="cell markdown">

#### c. Écrire le code source de cette fonction afin de trouver les paramètres du modèle

</div>

<div class="cell code" execution_count="32">

``` python
import numpy as np

def fit(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    X = np.c_[np.ones(m), X]  
    theta = np.zeros(n + 1)   
    
    for i in range(iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients
    
    return theta
```

</div>

<div class="cell markdown">

#### d. En utilisant la fonction fit, trouver les paramètres du modèle

</div>

<div class="cell code" execution_count="33">

``` python
theta = fit(x_train, y_train)
print("Paramètres du modèle (gradient descent) :", theta)
```

<div class="output stream stdout">

    Paramètres du modèle (gradient descent) : [ 0.2024969   0.36401731  0.38150076  1.67149605  0.30462462 -0.60238064]

</div>

</div>

<div class="cell markdown">

#### e. En utilisant les paramètres trouvés, calculer la prédiction à l’aide de la fonction predict1

</div>

<div class="cell code" execution_count="34">

``` python
def predict1(x, theta):
    x = np.c_[np.ones(x.shape[0]), x] 
    return x.dot(theta)
```

</div>

<div class="cell code" execution_count="35">

``` python
# Calculer les prédictions pour X_test
y_pred_gd = predict1(x_test, theta)
print("Prédictions (gradient descent) pour x_test :", y_pred_gd)
```

<div class="output stream stdout">

    Prédictions (gradient descent) pour x_test : [ 1.67428108  2.51528053  2.70129446  2.06400192  1.31545225  1.30931988
     -0.06417559  0.62978286  4.72722353  1.74624071  2.38923491  2.71896317
      3.62634669  1.2968533   2.21627432  1.1739459   4.18308041  0.51148774
      1.19590465  3.12574423  4.08650697  1.21549275  0.22240099  2.35928335
      0.58466828  1.69084011  2.10957873  2.55340003  0.93647911  0.8792516
      0.59583534  1.49750735  1.59815182  1.66959609  1.73561149  1.28240818
      2.68470953  3.24060322  1.13673126  1.11338293  1.62818959  0.86618247
      1.00603474  1.2064866   0.93154942  3.4299499   0.17777165  0.2874426
      2.14910491  1.48818015  2.00601043  5.13881909  3.3573048   1.9622398
      2.41352956  3.69379536  5.34620147  2.01107784  3.70649715  4.07471928
      2.37366427  1.14027038  2.10730546  2.25810269  0.55307242  1.32754822
      1.30845704  0.44284446  2.71533252  3.07969064  1.87965621  3.45700582
      1.06725395  1.47690977  3.16883408  1.36683639  2.17032886  2.51419657
      3.14622805  0.70059318  2.1761869   1.39802786  3.53626634  1.34717828
      4.28578542  1.2472235   1.77519906  2.23342672  4.2571999   2.41665271
      2.01954508  1.32505217  2.22118291  1.14981534  0.66051183  4.32623281
      3.02515208  2.53090538  1.02363829  1.50617646  3.34787903  1.23322604
      1.15489589  2.31267894  1.01142091  1.76364469  0.53218674  3.25567037
      2.7813609   1.80998675  1.93155524 -0.20055258  1.27929439  1.35015122
      0.24307404 -0.11225499  0.39215613  1.0092713   2.93378044  3.67186849
      1.93264242  0.69499915  1.87523057  1.13623917  0.65239731  3.1009289
      2.70092847  1.86056754  3.72524619  2.09825166  1.94232987  2.31818538
      1.62586394  3.58255268  1.46251141  2.34978234 -0.12055136  1.17509481
      1.19901752  0.85426623  1.57864544  1.68530032  1.07000705  0.75185413
      1.13796724  1.36255707  2.0612456   2.13725634 -0.35360397  3.38534544
      3.17591732  4.38239614  1.89477884  2.82046807  1.47827488  3.73208534
      1.3945773   2.97806155  3.1284045  -0.14990235  3.05267491  0.2643434
      0.82790998  1.19706142  1.32477158  1.43262149  1.96097926  1.65347715
      2.33787049  1.14481158  1.88482028  0.89279567  2.5878201   1.35667519
      0.54629313 -0.40964292  2.4062646   0.74460576  3.98979336  0.96076968
      2.23172984  0.68803554  2.79000532  0.25787478  1.75503799  2.0096625
      1.90338566  0.63259686  0.9642704   1.7077965   1.81639359  0.49553294
      0.88833433  2.05138772  0.57126628  2.24469496  2.12861341  1.28518664
      1.86098014  2.41301433]

</div>

</div>

<div class="cell markdown">

#### f. Évaluer la performance du modèle trouvé

</div>

<div class="cell code" execution_count="36">

``` python
def evaluer(y_hat, y_test):
    mse = np.mean((y_hat - y_test) ** 2)
    r2 = 1 - (np.sum((y_hat - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    return mse, r2
```

</div>

<div class="cell code" execution_count="37">

``` python
# Évaluation de la performance du modèle
mse_gd, r2_gd = evaluer(y_pred_gd, y_test)
print("MSE (gradient descent) :", mse_gd)
print("R2 (gradient descent) :", r2_gd)
```

<div class="output stream stdout">

    MSE (gradient descent) : 0.22918707258608906
    R2 (gradient descent) : 0.8388227589314841

</div>

</div>

<div class="cell markdown">

#### g. Comparer sa performance avec celle du modèle trouvé à l’aide de sklearn.

</div>

<div class="cell code" execution_count="40">

``` python
# Créer et entraîner le modèle sklearn
model = LinearRegression()
model.fit(x_train, y_train)
y_pred_sklearn = model.predict(x_test)

# Évaluer la performance du modèle sklearn
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print("MSE (sklearn) :", mse_sklearn)
print("R2 (sklearn) :", r2_sklearn)
```

<div class="output stream stdout">

    MSE (sklearn) : 0.2299318893415373
    R2 (sklearn) : 0.8382989618935877

</div>

</div>

<div class="cell code" execution_count="41">

``` python
print(f"Performance du modèle (Gradient Descent) : MSE = {mse_gd}, R2 = {r2_gd}")
print(f"Performance du modèle (sklearn) : MSE = {mse_sklearn}, R2 = {r2_sklearn}")
```

<div class="output stream stdout">

    Performance du modèle (Gradient Descent) : MSE = 0.22918707258608906, R2 = 0.8388227589314841
    Performance du modèle (sklearn) : MSE = 0.2299318893415373, R2 = 0.8382989618935877

</div>

</div>

<div class="cell markdown">

<h1>Projet de classification</h1>

</div>

<div class="cell markdown">

### 1. Dataset :

</div>

<div class="cell markdown">

#### a. Télécharger le dataset « heart.csv »

</div>

<div class="cell markdown">

#### b. À l’aide de pandas, ouvrir le dataset « heart.csv »

</div>

<div class="cell code" execution_count="66">

``` python
df2 = pd.read_csv('heart.csv')
```

</div>

<div class="cell markdown">

#### c. Afficher l’entête du dataset. Utiliser head(10).

</div>

<div class="cell code" execution_count="67">

``` python
df2.head(10)
```

<div class="output execute_result" execution_count="67">

       age  sex  cp  trtbps  chol  fbs  restecg  thalachh  exng  oldpeak  slp  \
    0   63    1   3     145   233    1        0       150     0      2.3    0   
    1   37    1   2     130   250    0        1       187     0      3.5    0   
    2   41    0   1     130   204    0        0       172     0      1.4    2   
    3   56    1   1     120   236    0        1       178     0      0.8    2   
    4   57    0   0     120   354    0        1       163     1      0.6    2   
    5   57    1   0     140   192    0        1       148     0      0.4    1   
    6   56    0   1     140   294    0        0       153     0      1.3    1   
    7   44    1   1     120   263    0        1       173     0      0.0    2   
    8   52    1   2     172   199    1        1       162     0      0.5    2   
    9   57    1   2     150   168    0        1       174     0      1.6    2   

       caa  thall  output  
    0    0      1       1  
    1    0      2       1  
    2    0      2       1  
    3    0      2       1  
    4    0      2       1  
    5    0      1       1  
    6    0      2       1  
    7    0      3       1  
    8    0      3       1  
    9    0      2       1  

</div>

</div>

<div class="cell markdown">

#### d. Afficher le type de données des colonnes du dataset

</div>

<div class="cell code" execution_count="68">

``` python
print(df2.dtypes)
```

<div class="output stream stdout">

    age           int64
    sex           int64
    cp            int64
    trtbps        int64
    chol          int64
    fbs           int64
    restecg       int64
    thalachh      int64
    exng          int64
    oldpeak     float64
    slp           int64
    caa           int64
    thall         int64
    output        int64
    dtype: object

</div>

</div>

<div class="cell markdown">

#### e. Afficher les dimensions du dataset (nombre de lignes, nombre de colonnes)

</div>

<div class="cell code" execution_count="69">

``` python
print(df2.shape)
```

<div class="output stream stdout">

    (303, 14)

</div>

</div>

<div class="cell markdown">

### 2. Type de problème

</div>

<div class="cell markdown">

#### a. Pourquoi il s’agit d’un problème de classification ?

</div>

<div class="cell markdown">

C'est un problème de classification car la colonne output (la cible)
contient des valeurs discrètes indiquant la présence (1) ou l'absence
(0) d'une maladie cardiaque.

</div>

<div class="cell markdown">

### 3. Features selection

</div>

<div class="cell markdown">

### a. Combien de features existe dans ce dataset ?

</div>

<div class="cell code" execution_count="70">

``` python
n_features = df2.shape[1] - 1
print(f"Nombre de features : {n_features}")
```

<div class="output stream stdout">

    Nombre de features : 13

</div>

</div>

<div class="cell code" execution_count="71">

``` python
print(df2.columns)
```

<div class="output stream stdout">

    Index(['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
           'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'],
          dtype='object')

</div>

</div>

<div class="cell markdown">

### b. Extraire la partie features : x

</div>

<div class="cell code" execution_count="72">

``` python
x = df2.drop('output', axis=1)
x
```

<div class="output execute_result" execution_count="72">

         age  sex  cp  trtbps  chol  fbs  restecg  thalachh  exng  oldpeak  slp  \
    0     63    1   3     145   233    1        0       150     0      2.3    0   
    1     37    1   2     130   250    0        1       187     0      3.5    0   
    2     41    0   1     130   204    0        0       172     0      1.4    2   
    3     56    1   1     120   236    0        1       178     0      0.8    2   
    4     57    0   0     120   354    0        1       163     1      0.6    2   
    ..   ...  ...  ..     ...   ...  ...      ...       ...   ...      ...  ...   
    298   57    0   0     140   241    0        1       123     1      0.2    1   
    299   45    1   3     110   264    0        1       132     0      1.2    1   
    300   68    1   0     144   193    1        1       141     0      3.4    1   
    301   57    1   0     130   131    0        1       115     1      1.2    1   
    302   57    0   1     130   236    0        0       174     0      0.0    1   

         caa  thall  
    0      0      1  
    1      0      2  
    2      0      2  
    3      0      2  
    4      0      2  
    ..   ...    ...  
    298    0      3  
    299    0      3  
    300    2      3  
    301    1      3  
    302    1      2  

    [303 rows x 13 columns]

</div>

</div>

<div class="cell markdown">

### c. Extraire la target : y

</div>

<div class="cell code" execution_count="73">

``` python
y = df2['output']
y
```

<div class="output execute_result" execution_count="73">

    0      1
    1      1
    2      1
    3      1
    4      1
          ..
    298    0
    299    0
    300    0
    301    0
    302    0
    Name: output, Length: 303, dtype: int64

</div>

</div>

<div class="cell markdown">

### 4. Encodage des variables catégorielles

</div>

<div class="cell markdown">

### a. Pourquoi la vérification de type de variable est une phase importante ?

</div>

<div class="cell markdown">

La vérification du type de variable est importante pour savoir quelles
variables doivent être encodées. Dans ce dataset, les variables
catégorielles sont `sex`, `cp`, `fbs`, `restecg`, `exng`, `slp`, `caa`,
et `thall`.

</div>

<div class="cell markdown">

### b. Écrire le code source qui permet d’afficher le type de chaque feature

</div>

<div class="cell code" execution_count="74">

``` python
print(x.dtypes)
```

<div class="output stream stdout">

    age           int64
    sex           int64
    cp            int64
    trtbps        int64
    chol          int64
    fbs           int64
    restecg       int64
    thalachh      int64
    exng          int64
    oldpeak     float64
    slp           int64
    caa           int64
    thall         int64
    dtype: object

</div>

</div>

<div class="cell markdown">

#### c. Pour ce dataset, est ce qu’il y a des variables à encoder ? si oui, effectuer l’encodage adéquat

</div>

<div class="cell code" execution_count="75">

``` python
# Encodage les variables
x = pd.get_dummies(x, columns=['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall'], drop_first=True)
```

</div>

<div class="cell code" execution_count="76">

``` python
x.columns
```

<div class="output execute_result" execution_count="76">

    Index(['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'sex_1', 'cp_1', 'cp_2',
           'cp_3', 'fbs_1', 'restecg_1', 'restecg_2', 'exng_1', 'slp_1', 'slp_2',
           'caa_1', 'caa_2', 'caa_3', 'caa_4', 'thall_1', 'thall_2', 'thall_3'],
          dtype='object')

</div>

</div>

<div class="cell markdown">

### 5. Split le dataset

</div>

<div class="cell markdown">

#### a. Diviser le data set en training (80%) et en test (20%)

</div>

<div class="cell code" execution_count="77">

``` python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

</div>

<div class="cell markdown">

### 6. Model

</div>

<div class="cell markdown">

#### a. Quelle est la forme mathématique du modèle qui correspond à ce dataset ?

</div>

<div class="cell markdown">

#### b. En se basant sur LogisticRegression, créer le modèle

</div>

<div class="cell code" execution_count="78">

``` python
model = LogisticRegression(max_iter=1000)
```

</div>

<div class="cell markdown">

#### c. Entrainer le modèle

</div>

<div class="cell code">

``` python
model.fit(x_train, y_train)
```

</div>

<div class="cell markdown">

#### d. Afficher les paramètres du modèle

</div>

<div class="cell code" execution_count="80">

``` python
print(model.coef_)
print(model.intercept_)
```

<div class="output stream stdout">

    [[ 0.00995507 -0.01671114 -0.00188629  0.0157984  -0.53510421 -1.16569858
       0.4538182   1.31124099  1.34365482  0.3212367   0.33897175 -0.03364818
      -0.82393654 -0.42828541  0.56232048 -1.6898486  -1.8572444  -1.04515146
       0.46673626  0.22245681  0.52421113 -0.77064914]]
    [1.25787342]

</div>

</div>

<div class="cell markdown">

#### e. Sur la base des paramètres du modèle trouvé, créer une fonction predict2 (X) capable de faire des prédictions

</div>

<div class="cell code" execution_count="82">

``` python
def predict2(x):
    return model.predict(x)
```

</div>

<div class="cell markdown">

#### f. Afficher le résultat retourné par la fonction predict2 si on lui passe le X_test

</div>

<div class="cell code" execution_count="84">

``` python
y_pred = predict2(x_test)
print(y_pred)
```

<div class="output stream stdout">

    [0 0 1 0 1 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0 0 0 0 1 1 1 1 1 0 1 0 0 0 0 1 0 1
     1 1 1 1 1 1 1 1 0 0 1 0 0 0 0 1 1 0 0 0 1 0 0 0]

</div>

</div>

<div class="cell markdown">

### 7. Evaluation du modèle

</div>

<div class="cell markdown">

#### a. Quelle est la valeur de mse du modèle ? Interpréter le résultat.

</div>

<div class="cell code" execution_count="85">

``` python
# Calcul de la valeur de mse
mse = mean_squared_error(y_test, y_pred)
print(f"Valeur de MSE : {mse}")
```

<div class="output stream stdout">

    Valeur de MSE : 0.09836065573770492

</div>

</div>

<div class="cell markdown">

#### b. En utilisant sklearn, quelle est la valeur de r2 du modèle ? Interpréter le résultat.

</div>

<div class="cell code" execution_count="87">

``` python
# Calcul de la valeur de r2
r2 = r2_score(y_test, y_pred)
print(f"Valeur de R2 : {r2}")
```

<div class="output stream stdout">

    Valeur de R2 : 0.6056034482758621

</div>

</div>

<div class="cell markdown">

### Interprétation des résultats

MSE : Un MSE plus faible indique que les prédictions sont plus proches
des valeurs réelles.

R2 : La valeur de R2 indique la proportion de la variance expliquée par
le modèle. Un R2 proche de 1 indique un bon modèle.

</div>

<div class="cell markdown">

#### c. Sans utiliser sklearn, créer une fonction capable de calculer r2 et mse et les retourner

</div>

<div class="cell code" execution_count="88">

``` python
# Calcul manuel de MSE et R2
def calculate_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
```

</div>

<div class="cell code" execution_count="89">

``` python
def calculate_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot)
```

</div>

<div class="cell code" execution_count="90">

``` python
mse_manual = calculate_mse(y_test, y_pred)
r2_manual = calculate_r2(y_test, y_pred)
print(f"Valeur de MSE (manuel) : {mse_manual}")
print(f"Valeur de R2 (manuel) : {r2_manual}")
```

<div class="output stream stdout">

    Valeur de MSE (manuel) : 0.09836065573770492
    Valeur de R2 (manuel) : 0.6056034482758621

</div>

</div>

<div class="cell markdown">

# Partie 2 : création du modèle sans utiliser sklearn

</div>

<div class="cell markdown">

### 2. L’objectif est de créer notre propre fonction fit qui se base sur l’algorithme de gradient descent

</div>

<div class="cell markdown">

#### a. Écrire le code source de cette fonction afin de trouver les paramètres du modèle

</div>

<div class="cell code" execution_count="111">

``` python
def predict2(X):
    return model.predict(X)

# Prédictions
predictions = predict2(x_test)
print(predictions)

# Evaluation du modèle
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix: \n{cm}")

# Visualiser la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

<div class="output stream stdout">

    [0 0 1 0 1 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0 0 0 0 1 1 1 1 1 0 1 0 0 0 0 1 0 1
     1 1 1 1 1 1 1 1 0 0 1 0 0 0 0 1 1 0 0 0 1 0 0 0]
    Accuracy: 0.9016393442622951
    Precision: 0.9333333333333333
    Recall: 0.875
    F1 Score: 0.9032258064516129
    Confusion Matrix: 
    [[27  2]
     [ 4 28]]

</div>



<div class="output display_data">

![](0102d35ff239571a1d7635b2284de2f585395719.png)

</div>

</div>




