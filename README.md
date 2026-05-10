\# MNQ IFVG Strategy Backtester \& Quant Research Framework



Framework de recherche quantitative et de backtesting développé en Python permettant d’analyser, optimiser et évaluer une stratégie algorithmique multi-timeframe sur les futures MNQ (Micro E-mini Nasdaq-100).



Le projet combine :

\- détection algorithmique de structures de marché ;

\- génération de signaux IFVG ;

\- moteur de backtesting ;

\- gestion du risque ;

\- analyse statistique avancée ;

\- optimisation paramétrique ;

\- dashboard interactif Streamlit.



\---



\# Aperçu du projet



L’objectif du projet est de construire un environnement de recherche quantitatif complet autour d’une stratégie discrétionnaire transformée en logique algorithmique.



Le framework permet notamment :



\- le chargement et le nettoyage de données futures MNQ ;

\- la sélection automatique du contrat actif ;

\- l’analyse multi-timeframe ;

\- la détection algorithmique de signaux IFVG ;

\- la génération de trades ;

\- l’évaluation statistique des performances ;

\- l’optimisation des paramètres de stratégie ;

\- l’étude de la robustesse mensuelle et annuelle.



Le moteur a été conçu dans une logique de recherche empirique et d’itérations successives.



\---



\# Pipeline du framework



```text

Données Futures MNQ

&#x20;       ↓

Préparation \& nettoyage

&#x20;       ↓

Sélection du contrat actif

&#x20;       ↓

Analyse multi-timeframe

&#x20;       ↓

Détection IFVG

&#x20;       ↓

Filtres de marché \& confirmations

&#x20;       ↓

Gestion du risque

&#x20;       ↓

Backtesting

&#x20;       ↓

Analyse statistique

&#x20;       ↓

Optimisation paramétrique

&#x20;       ↓

Dashboard \& visualisation

```



\---



\# Fonctionnalités principales



\## Backtesting algorithmique



\- moteur de backtesting custom en Python ;

\- gestion des entrées/sorties ;

\- gestion du risque par trade ;

\- take profit / stop loss ;

\- break-even automatique ;

\- trailing stop dynamique ;

\- limitation du nombre de trades par session ;

\- cooldown entre trades.



\---



\## Analyse multi-timeframe



Le framework utilise plusieurs horizons temporels afin de :



\- déterminer le biais de marché ;

\- analyser la structure ;

\- détecter les zones IFVG ;

\- confirmer les entrées.



\---



\## Gestion des filtres de marché



Le moteur inclut différents filtres permettant de renforcer la robustesse des signaux :



\- displacement filters ;

\- opening range filters ;

\- session filters ;

\- momentum filters ;

\- trend filters ;

\- confirmations multi-timeframe ;

\- filtres temporels.



\---



\## Optimisation paramétrique



Le projet inclut un moteur d’optimisation permettant de tester automatiquement de nombreuses configurations :



\- risk/reward ;

\- taille minimale des IFVG ;

\- âge maximal des structures ;

\- paramètres de trailing ;

\- paramètres de break-even ;

\- fenêtres horaires ;

\- retracement percentage ;

\- paramètres de risque.



Des scripts d’itération successifs ont été développés afin d’améliorer progressivement :



\- la robustesse mensuelle ;

\- la stabilité du win rate ;

\- les drawdowns ;

\- la cohérence inter-annuelle.



\---



\## Analyse statistique avancée



Le framework inclut des outils d’analyse quantitative :



\- cohort analysis ;

\- monthly consistency analysis ;

\- yearly performance analysis ;

\- streak analysis ;

\- drawdown analysis ;

\- expectancy ;

\- profit factor ;

\- Kelly fraction ;

\- runs test ;

\- analyse de volatilité des performances.



\---



\# Technologies utilisées



\## Langage



\- Python



\## Bibliothèques



\- Pandas

\- NumPy

\- SciPy

\- Streamlit

\- Plotly

\- Matplotlib



\---



\# Architecture du projet



```text

mnq-ifvg-strategy-backtester/

│

├── README.md

├── requirements.txt

├── screenshots/

└── results/

```



\---



\# Dashboard



\## Dashboard principal



!\[Dashboard](screenshots/dashboard\_overview.png)



\---



\## Equity Curve



!\[Equity Curve](screenshots/equity\_curve.png)



\---



\## Analyse mensuelle



!\[Monthly Performance](screenshots/monthly\_performance.png)



\---



\## Analyse des drawdowns



!\[Drawdown Analysis](screenshots/drawdown\_analysis.png)



\---



\## Optimisation paramétrique



!\[Optimization](screenshots/parameter\_optimization.png)



\---



\## Analyse statistique



!\[Statistical Analysis](screenshots/statistical\_analysis.png)



\---



\## Analyse événements macroéconomiques



!\[Macro Events](screenshots/macro\_event\_analysis.png)



\---



\# Résultats



Le framework permet :



\- d’évaluer quantitativement une stratégie algorithmique sur plusieurs années de données ;

\- de mesurer la stabilité mensuelle des performances ;

\- d’identifier les périodes de faiblesse ;

\- d’optimiser les paramètres de gestion du risque ;

\- d’analyser la robustesse statistique du système.



Les différentes itérations de recherche ont permis d’améliorer progressivement :



\- la stabilité des performances ;

\- la cohérence mensuelle ;

\- les drawdowns ;

\- la qualité des signaux ;

\- la régularité des résultats.



\---



\# Confidentialité de la stratégie



Le repository public présente principalement :



\- l’architecture du framework ;

\- les outils de recherche ;

\- les composants analytiques ;

\- les visualisations ;

\- les résultats statistiques.



Certaines parties liées à la logique exacte de génération des signaux et à des éléments propriétaires de la stratégie ne sont volontairement pas publiées.



L’objectif du projet est avant tout de présenter les compétences en :



\- développement Python ;

\- architecture de backtesting ;

\- analyse quantitative ;

\- optimisation ;

\- recherche algorithmique.



\---



\# Installation



Cloner le repository :



```bash

git clone https://github.com/antoineborie/mnq-ifvg-strategy-backtester.git

```



Installer les dépendances :



```bash

pip install -r requirements.txt

```



\---



\# Auteur



Antoine Borie



Étudiant ingénieur spécialisé en cybersécurité, intelligence artificielle et finance quantitative.



\- LinkedIn : www.linkedin.com/in/antoine-borie

\- Email : borie@et.esiea.fr

