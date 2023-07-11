# Machine learning for physics project
Project for Metodi di Apprendimento Automatico  (machine learning methods for physics) exam.
This projects consist in the application of different machine learning models on a dataset consisting of Cern's CMS experiment (simulated) data. The goal is to discriminate between signal and background for the Higgs to anti-b b process, where the two b quarks produce jets which can be fat or narrow jets, based on their radious.

The techniques used are: 
1. Decision Trees, Random Forests, Boosted Decision Trees
2. Linear Regression
3. KNN
4. Support Vector Machines

The actual optimisation has been done only for the BDT, since it is also my BSc thesis and I spent more time on it. There is also a repository for that with more details. Here, the purpose was just to see how those algorithms compare and try to build an undesstanding of how they work.

## Config.py file
You may have noticed the config.py file. It contains a dictionary with all the variables for different analysis configurations. 
The reason is because, as mentioned above, there can be both fat and narrow jets, and we may have situations where we have 2 narrow or 1 fat jets. The former is called resolved configuration, the latter is called resolved.
Based on some physics considerations (I am happy to discuss it via mail!), you may want to select different configurations (hence differnet kinematic variables) but also different energy tresholds.
Basicaly in every ML algorithm you will select a configuration and an energy regime in the config dictionary.
