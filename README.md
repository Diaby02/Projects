# Projects
This repo contains all the projects I have made during my studies (not necessarily linked to computer science).
Some projects related to mechanics, electronic and computer science were conducted during my bachelor degree, but the majority have been done
during my Master. Some code can be made available on demand.

The differents reports have been placed in three different folders:

## Bachelor

This folder contains some projects I have made during my frist three years (but not all of them). Most of them are written in french.

* [Mutex and semaphors ](https://github.com/Diaby02/Projects/blob/main/Bachelor/MutexAndSemaphorsProject.pdf)

This project compares the execution of multithreaded programs using POSIX semaphores and active waiting mechanisms like Test-And-Set and Test-And-Test-And-Set. It implements three classic concurrency problems: the producer-consumer, dining philosophers, and readers-writers problems. Performance measurements were conducted by varying the number of threads from 2 to 64 on a 32-core machine, showing that POSIX semaphores generally outperform active waiting mechanisms, especially as the number of threads increases.

* [C_project](https://github.com/Diaby02/Projects/blob/main/Bachelor/C_project.pdf)

This project involved translating a Python program that uses error-correcting codes into the C language and optimizing it through multithreading. The goal was to efficiently recover lost source symbols in binary file transmissions using the RLC method and to run the program on a Raspberry Pi, leveraging its four cores. The project included implementing a sequential C version, developing a multithreaded version for performance improvement, and conducting various tests to ensure functionality and efficiency.

* [Music Oz](https://github.com/Diaby02/Projects/blob/main/Bachelor/MusicUsingOZlanguage.pdf)

This project, titled "Maestroz," involves creating music using the OZ programming language and addresses known issues with handling floating-point numbers in OZ. The project includes the implementation of two extensions: smoothing to enhance sound quality by reducing noise between notes, and instruments to add a creative dimension to music composition. The implementation choices and non-declarative constructions were adjusted to meet the requirements of the Inginious platform, improving code efficiency at the cost of readability.

* [Finite Elements](https://github.com/Diaby02/Projects/blob/main/Bachelor/FiniteElementsProject.pdf)

This project focuses on studying the linear elasticity of a climbing carabiner in both open and closed positions using the finite element method. The goal is to understand how the carabiner responds to traction forces, which is crucial for ensuring the safety of climbers. The study involves modeling the mechanical behavior of the carabiner under different load conditions and using optimization techniques to ensure reasonable computation times and sufficient accuracy.

* [Morse Communication](https://github.com/Diaby02/Projects/blob/main/Bachelor/FiniteElementsProject.pdf)

This project involves developing a Morse code communication system using ultrasonic waves. The system consists of a transmitter and a receiver. The transmitter converts user inputs (dots, dashes, and spaces) into ultrasonic signals of specific durations, while the receiver processes these signals and converts them into visual outputs using LEDs. The project explores the design and implementation of various electronic circuits, including timers, oscillators, amplifiers, filters, and converters, to achieve reliable ultrasonic communication. Despite some challenges in the final implementation, the project provides valuable insights into electronic circuit design and the complexities of real-world applications.

* [Packets Analysis](https://github.com/Diaby02/Projects/blob/main/Bachelor/TeamsPacketsAnalysis.pdf)

This document presents an analysis of the Microsoft Teams application, focusing on its network communication protocols and security measures. The study examines DNS queries, network and transport layers, encryption techniques, and the application's behavior under various usage scenarios, such as text messaging, audio and video calls, and file sharing. The analysis reveals that Microsoft Teams primarily uses TCP for reliable data transmission and UDP for real-time audio and video communication. It also highlights the use of various security protocols like TLS and the involvement of multiple servers and domains managed by Microsoft and other service providers. The findings suggest that Microsoft Teams operates predominantly through cloud-based services, ensuring efficient and secure communication.

## Master 1

This folder contains all the projects I have made during my first year of Master

* [Data Visualization](https://github.com/Diaby02/Projects/blob/main/Master1/DataVisualization_Project.pdf)

This project report details a visualization dashboard for exploring celebrity image datasets. It features customizable visualizations using PCA and t-SNE for dimensionality reduction, and clustering methods like K-Means and DBSCAN. Users can interact with plots, toggle datasets, and find celebrity look-alikes based on selected attributes, offering a comprehensive tool for data exploration and visualization.

* [Machine Learning 1](https://github.com/Diaby02/Projects/blob/main/Master1/MachineLearning_LELEC2870_Project.pdf)

This project focuses on predicting heart disease risk using machine learning models. It involves data preprocessing, feature selection, and model tuning with techniques like K-Nearest Neighbors, Multilayer Perceptron, and Random Forest. The K-Nearest Neighbors model performed best, emphasizing the importance of accurate risk prediction in a medical context. The analysis highlights key features influencing heart disease risk.

* [Machine Learning_2](https://github.com/Diaby02/Projects/blob/main/Master1/MachineLearning_LSTAT2120_Project.pdf)

This project focuses on predicting car prices using linear regression techniques. It involves data cleaning, analysis, and model selection to understand how various car features influence market value. The dataset includes 200 cars described by 25 features, both quantitative and qualitative. The project addresses data preprocessing, feature selection, and the application of linear regression to model relationships between car specifications and prices. The final model aims to provide insights into significant predictors of car valuation.

* [Deep Learning](https://github.com/Diaby02/Projects/blob/main/Master1/DeepLearning_LELEC2885_Projet.pdf)

This project focuses on image segmentation using a modified UNet architecture, optimized for a smaller dataset. The study compares a standard UNet with a smaller version, showing that the latter performs better in terms of accuracy and computational efficiency. The project includes training with Binary Cross-Entropy loss, data augmentation, and early stopping to prevent overfitting, achieving high accuracy and IoU scores on test datasets.

* [Network Algorithms](https://github.com/Diaby02/Projects/blob/main/Master1/NetworkAlgorithms_Project.pdf)

This project analyzes the network of interactions from "Star Wars IV" using graph theory techniques. It explores degree assortativity, community detection with the Louvain algorithm, and spectral clustering. The study also simulates information propagation using the Independent Cascade Model and compares the network to a Barabasi-Albert model. The analysis highlights key characters and community structures within the movie's narrative.

* [Generative Adversarial Network](https://github.com/Diaby02/Projects/blob/main/Master1/GenerativeAdversarialNetwork_Project.pdf)

This project focuses on improving a Conditional Generative Adversarial Network (CGAN) for generating MNIST-like images. It explores architectural enhancements, training optimizations, and evaluations using metrics like the Inception Score. The study also delves into the latent space exploration, examining the diversity and discrimination capabilities of the generated images, and discusses potential improvements and insights gained from the experiments.

* [Kernel PCA](https://github.com/Diaby02/Projects/blob/main/Master1/KernelPCA_Project.pdf)

This project explores the Weisfeiler-Lehman subtree kernel for graph analysis, focusing on its computational complexity and explicit feature mapping. It compares kernel-based methods like Kernel Principal Component Analysis (KPCA) and t-SNE for visualization, and evaluates classification performance using Support Vector Machines (SVM) on datasets such as MUTAG, ENZYMES, and NCI1, highlighting the effectiveness of kernel methods in graph-based machine learning tasks.

* [Kernel SVM](https://github.com/Diaby02/Projects/blob/main/Master1/KernelSVM_Project.pdf)

This project compares the performance and testing times of different SVM methods: Linear SVM, Kernel SVM with Gaussian Kernel, and Linear SVM with Random Fourier Features (RFF). It explores how varying the dimensionality of RFF impacts accuracy and computational efficiency, highlighting the trade-offs between dimensionality and performance in machine learning models.

* [Machine Learning 3](https://github.com/Diaby02/Projects/blob/main/Master1/MachineLearning_LINFO2262_Project.pdf)

Machine Learning competition. Ranking: #2 out of 200 students.
However, code and full report can't be published, according to the willing of the professor in charge.

* [Reinforcement Learning](https://github.com/Diaby02/Projects/blob/main/Master1/ReinforcementLearning_Project.pdf)

The project explores the application of Markov Decision Processes and reinforcement learning techniques to optimize strategies in two games: Snakes and Ladders and Connect Four. For Snakes and Ladders, the focus is on minimizing costs to reach the goal using value iteration. In Connect Four, the goal shifts to maximizing rewards, employing Q-learning and Deep Q-Learning to improve agent performance against various opponents.

* [Mining Pattern 1](https://github.com/Diaby02/Projects/blob/main/Master1/MiningPattern_Project1.pdf)

This project compares algorithms for frequent itemset mining, specifically three variations of the Apriori algorithm (Dummy, Smart, and Trie-based) and the ECLAT algorithm. The study evaluates their performance on different datasets, analyzing execution time and memory usage. ECLAT generally outperforms Apriori variants, although the Trie-based Apriori shows advantages in specific scenarios, particularly with datasets having fewer frequent itemsets. The choice of algorithm depends on dataset characteristics and resource constraints.

* [Mining Pattern 2](https://github.com/Diaby02/Projects/blob/main/Master1/MiningPattern_Project2.pdf)

This project focuses on implementing and analyzing the SPADE algorithm for frequent sequence mining and its application in classification tasks. It explores both the most frequent sequences and sequences with high WRACC scores for classification. The study evaluates these methods using cross-validation and compares their performance on different datasets, highlighting the effectiveness of sequence mining techniques in capturing sequential data patterns for classification purposes.

* [Mining Pattern 3](https://github.com/Diaby02/Projects/blob/main/Master1/MiningPattern_Project3.pdf)

This project explores the use of Bayesian networks for imputing missing values in tabular datasets. It covers Bayesian inference, parameter learning with Laplace smoothing, and structure learning using scoring functions like K2, AIC, and BIC. The study evaluates the effectiveness of these methods across different datasets, demonstrating high accuracy in datasets with many variables but highlighting challenges with fewer variables or complex missing patterns.

* [Bayesian Statistics](https://github.com/Diaby02/Projects/blob/main/Master1/BayesianStatistics_Project.pdf)

This project explores Bayesian statistical methods for analyzing rent prices using a Gamma regression model. It involves deriving the likelihood function, implementing log-likelihood and log-posterior computations in R, and estimating posterior distributions using Metropolis algorithms. The study evaluates model parameters, convergence, and credible intervals, and compares models with different covariates. Additionally, it employs JAGS for more sophisticated regression modeling and assesses predictive distributions for rent prices under various conditions.

## Master 2

in Canada

*[Combinatorial Optimization 1 (FR)](link_to_file)

The document presents solutions to combinatorial optimization problems. The first problem involves aligning four colored cubes to form a prism with distinct colors on each face. The second problem focuses on optimizing employee schedules to minimize costs while meeting demand and constraints, using solvers like OR-Tools and Highs.

*[Combinatorial Optimization Research (FR)](link_to_file)
*[HCM Research 5 (FR)](link_to_file)

This document explores the advantages and disadvantages of gamification in user interfaces. It highlights how gamification can enhance motivation, learning, and user engagement across various fields such as health, education, and business. However, it also points out significant challenges, including a lack of longitudinal studies, ethical concerns, and difficulties in standardization. The paper critically examines these aspects and suggests pathways for better integration of gamification in user interface design.

*[NLP Project 1 (FR)](link_to_file)

The documents describe three distinct tasks related to natural language processing and data extraction. The first task involves extracting information from recipes, likely focusing on ingredients and instructions. The second task is about correcting proverbs, which involves identifying and fixing errors in traditional sayings. The third task pertains to the automatic classification of incident descriptions, which involves categorizing textual descriptions of incidents into predefined classes.

*[NLP Project 2 (FR)](link_to_file)

This project aims to correct distorted French proverbs by replacing incorrect verbs with suitable ones from a given list. It uses two transformer models: one for POS tagging to identify the verb, and another for masked language modeling to select the best replacement. The approach avoids large language models and relies on pre-trained Hugging Face models like CamemBERT. Performance is evaluated quantitatively and qualitatively on a test dataset, with improvements through post-processing and verb tense alignment.

*[NLP Project 3 (FR)](link_to_file)

This project compares the summarization performance of eight lightweight LLMs on Enron emails using various prompt strategies. It evaluates BLEU, ROUGE, and BERTScore metrics against reference models (Pegasus, BART). Results show that RedPajama and Gemma consistently outperform others, especially with well-designed prompts. The study also analyzes inference time and highlights the impact of prompt engineering on model accuracy and efficiency.


in Belgium
*[Master thesis](link_to_repo)

