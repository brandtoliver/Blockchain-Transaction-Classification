# Blockchain-Transaction-Classification

Two-sided Bitcoin Classification project:

* Gaussian Mixture Models on Bitcoin Blockchain user record transaction data. Data can be downloaded from: 
[User data](https://drive.google.com/file/d/1CNsVfor7k1NqpMb1Abq_aGxI9RzVhcp5/view?usp=sharing) 

* Latent Space Models and Stochastic Block Models on Bitcoin Blockchain network data. Data can be downloaded from: 
[Network data](https://drive.google.com/file/d/1k0_gx5ehk4ZXxLQAiKF1o2TMmJoq_Q4v/view?usp=sharing) 

## Our models:
* [Gaussian Mixture Models](GMM.ipynb)
* [Latent Space Models](LSM.ipynb)
* [Stochastic Block Models](SBM.ipynb)

## Project abstract
The anonymity properties of the bitcoin blockchain have recently been a popular topic of discussion, especially in relation to money laundering. Given these properties, a question can be posed about whether it is possible to extract meaningful insights from raw transaction data. Using Bayesian machine learning techniques, a framework is proposed in order to analyse the flow of Bitcoins, cluster the users of Bitcoin, and detect abnormal behaviour of these users.

The flow of Bitcoins is analysed by predicting future transactions based solely on the current flow. It is regarded as a network problem, and modelled with a Latent Space Model and a Stochastic Block Model. Both models proves to capture the network structures well, though the Stochastic Block Model turned out to be the most correct model, since it captured group structures assumed to exist between users. The Stochastic Block Model was, with high precision, able to predict missing or future transactions between the most active users.

A Gaussian Mixture Model was utilised on features extracted from Bitcoin transactions. It was applied to cluster users on the bitcoin blockchain, and detect users with abnormal behaviour. With held out data, abnormal users are successfully identified and described.

Given this framework, it was possible to extract meaningful information from a subset of the bitcoin blockchain. Using the Edward library, models was implemented in the scalable architecture of TensorFlow, though a sufficient level of computational power was required to scale-up the entire bitcoin blockchain.

## Contributors
* Martin Johnsen ([mar10nJohns1](https://github.com/mar10nJohns1))
* Andreas Jensen ([MotzWanted](https://github.com/MotzWanted))
* Oliver Brandt ([brandtoliver](https://github.com/brandtoliver))

## Full proejct report
[Identifying blockchain transactions using unsupervised machine learning](Project_report.pdf)
