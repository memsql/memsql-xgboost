# Machine Learning via MemSQL

This repository contains examples of how machine learning models can be evaluated in a MemSQL database. Examples are contained in the [examples](/examples) folder in the form of Jupyter notebooks.

## Contents

1. Training an XGBoost model on a local host, deploying and evaluating it on a MemSQL instance.
2. Training an XGBoost model using an [AWS Sagemaker](https://aws.amazon.com/sagemaker/), deploying and evaluating it on a MemSQL instance.
3. Evaluating an XGBoost model in a MemSQL instance with data that is streamed from Kafka.
4. Evaluating an XGBoost model in a MemSQL instance on a TPC-H 50GB dataset.
5. Implementing a recommendation engine in a MemSQL instance based on Matrix factorization. This method is a very popular implementation used by Spotify, Netflix, etc.
6. Using MemSQL user-defined functions to evaluate cascaded models on the fly. This allows models to be applied sequentially, from simple to complex, while narrowing the dataset with each step.   

## Setup

Install the dependencies (Ubuntu/Debian). These steps are similar on non-dpkg-based systems.

```
apt install python3-pip python3-dev libmariadbclient-dev
```

Create and activate a virtual environment

```
pip3 install --upgrade pip && pip3 install virtualenv
virtualenv -p $(which python3) venv
source venv/bin/activate
```

Install all Python requirements

```
pip3 install -r requirements.txt
```

Run Jupyter

```
jupyter notebook
```

## Obtain a MemSQL instance

If you don't yet have a MemSQL instance, you can try one for free at [memsql.com/helios](https://bit.ly/3jAKBUK):
1. Sign up, verify your email, and log in   
   ![signup](imgs/helios-1.png)
2. Click "Create Managed Cluster"   
   ![signup](imgs/helios-2.png)
3. Give your cluster a name and select AI/ML   
   ![signup](imgs/helios-3.png)
4. Select free trial   
   ![signup](imgs/helios-4.png)
5. Provide an admin password, select "Allow access from anywhere" (to simplify testing), and click "Create Cluster"   
   ![signup](imgs/helios-5.png)
6. Wait for the cluster to be created and then use the admin endpoint to connect to it   
   ![signup](imgs/helios-6.png)
