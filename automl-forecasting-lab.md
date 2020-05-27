# MLADS 2020 LAB: Time-series Forecasting with Automated Machine Learning
The content below is a guide for a self-paced lab to understand time-series forecasting with Automated Machine Learning both through the Python code experience and UI no-code experience. We will walk through common forecasting parameters and considerations when working with time-series data.

# Key Words
- Time-series, Forecasting, Hyperparameter tuning, Python, JupyerLab, No-code UI

# Table of Contents
1. [Prerequisites](#prereqs)
1. [Automated ML Introduction](#introduction)
1. [The studio Introduction](#studio)
1. [Setup a Compute Instances](#compute)
1. [Train in JupyertLab](#train)
1. [Documentation](#documentation)
1. [Running using python command](#pythoncommand)
1. [Troubleshooting](#troubleshooting)

<a name="prereqs"></a>
# Prerequisites
In order to successfully follow along with this lab, let's ensure you have an Azure Machine Learning workspace created.
* [Create a workspace](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup#create-a-workspace)

![image](./createworkspace.gif)

# Lab agenda
* Introduction to automated ML
* The studio and model authoring experiences
* Creating a compute instance
* Running a python code automated ML experiment
* Understanding forecasting parameters and concepts
* Train, evaluate, and deploy your model
* Easily perform these steps again in a no-code experience

<a name="introduction"></a>
## Automated ML introduction
Automated machine learning (automated ML) builds high quality machine learning models for you by automating model and hyperparameter selection. Bring a labelled dataset that you want to build a model for, automated ML will give you a high quality machine learning model that you can use for predictions.

If you are new to Data Science, automated ML will help you get jumpstarted by simplifying machine learning model building. It abstracts you from needing to perform model selection, hyperparameter selection, or feature engineering and in one step creates a high quality trained model for you to use.

![image](./diagram.png)

If you are an experienced data scientist, automated ML will help increase your productivity by intelligently performing the model and hyperparameter selection for your training and generates high quality models much quicker than manually specifying several combinations of the parameters and running training jobs. Automated ML provides visibility and access to all the training jobs and the performance characteristics of the models to help you further tune the pipeline if you desire.

<a name="studio"></a>
## The studio and automated ML authoring experience
Azure ML has a suite of authoring experiences in a user-interface format in [the studio](https://ml.azure.com). 

1. Visit [https://ml.azure.com](https://ml.azure.com)
2. Select your `directory`, `subscription`, and `workspace`
![image](./portalsignin.PNG)

Automated ML is supported by multiple execution environments including:
* Compute Instances
* Local Conda environment
* Azure Databricks

<a name="compute"></a>
## Setup using Compute Instances
In this lab we will be focusing on the Compute Instance environment.

1. On the left hand panel under the `Manage` section, click on `Compute`
![image](./managecompute.PNG)
2. Click `+ New` to create a new compute. *It will take a couple minutes for this compute to to spin-up.*
![image](./createcompute.gif)
3. When the compute `Status` is `Running`, select the `JupyertLab` Application URI.

<a name="compute"></a>
## Run the lab on JupyterLab
In this part of the lab we will be covering a python code 
