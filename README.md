# Iris Flower Species Classification 🌸

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)
![scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A simple web application that predicts the species of an Iris flower (Setosa, Versicolor, or Virginica) based on its petal length and width using a K-Nearest Neighbors (k-NN) machine learning model.

## Project Description

This project implements a machine learning model to solve the classic Iris flower classification problem. The model is trained on the well-known Iris dataset to classify flowers into one of three species. The core of the project is a K-Nearest Neighbors (k-NN) classifier built with Scikit-learn, which is deployed as a web application using the Flask framework. Users can input the petal length and width through a web interface to get an instant prediction of the flower's species.

## Features

-   **Interactive Web Interface:** Easy-to-use web UI to get predictions.
-   **Real-time Prediction:** Instantly classifies the Iris species based on user input.
-   **Pre-trained Model:** Uses a saved (pickled) model (`specie.pkl`) for fast predictions without needing to retrain.
-   **Clean Architecture:** Separation of the ML model logic (`ml.py`) from the web application (`app.py`).

## Technologies Used

-   **Backend:** Python
-   **Web Framework:** Flask
-   **ML Library:** Scikit-learn
-   **Data Manipulation:** Pandas, NumPy
-   **Frontend:** HTML, CSS

## Project Structure

<img width="598" height="475" alt="image" src="https://github.com/user-attachments/assets/7911732a-0176-4ed1-9412-2217ce8a9931" />
