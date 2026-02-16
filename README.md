# ToxiQ
ToxiQ is a hybrid Quantum Machine Learning pipeline for early drug toxicity prediction. The system dynamically adjusts the number of qubits based on selected molecular features and returns toxicity probability results through an interactive interface.
ToxiQ is an end-to-end Quantum Machine Learning system designed to explore early drug toxicity prediction using a hybrid classical–quantum pipeline. The project integrates an R Shiny interface for user interaction, data preprocessing, and feature selection with a Python-based variational quantum classifier for model training and inference.

Users can upload a dataset and select the number of features to use. The R pipeline automatically performs feature selection, creates training/testing splits, and exports structured data to the Python engine. The quantum model then dynamically adjusts the number of qubits based on the selected features, learns patterns between toxic and non-toxic samples, and outputs probability-based predictions that are displayed back in the interface.

This project demonstrates how quantum machine learning can be integrated into a modular, real-world workflow using a clean R–Python bridge through CSV-based communication.
