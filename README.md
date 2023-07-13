# OptuMNIST

75% to 85% accuracy on MNIST with 355 parameters.
Can **you** do better?

If so, contribute!

## What is this?

We aim to better understand the Pareto front of accuracy and number of parameters for MNIST.

This may prove useful to help understand hyperparameter tuning for frugal models.

Do you have ideas for improvements?

## How to use

Install torch, then torch-uncertainty with

```bash
pip install torch-uncertainty
```

Then run the model with

```bash
python optumnist/optumnist-v1.py
```

The dataset will be downloaded automatically.

## Reference

OptuMNIST-v1 has been found with Optuna:

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama. Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD 2019.
