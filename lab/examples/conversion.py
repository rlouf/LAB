""" Tutorial on the conversion models.
"""
import numpy as np

import lab
import pymc3 as pm


# Models

## Conversion probabilities
theta1 = 0.34
theta2 = 0.36
theta3 = 0.45

## Trials
trials_1 = 1101
trials_2 = 876
trials_3 = 1342

## Success
success_1 = np.sum(np.random.binomial(trials_1, theta1) == 0)
success_2 = np.sum(np.random.binomial(trials_2, theta1) == 0)
success_3 = np.sum(np.random.binomial(trials_3, theta1) == 0)


data = np.array([[trials1, success_1], [trials_2, success_2], [trials_3, success_3]])


# Model
model = lab.Conversion()
