# DITECT_2024spring_symbolic_regression

# Symbolic Regression Project for DITECT 2024 Spring Research Course

## Summary
This project explores and integrates advanced symbolic regression techniques to develop a robust symbolic regression model. After a comprehensive literature review on existing symbolic regression techniques, we proposed a model that combines ideas from "Deep Symbolic Regression" (DSR) and "Neural Additive Models" (NAMs) to reduce the search space while maintaining the differentiable characteristics of DSR methods. We also experimented the performance of a recently published KAN network on symbolic regression.

## Literature Review
1. **Literature Search**: Conducted a review of 20 papers on symbolic regression techniques and organized the findings into an easily readable Excel table.
   - Literature search SR.xlsx

2. **Deep Symbolic Regression (DSR)**: Explored the paper "Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients" in detail and tested the provided code locally.

3. **Neural Additive Models (NAMs)**: Investigated the paper "Neural Additive Models: Interpretable Machine Learning with Neural Nets" and tested the code locally.

4. **Integration Proposal**: Proposed a method to integrate DSR and NAM models to reduce search space while maintaining the differentiable characteristics of DSR methods. A mini-version of the sample implementation can be found in SR_model.py.

5. **Kolmogorov–Arnold Networks (KAN)**: Explored and tested the recent paper on KAN using the DSR Nguyen dataset and generated datasets.
   - KAN_dso_data.ipynb
   - To test the KAN notebook, follow the setup guide here: https://github.com/KindXiaoming/pykan/blob/master/README.md

## Key Ideas
### Deep Symbolic Regression (DSR)
- Uses a recurrent neural network (RNN) to generate a distribution over tractable mathematical expressions.
- Employs a risk-seeking policy gradient to train the RNN to generate better-fitting expressions.
- Represents expressions as pre-order traversals of symbolic expression trees.
- Applies constraints during autoregressive sampling to ensure only valid expressions are generated.
- Optimizes constants in expressions to further improve the fit.

### Neural Additive Models (NAMs)
- Learns a linear combination of neural networks that each attend to a single input feature.
- Neural nets are trained jointly and can learn arbitrarily complex relationships between their input feature and output.
- Employs specialized Exp-Centered (ExU) hidden units to allow learning jumpy/jagged functions.
- Uses regularization techniques to avoid overfitting the jagged ExU units.
- Utilizes ensembling to improve accuracy and provide uncertainty estimates.

## Integration of DSR and NAMs
1. **Structure of Expressions**: Use the DSR approach of generating expressions as pre-order traversals of symbolic expression trees.
2. **Feature Networks**: Replace the single RNN in DSR with feature-wise neural networks as in NAMs.
3. **Hidden Units**: Use ExU hidden units from NAMs in the feature networks.
4. **Regularization**: Apply regularization techniques from NAMs like dropout.
5. **Constant Optimization**: Optimize constants in the generated expressions, as done in DSR.
6. **Risk-Seeking Training**: Train the overall model using the risk-seeking policy gradient from DSR.
7. **Ensembling**: Maintain an ensemble of models, as in NAMs, to improve accuracy and quantify uncertainty.

## SR_mody.py: Implementation Overview
The `SR_mody.py` script implements a symbolic regression model based on the above key points. It uses neural networks with Batch Normalization and ExU units for improved performance. The `ExUNet` class defines a feature network with specified layers and units, applying Batch Normalization and ReLU activation. The `SymbolicRegressionModel` class uses these feature networks to combine their outputs additively. The `generate_expression` function creates symbolic expressions with constraints to ensure valid mathematical operations. The `sympy_to_torch` function converts symbolic expressions to PyTorch tensors for evaluation. The `train` function trains an ensemble of models using mini-batch gradient descent with a risk-seeking policy gradient, adjusting the loss to focus on finding high-quality fits. The script initializes data, sets hyperparameters, trains the model, and generates a symbolic expression for demonstration.

## Observations on Testing KAN Model
Applying Kolmogorov-Arnold Networks (KAN) to symbolic regression is challenging and requires extremely high precision. The process involves multiple pruning and fixing of activation functions to fit the exact expression. The datasets used in KAN's code, often containing only 20 data points per CSV, struggle to fit simple expressions while overfitting to complex ones. Modifying KAN's structure can lead to different fitting results. Using the `create_dataset` default setting with 1000 data points shows that KAN is powerful, achieving test losses as low as e-2 to e-4 with minimal effort. However, pruning and fitting activation functions are crucial, as KAN provides a list of functions ordered by R square, with many functions having R square values close to 1. If the correct function is not selected, the final expression will be incorrect. The process involves multiple iterations of training, fixing, and retraining, as symbolic regression requires extremely low test loss (around e-9) for exact expressions. This sensitivity is inherent to symbolic regression, not just KAN. Moreover, the addition of layers or neurons significantly slows down the operation. Symbolic regression is inherently noise-sensitive, necessitating high accuracy in trained networks for extracting exact symbolic formulas. For a detailed tutorial, refer to https://github.com/KindXiaoming/pykan/tree/master/tutorials.
