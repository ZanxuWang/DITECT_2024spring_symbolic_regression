import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
from torch.functional import F

# Define the feature network with improved handling of ExU units and Batch Normalization
class ExUNet(nn.Module):
    def __init__(self, input_dim, num_layers, num_units):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, num_units) if i == 0 else nn.Linear(num_units, num_units) for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_units) for _ in range(num_layers)])
        self.out_layer = nn.Linear(num_units, 1)

    def forward(self, x):
        for layer, bn in zip(self.layers, self.bns):
            x = torch.relu(bn(layer(x)))  # Using ReLU for demonstration, replace with appropriate ExU function
        return self.out_layer(x)

# Define the overall symbolic regression model
class SymbolicRegressionModel(nn.Module):
    def __init__(self, num_features, num_layers, num_units):
        super().__init__()
        self.feature_nets = nn.ModuleList([ExUNet(1, num_layers, num_units) for _ in range(num_features)])
    
    def forward(self, x):
        feature_outputs = [net(x[:, i:i+1]) for i, net in enumerate(self.feature_nets)]
        additive_output = torch.sum(torch.stack(feature_outputs), dim=0)
        return additive_output

# Utilize sympy for symbolic expressions handling
def generate_expression(feature_nets, min_length, max_length):
    ops = ['+', '-', '*', '/', '**', 'sqrt', 'exp', 'log', 'abs', 'sin', 'cos', 'tan', 'tanh', 'sigmoid', 'sgn', 'asin', 'atan', 'atanh', 'cosh', 'gaussian']
    variable = sp.symbols('x0')
    expr_len = np.random.randint(min_length, max_length + 1)
    expression = sp.sympify('0')
    
    for _ in range(expr_len):
        expr1 = np.random.choice([variable, sp.Float(np.random.randn())])
        expr2 = np.random.choice([variable, sp.Float(np.random.randn())])
        op = np.random.choice(ops)

        # Apply constraints
        if all(isinstance(e, sp.Float) for e in [expr1, expr2]) and op in ['+', '-', '*', '/']:
            continue  # Skip if both children are constants
        
        if op in ['log', 'exp', 'sqrt'] and (isinstance(expr1, sp.Float) and expr1 <= 0):
            continue  # Skip invalid domains for log, exp, sqrt
        
        if op in ['sin', 'cos', 'tan', 'asin', 'atan', 'atanh', 'cosh'] and isinstance(expr1, sp.Function):
            if expr1.func in [sp.sin, sp.cos, sp.tan, sp.asin, sp.atan, sp.atanh, sp.cosh]:
                continue  # Skip nested trigonometric operators
        
        if op == '/' and isinstance(expr2, sp.Float) and expr2 == 0:
            continue  # Skip division by zero

        if op == '**' and isinstance(expr1, sp.Float) and isinstance(expr2, sp.Float) and expr1 < 0 and not expr2.is_integer:
            continue  # Skip invalid powers

        if op in ['+', '-', '*', '/']:
            expression += sp.sympify(f'{expr1} {op} {expr2}')
        elif op == '**':
            expression += expr1**expr2
        elif op == 'gaussian':
            expression += sp.exp(-expr1**2)
        else:
            expression += sp.Function(op)(expr1)
    
    return sp.simplify(expression)

# Custom functions
def gaussian(x):
    return np.exp(-x**2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Convert sympy expression to torch expression
def sympy_to_torch(expression, feature_nets, X):
    expr = sp.lambdify(['x0'], expression, modules=['numpy', {'gaussian': gaussian, 'sigmoid': sigmoid}])
    return torch.tensor(expr(X[:, 0].numpy()), dtype=torch.float32)

# Training function with risk-seeking policy gradient
def train(model, X, y, num_epochs, batch_size, learning_rate, ensemble_size):
    ensemble = [SymbolicRegressionModel(X.shape[1], num_layers, num_units) for _ in range(ensemble_size)]
    optimizers = [optim.Adam(m.parameters(), lr=learning_rate) for m in ensemble]
    
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for batch_x, batch_y in data_loader:
            for i, (model, optimizer) in enumerate(zip(ensemble, optimizers)):
                optimizer.zero_grad()
                output = model(batch_x).squeeze()  # Ensure the output shape matches y's shape
                loss = F.mse_loss(output, batch_y)
                
                # Apply risk-seeking adjustment
                risk_loss = -torch.log(loss + 1e-8)  # Add a small value to prevent log(0)
                risk_loss.backward()
                optimizer.step()

                # Diagnostic print statement to ensure that parameters are being updated
                with torch.no_grad():
                    if epoch == 0 and len(data_loader) == 0:
                        initial_params = [p.clone() for p in model.parameters()]
                    elif epoch == num_epochs - 1 and len(data_loader) == 0:
                        final_params = [p.clone() for p in model.parameters()]
                        if all(torch.equal(i, f) for i, f in zip(initial_params, final_params)):
                            print(f"Model {i+1} parameters not updated.")

                print(f'Epoch {epoch+1}/{num_epochs}, Model {i+1}/{ensemble_size}, Loss: {loss.item()}, Risk-Seeking Loss: {risk_loss.item()}')

    return ensemble

# Additional settings and training call
torch.manual_seed(0)
np.random.seed(0)

X = torch.randn(1000, 1)  # Ensure X has only one variable
y = torch.sin(X[:, 0])  # Ensure y is a 1D tensor

num_layers = 5
num_units = 64
num_epochs = 10
batch_size = 64
learning_rate = 0.0001
ensemble_size = 5

model = SymbolicRegressionModel(X.shape[1], num_layers, num_units)
ensemble = train(model, X, y, num_epochs, batch_size, learning_rate, ensemble_size)

# Generate and print symbolic expression for demonstration
symbolic_expr = generate_expression(ensemble[0].feature_nets, min_length=4, max_length=30)
print(f"Generated symbolic expression: {symbolic_expr}")

# Convert symbolic expression to torch expression and print for demonstration
#torch_expr = sympy_to_torch(symbolic_expr, ensemble[0].feature_nets, X)
#print(f"Converted torch expression: {torch_expr}")

# Ensure valid values in torch expression
#print(f"Converted torch expression (valid values only): {torch_expr[torch.isfinite(torch_expr)]}")







