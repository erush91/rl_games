import torch

def classify_equilibrium(eigenvalues):
    """ Classify equilibrium point.
    
    Given the eigenvalues linearized around an equilibrium point,
    classify the type of equilibrium
    """
    real = torch.abs(eigenvalues)

    if torch.all(real < 1.0):
        equilibrium = "attractor"
    elif torch.all(real > 1.0):
        equilibrium = "repellor"
    else:
        n_unstable = len(torch.where(real > 1)[0])
        equilibrium = f"{n_unstable}-saddle"

    return equilibrium