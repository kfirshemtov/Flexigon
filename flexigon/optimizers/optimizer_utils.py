from flexigon.optimizers.Adam import AdamUniform

def initialize_optimizer(u, v, translation, step_size,use_translation=True,optimizer=AdamUniform):
    """
    Initialize the optimizer

    Parameters
    ----------
    - u : torch.Tensor or None
        Parameterized coordinates to optimize if not None
    - v : torch.Tensor
        Cartesian coordinates to optimize if u is None
    - tr : torch.Tensor
        Global translation to optimize if not None
    - step_size : float
        Step size

    Returns
    -------
    a torch.optim.Optimizer containing the tensors to optimize.
    """
    opt_params = []
    if translation is not None:
        translation.requires_grad = True
        opt_params.append(translation)
    if u is not None:
        u.requires_grad = True
        opt_params.append(u)
    else:
        v.requires_grad = True
        opt_params.append(v)

    if use_translation:
        opt_params = [{'params': opt_params[0]}, {'params': opt_params[1]}]
    else:
        opt_params = [{'params': opt_params[0]}]
    return optimizer(opt_params, lr=step_size)