import numpy as np
import torch
import robust_laplacian
import scipy

def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian

    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()

def compute_matrix(verts, faces, lambda_, alpha=None, laplacian_type='robust',cfg={}):
    """
    Build the parameterization matrix.

    If alpha is defined, then we compute it as (1-alpha)*I + alpha*L otherwise
    as I + lambda*L as in the paper. The first definition can be slightly more
    convenient as it the scale of the resulting matrix doesn't change much
    depending on alpha.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    lambda_ : float
        Hyperparameter lambda of our method, used to compute the
        parameterization matrix as (I + lambda_ * L)
    alpha : float in [0, 1[
        Alternative hyperparameter, used to compute the parameterization matrix
        as ((1-alpha) * I + alpha * L)
    cotan : bool
        Compute the cotangent laplacian. Otherwise, compute the combinatorial one
    """
    if 'laplacian_type' in cfg:
        laplacian_type = cfg['laplacian_type']
    if laplacian_type == 'cot':
        L = laplacian_cot(verts, faces.to(torch.int64))
    elif laplacian_type == 'uniform':
        L = laplacian_uniform(verts, faces)
    elif laplacian_type == 'robust':
        L, _ = robust_laplacian.mesh_laplacian(verts.cpu().numpy(),faces.cpu().numpy())

        # Normalize Laplacian using mass matrix (mass-normalized Laplacian)
        n = L.shape[0]
        mass_matrix = scipy.sparse.eye(n)
        mass_diag = mass_matrix.diagonal()
        inv_mass_diag = 1.0 / (mass_diag + 1e-10)
        mass_matrix_inv = scipy.sparse.diags(inv_mass_diag)
        L = mass_matrix_inv @ L  # Mass-normalized Laplacian
        L = L.tocoo()
        L = L.astype(np.float32)
        indices = np.vstack((L.row, L.col))
        values = L.data * lambda_
        L = torch.sparse_coo_tensor(
            indices=torch.tensor(indices, dtype=torch.int64),
            values=torch.tensor(values, dtype=torch.float32),
            size=L.shape,
            device='cuda:0'
        )

    if cfg.get('a_willmore',None) is not None and cfg.get('a_willmore') > 0.0:
        a_willmore = cfg.get('a_willmore',0)
        a_laplacian = 1 - a_willmore
        L = a_laplacian * L + a_willmore * L @ L

    idx = torch.arange(verts.shape[0], dtype=torch.long, device='cuda')
    eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(verts.shape[0], dtype=torch.float, device='cuda'), (verts.shape[0], verts.shape[0]))
    if alpha is None:
        M = torch.add(eye, lambda_ * L)
    else:
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError(f"Invalid value for alpha: {alpha} : it should take values between 0 (included) and 1 (excluded)")
        M = torch.add((1-alpha)*eye, alpha*L) # M = (1-alpha) * I + alpha * L
    return M.coalesce()


def compute_matrix_with_mass_matrix(verts, faces, lambda_, alpha=None, laplacian_type='robust'):
    """
    Build the parameterization matrix.

    If alpha is defined, then we compute it as (1-alpha)*I + alpha*L otherwise
    as I + lambda*L as in the paper. The first definition can be slightly more
    convenient as it the scale of the resulting matrix doesn't change much
    depending on alpha.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    lambda_ : float
        Hyperparameter lambda of our method, used to compute the
        parameterization matrix as (I + lambda_ * L)
    alpha : float in [0, 1[
        Alternative hyperparameter, used to compute the parameterization matrix
        as ((1-alpha) * I + alpha * L)
    cotan : bool
        Compute the cotangent laplacian. Otherwise, compute the combinatorial one
    """
    mass_matrix = None

    if laplacian_type == 'cotan':
        L = laplacian_cot(verts, faces)
    elif laplacian_type == 'uniform':
        L = laplacian_uniform(verts, faces)
    elif laplacian_type == 'robust':
        L, mass_matrix = robust_laplacian.mesh_laplacian(verts.cpu().numpy(), faces.cpu().numpy())

        # Normalize Laplacian using mass matrix (mass-normalized Laplacian)
        # n = L.shape[0]
        # mass_matrix = scipy.sparse.eye(n)
        # mass_diag = mass_matrix.diagonal()
        # inv_mass_diag = 1.0 / (mass_diag + 1e-10)
        # mass_matrix_inv = scipy.sparse.diags(inv_mass_diag)
        # L = mass_matrix_inv @ L  # Mass-normalized Laplacian

        # convert laplacian to sparse coo tensor
        L = L.tocoo()
        L = L.astype(np.float32)
        indices = np.vstack((L.row, L.col))
        values = L.data * lambda_
        L = torch.sparse_coo_tensor(
            indices=torch.tensor(indices, dtype=torch.int64),
            values=torch.tensor(values, dtype=torch.float32),
            size=L.shape,
            device='cuda:0'
        )

        # convert mass matrix to sparse coo tensor
        mass_matrix = mass_matrix.tocoo()
        mass_matrix = mass_matrix.astype(np.float32)
        indices = np.vstack((mass_matrix.row, mass_matrix.col))
        values = mass_matrix.data * lambda_
        mass_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(indices, dtype=torch.int64),
            values=torch.tensor(values, dtype=torch.float32),
            size=mass_matrix.shape,
            device='cuda:0'
        )

    idx = torch.arange(verts.shape[0], dtype=torch.long, device='cuda')
    if mass_matrix is None:
        eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0),
                                      torch.ones(verts.shape[0], dtype=torch.float, device='cuda'),
                                      (verts.shape[0], verts.shape[0]))
        mass_matrix = eye
    if alpha is None:
        M = torch.add(mass_matrix, lambda_ * L)
    else:
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError(
                f"Invalid value for alpha: {alpha} : it should take values between 0 (included) and 1 (excluded)")
        M = torch.add((1 - alpha) * mass_matrix, alpha * L)  # M = (1-alpha) * I + alpha * L
    return M.coalesce(), mass_matrix.coalesce()
