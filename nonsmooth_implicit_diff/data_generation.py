import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from nonsmooth_implicit_diff.utils import convert_to_torch, get_train_val_test_split

import torch
from torchvision import datasets, transforms




def make_sparse_regression(n_samples=100, n_features=100, noise=0.1, effective_rank=None,
                           test_size=0.2, val_size=0.3,
                           n_informative=20, correlated=False, shuffle=False, random_state=None):

    rng = np.random.default_rng(random_state)

    w = np.zeros(n_features)
    informative_indices = np.array(list(range(n_informative)))
    # print(informative_indices)
    if shuffle:
        informative_indices = rng.choice(np.arange(n_features), n_informative, replace=False)

    w[informative_indices] = np.abs(rng.standard_normal(n_informative))

    mean = np.zeros(n_features)


    cov = np.eye(n_features)
    if correlated:
        # Compute Covariance
        A = rng.random((n_informative, effective_rank))
        U, eigenvals, _ = np.linalg.svd(A,full_matrices=True)
        eigenvals_new = np.zeros(n_informative)
        eigenvals_new[:len(eigenvals)] = eigenvals/eigenvals.max()
        cov[informative_indices[:, None], informative_indices] =  U @ np.diag(eigenvals_new) @ U.T


    X = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)


    y = X @ w + noise * rng.standard_normal(n_samples)
    
    X_torch, y_torch = convert_to_torch((X, y))
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_split(
        X_torch,
        y_torch,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, w

def load_minst(n_samples=500, val_size=0.3, shuffle=False, random_state=None, labels=None):

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to be in the range [-1, 1]
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if labels is not None:
        filtered_indices = torch.tensor([i for i, (_, label) in enumerate(train_dataset) if label in labels])
        train_dataset = torch.utils.data.Subset(train_dataset, filtered_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, filtered_indices)
    
    train_dataset = torch.utils.data.Subset(train_dataset, range(n_samples))
    
    X = torch.stack([image for image, _ in train_dataset], dim=0)
    y = torch.tensor([label for _, label in train_dataset], dtype=torch.long)
    # y = torch.tensor([1 if label == labels[1] else -1 for _, label in train_dataset], dtype=torch.float)


    X_test = torch.stack([image for image, _ in test_dataset], dim=0)
    y_test = torch.tensor([label for _, label in test_dataset], dtype=torch.long)
    # y_test = torch.tensor([1 if label == labels[1] else -1 for _, label in test_dataset], dtype=torch.float)



    # Flatten the images to vectors
    X = X.view(X.size(0), -1)
    X_test = X_test.view(X_test.size(0), -1)

    random_state = check_random_state(0)
    if shuffle:
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation][:n_samples]
        y = y[permutation][:n_samples]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        random_state=random_state,
        test_size=val_size,
    )
    n_classes = 10 if labels is None else len(labels)
    
    w = torch.zeros((X.shape[-1], n_classes))
    
    return X_train, X_val, X_test, y_train, y_val, y_test, w
