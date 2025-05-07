import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def _get_frequency_ticks(center_freqs: np.ndarray, num_ticks: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Get frequency ticks and labels for the y-axis.
    
    Args:
        center_freqs (torch.Tensor): Center frequencies of the filters
        num_ticks (int): Number of ticks to show
        
    Returns:
        tuple: (tick positions, tick labels)
    """
    # Convert to numpy
    
    # Get log-spaced indices
    indices = np.logspace(0, np.log10(len(center_freqs)-1), num_ticks, dtype=int)
    
    # Get corresponding frequencies
    tick_freqs = center_freqs[indices]
    
    # Format labels
    tick_labels = [f"{freq:.0f} Hz" for freq in tick_freqs]
    
    return indices, tick_labels

def visualize_features(
    features: torch.Tensor,
    center_freqs: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 256,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'viridis',
    return_figure: bool = False
) -> Optional[plt.Figure]:
    """
    Visualize the features extracted by the feature extractor network.
    
    Args:
        features (torch.Tensor): Feature tensor of shape (1, F, T)
        center_freqs (np.ndarray): Center frequencies of the filters
        sample_rate (int): Sample rate of the original audio
        hop_length (int): Hop length used in feature extraction
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        cmap (str): Colormap to use for visualization
        return_figure (bool): Whether to return the figure object
        
    Returns:
        plt.Figure if return_figure is True, else None
    """
    # Convert to numpy and squeeze batch dimension
    features_np = features.squeeze(0).numpy()
    
    # Create time axis
    num_frames = features_np.shape[1]
    time_axis = np.arange(num_frames) * hop_length / sample_rate
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot features
    im = ax.imshow(
        features_np,
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], 0, features_np.shape[0]-1],
        cmap=cmap
    )
    
    # Set y-axis to log scale and add frequency ticks
    y_ticks, y_labels = _get_frequency_ticks(center_freqs)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (dB)')
    
    # Set labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    if return_figure:
        return fig
    else:
        plt.show()
        return None

def summarize_features(
    features: torch.Tensor,
    window_size: int,
    stride: Optional[int] = None,
    mode: str = 'mean'
) -> torch.Tensor:
    """
    Summarize features using window averaging.
    
    Args:
        features (torch.Tensor): Feature tensor of shape (N, F, T)
        window_size (int): Size of the averaging window
        stride (int, optional): Stride for the window. If None, uses window_size
        mode (str): Aggregation mode ('mean' or 'max')
        
    Returns:
        torch.Tensor: Summarized features of shape (N, F, T_new)
    """
    if stride is None:
        stride = window_size
    
    N, F, T = features.shape
    
    # Calculate new temporal dimension
    T_new = (T - window_size) // stride + 1
    
    # Initialize output tensor
    summarized = torch.zeros((N, F, T_new), device=features.device)
    
    # Apply window averaging
    for i in range(T_new):
        start = i * stride
        end = start + window_size
        window = features[:, :, start:end]
        
        if mode == 'mean':
            summarized[:, :, i] = window.mean(dim=2)
        elif mode == 'max':
            summarized[:, :, i] = window.max(dim=2)[0]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    return summarized

def visualize_feature_embeddings(
    features: torch.Tensor,
    labels: torch.Tensor,
    label_names: List[str],
    method: str = 'pca',
    n_components: int = 2,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    return_figure: bool = False,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    mode: str = 'mean'
) -> Optional[plt.Figure]:
    """
    Visualize feature embeddings using dimensionality reduction.
    
    Args:
        features (torch.Tensor): Feature tensor of shape (N, F, T)
        labels (torch.Tensor): Label tensor of shape (N,)
        label_names (List[str]): Names of the classes
        method (str): Dimensionality reduction method ('pca' or 'tsne')
        n_components (int): Number of components to reduce to
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        return_figure (bool): Whether to return the figure object
        window_size (int, optional): Size of the averaging window
        stride (int, optional): Stride for the window
        mode (str): Aggregation mode ('mean' or 'max')
        
    Returns:
        plt.Figure if return_figure is True, else None
    """
    # Summarize features if window_size is provided
    if window_size is not None:
        features = summarize_features(features, window_size, stride, mode)
        features = features.permute(0, 2, 1)
    
    # Convert to numpy and reshape features
    features_np = features.numpy()
    N, T, F = features_np.shape
    features_flat = features_np.reshape(N*T, F)
    
    # Convert labels to numpy
    labels_np = labels.numpy()
    labels_np = np.repeat(labels_np, T)
    
    # Apply dimensionality reduction
    embeddings = dimensionality_reduction(method, features=features_np, 
                                          n_components=n_components, features_flat=features_flat, 
                                          random_sate=None)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    print(embeddings.shape)
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels_np,
        alpha=0.7
    )
    
    # Add colorbar with class names
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks(np.arange(len(label_names)))
    cbar.set_ticklabels(label_names)
    
    # Set labels
    if method.lower() == 'pca':
        ax.set_xlabel(f'PC1')
        ax.set_ylabel(f'PC2')
    else:
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        method_name = 'PCA' if method.lower() == 'pca' else 't-SNE'
        agg_mode = f' ({mode} over {window_size} frames)' if window_size else ''
        ax.set_title(f'{method_name} Visualization of Feature Embeddings{agg_mode}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    if return_figure:
        return fig
    else:
        plt.show()
        return None
    
def dimensionality_reduction(
        method : str ,
        features : np.ndarray , 
        n_components : int ,
        features_flat : int , 
        random_sate : int , 
):
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        embeddings = reducer.fit_transform(features_flat)
        explained_variance = reducer.explained_variance_ratio_
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        embeddings = reducer.fit_transform(features_flat)
        explained_variance = None
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    return embeddings

