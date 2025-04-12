import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

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
