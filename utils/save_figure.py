import matplotlib.pyplot as plt
import numpy as np
import os

def save_color_disp(color: np.ndarray, disp: np.ndarray, fn: str, title: str = None, max_p: int = 100, dpi: int = 256,
                    disp_cmap: str = 'magma', show_ticks=True):
    # preprocess
    vmax = np.percentile(disp, max_p)
    # draw
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(color)
    axes[0].set_title('Color')
    axes[1].imshow(disp, cmap=disp_cmap, vmax=vmax)
    axes[1].set_title('Disparity')
    if not show_ticks:
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    if title is not None:
        fig.suptitle(title)
    # save and close
    plt.savefig(fn, dpi=dpi)
    plt.close()

def save_disp(color: np.ndarray, disp: np.ndarray, fn: str, color_fn: str, title: str = None, max_p: int = 100, dpi: int = 256,
                    disp_cmap: str = 'magma', show_ticks=True):
    # preprocess
    vmax = np.percentile(disp, max_p)
    # draw
    plt.imshow(disp, cmap=disp_cmap, vmax=vmax)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([]) 
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(fn), exist_ok=True)

    # save and close
    plt.savefig(fn, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.imshow(color)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(color_fn), exist_ok=True)
    
    plt.savefig(color_fn, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_disp(color: np.ndarray, disp: np.ndarray, fn: str, title: str = None, max_p: int = 100, dpi: int = 256,
              disp_cmap: str = 'turbo', show_ticks=True):
    # preprocess
    vmax = np.percentile(disp, max_p)
    
    # draw
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the first image (disp)
    ax[0].imshow(disp, cmap=disp_cmap, vmax=vmax)
    ax[0].axis('off')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    # Plot the second image (color)
    ax[1].imshow(color)
    ax[1].axis('off')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    # Adjust space between subplots
    plt.tight_layout()
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    
    # Save the concatenated image and close the figure
    plt.savefig(fn, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # # Save the individual color image and close the figure
    # os.makedirs(os.path.dirname(color_fn), exist_ok=True)
    # plt.imshow(color)
    # plt.axis('off')
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(color_fn, dpi=dpi, bbox_inches='tight', pad_inches=0)
    # plt.close()

def save_colors(in_color: np.ndarray, out_color: np.ndarray, fn: str, title: str = None, dpi: int = 256,
                show_ticks=True):
    # draw
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(in_color)
    axes[0].set_title('Input')
    axes[1].imshow(out_color)
    axes[1].set_title('Output')
    if not show_ticks:
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    if title is not None:
        fig.suptitle(title)
    # save and close
    plt.savefig(fn, dpi=dpi)
    plt.close()
