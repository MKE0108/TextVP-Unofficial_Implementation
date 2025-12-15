
"""
Image utilities for grid arrangement, visualization and saving.

This module provides functions for arranging images in grids with row/column labels,
displaying images, and saving image grids.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from IPython.display import display


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Add text label under an image.
    
    Args:
        image: Input image as numpy array (H, W, C)
        text: Text to add under the image
        text_color: RGB color tuple for text
    
    Returns:
        Image with text added below
    """
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def create_label_image(text: str, width: int, height: int, 
                       font_scale: float = 0.8, 
                       bg_color: Tuple[int, int, int] = (255, 255, 255),
                       text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Create an image containing only text label.
    
    Args:
        text: Text to display
        width: Width of the label image
        height: Height of the label image
        font_scale: Font scale for cv2
        bg_color: Background color (RGB)
        text_color: Text color (RGB)
    
    Returns:
        Label image as numpy array
    """
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:, :] = bg_color
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, font_scale, 2)[0]
    text_x = (width - textsize[0]) // 2
    text_y = (height + textsize[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, 2)
    return img


def arrange_images_grid(
    images: List[np.ndarray],
    num_rows: int,
    num_cols: Optional[int] = None,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    cell_labels: Optional[List[str]] = None,
    offset_ratio: float = 0.02,
    label_height: int = 40,
    label_width: int = 100,
) -> np.ndarray:
    """
    Arrange images in a grid with optional row and column labels.
    
    Args:
        images: List of images as numpy arrays (must be same size)
        num_rows: Number of rows in the grid
        num_cols: Number of columns (auto-calculated if None)
        row_labels: Labels for each row (displayed on the left)
        col_labels: Labels for each column (displayed on top)
        cell_labels: Labels for each cell (displayed under each image)
        offset_ratio: Ratio of offset between images
        label_height: Height of column label area
        label_width: Width of row label area
    
    Returns:
        Combined grid image as numpy array
    """
    if not images:
        raise ValueError("Images list cannot be empty")
    
    # Convert to list if needed
    if isinstance(images, np.ndarray) and images.ndim == 4:
        images = [images[i] for i in range(images.shape[0])]
    
    # Ensure all images are uint8
    images = [img.astype(np.uint8) for img in images]
    
    # Calculate grid dimensions
    if num_cols is None:
        num_cols = (len(images) + num_rows - 1) // num_rows
    
    # Pad with empty images if needed
    num_empty = num_rows * num_cols - len(images)
    if num_empty > 0:
        empty_image = np.ones(images[0].shape, dtype=np.uint8) * 255
        images = images + [empty_image] * num_empty
    
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    
    # Add cell labels if provided
    if cell_labels:
        labeled_images = []
        for i, img in enumerate(images):
            if i < len(cell_labels) and cell_labels[i]:
                labeled_images.append(text_under_image(img, cell_labels[i]))
            else:
                # Add empty space for consistency
                labeled_images.append(text_under_image(img, ""))
        images = labeled_images
        h = images[0].shape[0]  # Update height after adding labels
    
    # Calculate total dimensions
    has_row_labels = row_labels is not None and len(row_labels) > 0
    has_col_labels = col_labels is not None and len(col_labels) > 0
    
    total_width = w * num_cols + offset * (num_cols - 1)
    total_height = h * num_rows + offset * (num_rows - 1)
    
    if has_row_labels:
        total_width += label_width + offset
    if has_col_labels:
        total_height += label_height + offset
    
    # Create canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Calculate starting positions
    start_x = label_width + offset if has_row_labels else 0
    start_y = label_height + offset if has_col_labels else 0
    
    # Place images
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < len(images):
                y = start_y + i * (h + offset)
                x = start_x + j * (w + offset)
                canvas[y:y + h, x:x + w] = images[idx]
    
    # Add column labels
    if has_col_labels:
        for j, label in enumerate(col_labels):
            if j < num_cols:
                x = start_x + j * (w + offset)
                label_img = create_label_image(label, w, label_height)
                canvas[0:label_height, x:x + w] = label_img
    
    # Add row labels
    if has_row_labels:
        for i, label in enumerate(row_labels):
            if i < num_rows:
                y = start_y + i * (h + offset)
                label_img = create_label_image(label, label_width, h, font_scale=0.5)
                canvas[y:y + h, 0:label_width] = label_img
    
    return canvas


def view_images(
    images: Union[List[np.ndarray], np.ndarray],
    num_rows: int = 1,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    cell_labels: Optional[List[str]] = None,
    offset_ratio: float = 0.02,
):
    """
    Display images in a grid with optional labels.
    
    Args:
        images: List of images or 4D numpy array
        num_rows: Number of rows in the grid
        row_labels: Labels for each row
        col_labels: Labels for each column
        cell_labels: Labels for each cell
        offset_ratio: Ratio of offset between images
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images]
    
    grid = arrange_images_grid(
        images=images,
        num_rows=num_rows,
        row_labels=row_labels,
        col_labels=col_labels,
        cell_labels=cell_labels,
        offset_ratio=offset_ratio,
    )
    
    pil_img = Image.fromarray(grid)
    display(pil_img)


def save_images(
    images: Union[List[np.ndarray], np.ndarray],
    save_path: str,
    num_rows: int = 1,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    cell_labels: Optional[List[str]] = None,
    offset_ratio: float = 0.02,
) -> Image.Image:
    """
    Save images as a grid with optional labels.
    
    Args:
        images: List of images or 4D numpy array
        save_path: Path to save the image
        num_rows: Number of rows in the grid
        row_labels: Labels for each row
        col_labels: Labels for each column
        cell_labels: Labels for each cell
        offset_ratio: Ratio of offset between images
    
    Returns:
        PIL Image object
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images]
    
    grid = arrange_images_grid(
        images=images,
        num_rows=num_rows,
        row_labels=row_labels,
        col_labels=col_labels,
        cell_labels=cell_labels,
        offset_ratio=offset_ratio,
    )
    
    pil_img = Image.fromarray(grid)
    pil_img.save(save_path)
    return pil_img


class ImageGrid:
    """
    A class to manage image collection and grid arrangement with labels.
    
    Example:
        grid = ImageGrid(row_labels=["Image1", "Image2"], col_labels=["Epoch0", "Epoch5", "Epoch10"])
        grid.add_image(img1)
        grid.add_image(img2)
        grid.view(num_rows=2)
        grid.save("output.png", num_rows=2)
    """
    
    def __init__(
        self,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
    ):
        """
        Initialize ImageGrid.
        
        Args:
            row_labels: Labels for each row
            col_labels: Labels for each column
        """
        self.images: List[np.ndarray] = []
        self.cell_labels: List[str] = []
        self.row_labels = row_labels
        self.col_labels = col_labels
    
    def add_image(self, image: Union[np.ndarray, List[np.ndarray]], label: Optional[str] = None):
        """
        Add image(s) to the grid.
        
        Args:
            image: Single image or list of images
            label: Optional label for the image (or list of labels)
        """
        if isinstance(image, list):
            self.images.extend(image)
            if isinstance(label, list):
                self.cell_labels.extend(label)
            else:
                self.cell_labels.extend([""] * len(image))
        else:
            self.images.append(image)
            self.cell_labels.append(label if label else "")
    
    def set_row_labels(self, labels: List[str]):
        """Set row labels."""
        self.row_labels = labels
    
    def set_col_labels(self, labels: List[str]):
        """Set column labels."""
        self.col_labels = labels
    
    def view(self, num_rows: int = 1):
        """Display the image grid."""
        view_images(
            images=self.images,
            num_rows=num_rows,
            row_labels=self.row_labels,
            col_labels=self.col_labels,
            cell_labels=self.cell_labels if any(self.cell_labels) else None,
        )
    
    def save(self, save_path: str, num_rows: int = 1) -> Image.Image:
        """Save the image grid to a file."""
        return save_images(
            images=self.images,
            save_path=save_path,
            num_rows=num_rows,
            row_labels=self.row_labels,
            col_labels=self.col_labels,
            cell_labels=self.cell_labels if any(self.cell_labels) else None,
        )
    
    def clear(self):
        """Clear all images and cell labels."""
        self.images = []
        self.cell_labels = []
    
    def __len__(self) -> int:
        return len(self.images)
