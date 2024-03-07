import numpy as np
import cv2
from typing import List, Any
from funcs import denormalization


#save_tensor_grid(inputs, masked_outputs, k)
def save_tensor_grid(tensor_list1: List[Any], tensor_list2: List[Any], k: int) -> None:
    """ 
    This function saves a grid of images created from tensor_list1 and tensor_list2.
    
    Args:
        tensor_list1 (List[Any]): A list of tensors.
        tensor_list2 (List[Any]): A list of tensors.
        k (int): The value of k.
        
    Returns:
        None
    """
    n = len(tensor_list1)

    for i in range(2):

        # Create an empty list to store the images
        images = []

        for j in range(n):
            # Plot the first tensor
            img1 = tensor_list1[j][i].cpu().numpy()
            img1 = denormalization(img1)

            # Plot the second tensor
            img2 = tensor_list2[j][i].cpu().numpy()
            img2 = denormalization(img2)

            # Concatenate the images horizontally
            img = np.vstack((img1, img2))

            # Add the concatenated image to the list
            images.append(img)

        # Concatenate the images vertically
        print(len(images))
        grid = np.hstack(images)

        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        # Save the grid as an image
        cv2.imwrite(f"examining/tensor_grid_k{k}_{i}.png", grid)