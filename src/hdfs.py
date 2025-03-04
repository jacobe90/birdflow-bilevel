import h5py
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Bool


def rotate_array(arr: Float[Array, "T N"], shift: int) -> Float[Array, "T N"]:
    """
    Rotates a 2D array along the first axis with a wrap-around effect,
    the first and last rows of arr are assumed to be identical. The rotation
    adds a copy of the first row to the end of the array after shifting.
    """
    # Remove the last row
    trimmed_arr = jnp.delete(arr, -1, axis=0)

    # Roll the array by the specified shift
    rolled_arr = jnp.roll(trimmed_arr, shift=-shift, axis=0)

    # Append the first row of the rolled array to the end
    appended_row = jnp.expand_dims(rolled_arr[0, :], axis=0)
    return jnp.append(rolled_arr, appended_row, axis=0)

def read_hdf(hdf_src: str) -> tuple[Array, Array, Array]:
    """
    Reads ebird densities, great circle distances, and dynamic masks from an HDF5 file
    """
    with h5py.File(hdf_src, 'r') as f:
        true_densities = jnp.asarray(f['distr']).T
        distance_vector = jnp.asarray(f['distances'])
        dynamic_masks = jnp.asarray(f['geom']['dynamic_mask']).T.astype(bool)
    return true_densities, distance_vector, dynamic_masks

def get_dynamic_masks(hdf_src: str, shift=None) -> list[Array]:
    '''Extracts dynamic_masks from hdf5 file'''
    with h5py.File(hdf_src, 'r') as f:
        dynamic_masks = np.asarray(f['geom']['dynamic_mask']).T.astype(bool)
    if shift:
        dynamic_masks = rotate_array(dynamic_masks, shift)

    return dynamic_masks

def get_plot_parameters(hdf_src: str, shift=None) -> tuple[int, int, Array, Array]:
    """
    Extracts plot-related parameters from an HDF5 file
    """
    with h5py.File(hdf_src, 'r') as f:
        ncol =jnp.asarray(f['geom']['ncol']).astype(int)[0]
        nrow = jnp.asarray(f['geom']['nrow']).astype(int)[0]
        big_mask = jnp.asarray(f['geom']['mask']).flatten().astype(bool)
        dynamic_masks = jnp.asarray(f['geom']['dynamic_mask']).T.astype(bool)

    if shift:
        dynamic_masks = rotate_array(dynamic_masks, shift)

    return ncol, nrow, dynamic_masks, big_mask


def mask_input(true_densities: Float[Array, "T N"], 
               distances: Float[Array, "N"], 
               masks: Bool[Array, "T N"]) -> tuple[list[Array], list[Array]]:
    """
    Masks densities and distances based on dynamic masks, with optional self-feedback distances.
    """
    weeks, cells = true_densities.shape

    # Create a symmetric distance matrix
    distance_matrix = jnp.zeros((cells, cells))
    distance_matrix = distance_matrix.at[jnp.triu_indices(cells, k=1)].set(distances)
    distance_matrix += distance_matrix.T

    # Mask distances for each time step
    masked_distances = [
        distance_matrix[masks[i], :][:, masks[i + 1]] for i in range(weeks - 1)
    ]
    masked_distances.append(distance_matrix[masks[0], :][:, masks[0]])

    # Mask densities for each time step
    masked_densities = [density[mask] for density, mask in zip(true_densities, masks)]

    return masked_densities, masked_distances


def process_hdf(hdf_src: str, shift=None) -> tuple[list[Array], list[Array]]:
    """
    Processes an HDF5 file to compute masked densities and distances
    """
    # Read the HDF5 data
    true_densities, distances, masks = read_hdf(hdf_src)
    
    # shift the data if necessary
    if shift is not None:
        true_densities, masks = rotate_array(true_densities, shift), rotate_array(masks, shift)

    # Mask the densities and distances
    masked_densities, masked_distances = mask_input(
        true_densities, distances, masks
    )

    return masked_densities, masked_distances