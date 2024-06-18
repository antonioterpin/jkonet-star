import jax.numpy as jnp
from sklearn.mixture import GaussianMixture
import pickle

class GaussianMixtureModel:
    gms_means = []
    gms_covs_invs = []
    gms_den = []
    gms_weights = []

    def __init__(self):
         pass

    def fit(self, trajectory: jnp.ndarray, n_components: int):
        """
        Fits a Gaussian Mixture Model to the trajectory.

        Parameters:
            trajectory (jnp.ndarray): Trajectory of the particles.
            n_components (int): Number of components in the Gaussian Mixture Model.
        """
        data_dim = trajectory.shape[2]
        for t in range(trajectory.shape[0]):
            # Fit density at time t
            gm = GaussianMixture(n_components=n_components)
            gm.fit(trajectory[t,:,:])

            # Discard components with small determinants
            covariances = gm.covariances_
            dets = jnp.asarray([
                jnp.linalg.det(covariances[i]) for i in range(n_components)])
            idxs = jnp.where(jnp.greater(dets, 1e-4))

            # Store density parameters
            self.gms_means.append(gm.means_[idxs])
            self.gms_covs_invs.append(jnp.linalg.inv(gm.covariances_[idxs]))
            self.gms_den.append(1 / jnp.sqrt((2 * jnp.pi) ** data_dim * dets[idxs]))
            self.gms_weights.append(jnp.asarray(gm.weights_[idxs] / jnp.sum(gm.weights_[idxs])))

    def to_file(self, filename: str):
        """
        Saves the Gaussian Mixture Model to a file.
        """
        data = {
            'gms_means': self.gms_means,
            'gms_covs_invs': self.gms_covs_invs,
            'gms_den': self.gms_den,
            'gms_weights': self.gms_weights
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def from_file(self, filename: str):
        """
        Loads the Gaussian Mixture Model from a file.
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.gms_means = data['gms_means']
            self.gms_covs_invs = data['gms_covs_invs']
            self.gms_den = data['gms_den']
            self.gms_weights = data['gms_weights']

    def gmm_density(self, t, x):
        """
        Computes the density (differentiable) of the Gaussian Mixture Model at time t and state x.

        Parameters:
            t (int): Time index.
            x (jnp.ndarray): State.

        Returns:
            Density of the Gaussian Mixture Model at time t and state x.
        """
        # Step 1: Compute the differences between x and all means
        diffs = x - self.gms_means[t]  # (n_components, dim)

        # Step 2: Apply the matrix operation for each component
        # Using einsum for clarity in matrix multiplication
        mahalanobis_terms = jnp.einsum('ij,ijk,ik->i', diffs, self.gms_covs_invs[t], diffs)  # (n_components,)

        # Step 3: Compute the exponent terms
        exponent_terms = jnp.exp(-0.5 * mahalanobis_terms)  # (n_components,)

        # Step 4: Combine weights, density, and exponent terms
        weighted_terms = self.gms_weights[t] * self.gms_den[t] * exponent_terms  # (n_components,)

        # Step 5: Sum over all components
        result = jnp.sum(weighted_terms)  # Scalar value

        # Step 6: Clip the result to avoid extremely small values
        return jnp.clip(result, a_min=0.00001)