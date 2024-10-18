Adding a custom model
======================

In this section, we will walk through the key steps required to add a new model. As an example, we'll demonstrate
by adding a simple dummy model which, after training, returns a trivial potential, internal, and interaction energy.

The model interface
--------------------

First, we create a ``DummyModel`` class. This class will be the one used to create the object ``model`` in the
training script.

.. code-block:: python

    class DummyModel:
        def __init__(self, config: dict, data_dim: int, tau: float) -> None:
            self.data_dim = data_dim
            self.layers = config['energy']['model']['layers']
            self.config_optimizer = config['energy']['optim']

            # create energy models
            self.model_potential = MLP(self.layers)

Next, we need a method to load the dataset. There are several dataset classes already provided for different input data format. Refer to the :mod:`Dataset` module for more information on the different dataset classes.

For this tutorial, we will use the ``CouplingsDataset`` class, which returns information about the couplings and the density, and is the dataset used by the JKOnet\* model.

.. code-block:: python

    def load_dataset(self, dataset_name: str) -> CouplingsDataset:
        return CouplingsDataset(dataset_name)

Here, ``dataset_name`` is the name of the dataset to be loaded. The dataset must be in the ``data/`` directory. See also the :mod:`Dataset` module.

:code:`JAX` is a functional framework and we use classes only to provide a common interface. To mantain the state, we use a separate state variable. Thus, we neeed to define a method to create the state for our energy models.

.. code-block:: python

    def create_state(self, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
        potential = create_train_state(
            rng, self.model_potential, get_optimizer(self.config_optimizer), self.data_dim)
        return potential, _, _

Note that we have three return values, one for each energy term. In this case, we are only using the potential term.

Then, we need to provide a loss function for the model. In this case, since it is a dummy model, we generate a loss function with no physical meaning.
When we minimize the loss and update the parameters, the function will converge to a potential that always returns 0.

.. code-block:: python

    def loss(
        self,
        potential_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.abs(jnp.sum(self.model_potential.apply({'params': potential_params}, ys)))

We must also define a ``train_step`` method. This function must contain the calculation of the loss and gradients, as well as
the subsequent state update.

.. code-block:: python

    def train_step(
        self,
        state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
        sample: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:

        xs, ys, t, ws, rho, rho_grad = sample
        potential, internal, interaction = state
        loss, grads = jax.value_and_grad(
                self.loss, argnums=(0, 1, 2))(
                    potential.params, xs, ys)
        potential = potential.apply_gradients(grads=grads[0])
        return loss, (potential, _, _)

Finally, the model must provide methods to get the potential, beta, and interaction energies. In this case, we will return a trivial potential and interaction energy, and a beta of 0 using the methods: ``get_potential``, ``get_beta``, and ``get_interaction``.

.. code-block:: python

    def get_potential(self, state):
        potential, _, _ = state
        return lambda x: potential.apply_fn({'params': potential.params}, x)

    def get_beta(self, state):
        return 0.

    def get_interaction(self, state):
        return lambda x: 0.

Adding the ``DummyModel`` to the solvers
---------------------------------------------

Now that the model is ready, we need to add it to the list of available solvers.

For this, we edit the ``__init__.py`` file in the ``models`` directory:

.. code-block:: python

    class EnumMethod(Enum):
        ... # Other models here
        DUMMY = 'dummy-model'  # The dummy model we're adding.

.. code-block:: python

    def get_model(
        solver: EnumMethod,
        config: dict,
        data_dim: int,
        dt: float):

        if solver == EnumMethod.DUMMY:
            from models.jkonet_star import DummyModel
            cls = DummyModel

        # Other models retrieval logic here
        # ...

        return cls(config, data_dim, dt)

Adding a Colormap for the ``DummyModel``
---------------------------------------------

Finally, add a colormap specific to the model in the ``style.yaml`` file. This will be used when plotting the predictions.

.. code-block:: yaml

    # training
    groundtruth:
      light: '#F1F1F1'
      dark: '#C7B7A3'
      marker: 'o'

    jkonet-star:
      light: '#CDF5FD'
      dark: '#A0E9FF'
      marker: '+'

    dummy-model:
      light: '#FFC1C1'
      dark: '#FF6666'
      marker: '+'


Great! You are now ready to train your first ``DummyModel``! 