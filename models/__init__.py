import jax
from enum import Enum

class EnumMethod(Enum):
    """
    Enumeration of available solvers for model fitting.

    This enum defines different solver methods available for model fitting. Each
    solver corresponds to a specific configuration or variation of the `JKOnet` models,
    with options for different types of parametrizations and regularizations.

    Attributes
    ----------
    JKO_NET_STAR : str
        Solve with JKOnet* with full generality, accommodating all features.
    JKO_NET_STAR_POTENTIAL : str
        Fit only the potential energy component using JKOnet*.
    JKO_NET_STAR_POTENTIAL_INTERNAL : str
        Fit both the potential energy and the Wiener process component using JKOnet*.
    JKO_NET_STAR_TIME_POTENTIAL : str
        Fit the potential energy with time-dependent features using JKOnet*.
    JKO_NET_STAR_LINEAR : str
        Solve with JKOnet* using linear parametrization for the model.
    JKO_NET_STAR_LINEAR_POTENTIAL : str
        Solve with JKOnet* using linear parametrization for only potential and internal energies.
    JKO_NET_STAR_LINEAR_POTENTIAL_INTERNAL : str
        Solve with JKOnet* using linear parametrization for both potential and internal energies.
    JKO_NET_STAR_LINEAR_INTERACTION : str
        Solve with JKOnet* using linear parametrization for interaction energy only.
    JKO_NET : str
        Fit the potential energy using JKOnet, as described in https://arxiv.org/abs/2106.06345.
    JKO_NET_VANILLA : str
        Fit the potential energy using JKOnet without ICNN.
    JKO_NET_MONGE_GAP : str
        Fit the potential energy using JKOnet with a Monge gap regularizer.
    """
    JKO_NET_STAR = 'jkonet-star'
    JKO_NET_STAR_POTENTIAL = 'jkonet-star-potential'
    JKO_NET_STAR_POTENTIAL_INTERNAL = 'jkonet-star-potential-internal'
    JKO_NET_STAR_TIME_POTENTIAL = 'jkonet-star-time-potential'
    JKO_NET_STAR_LINEAR = 'jkonet-star-linear'
    JKO_NET_STAR_LINEAR_POTENTIAL = 'jkonet-star-linear-potential'
    JKO_NET_STAR_LINEAR_POTENTIAL_INTERNAL = 'jkonet-star-linear-potential-internal'
    JKO_NET_STAR_LINEAR_INTERACTION = 'jkonet-star-linear-interaction'
    JKO_NET = 'jkonet'
    JKO_NET_VANILLA = 'jkonet-vanilla'
    JKO_NET_MONGE_GAP = 'jkonet-monge-gap'

    def __str__(self) -> str:
        """
        Return the string representation of the enumeration value.

        Returns
        -------
        str
            The string value of the enumeration item.
        """
        return self.value

def get_model(
        solver: EnumMethod,
        config: dict,
        data_dim: int,
        dt: float):
    """
    Retrieve a model class based on the specified solver and configuration.

    Depending on the solver provided, this function imports the appropriate model class
    and returns an instance of it, configured according to the provided `config` dictionary.

    Parameters
    ----------
    solver : EnumMethod
        An enumeration value that specifies which model to retrieve. The value
        determines which model class to import and return.
    config : dict
        A dictionary containing configuration parameters for the model. The structure
        of this dictionary depends on the model being used and may include fields such as
        'train', 'energy', etc.
    data_dim : int
        The dimensionality of the input data, which will be used to configure the model.
    dt : float
        Represents the timescale over which the diffusion process described by the
        Fokker-Planck equation is considered.

    Returns
    -------
    Type
        The model class corresponding to the specified solver, initialized with the provided
        configuration, data dimensionality, and timestep.

    Raises
    ------
    NotImplementedError
        If the specified solver is not implemented or recognized.
    """
    cls = None
    if solver == EnumMethod.JKO_NET_STAR:
        from models.jkonet_star import JKOnetStar
        cls = JKOnetStar
    elif solver == EnumMethod.JKO_NET_STAR_POTENTIAL:
        from models.jkonet_star import JKOnetStarPotential
        cls = JKOnetStarPotential
    elif solver == EnumMethod.JKO_NET_STAR_POTENTIAL_INTERNAL:
        from models.jkonet_star import JKOnetStarPotentialInternal
        cls = JKOnetStarPotentialInternal
    elif solver == EnumMethod.JKO_NET_STAR_TIME_POTENTIAL:
        from models.jkonet_star import JKOnetStarTimePotential
        cls = JKOnetStarTimePotential
    elif solver in [
        EnumMethod.JKO_NET_STAR_LINEAR, 
        EnumMethod.JKO_NET_STAR_LINEAR_POTENTIAL, 
        EnumMethod.JKO_NET_STAR_LINEAR_POTENTIAL_INTERNAL,
        EnumMethod.JKO_NET_STAR_LINEAR_INTERACTION]:
        from models.jkonet_star import JKOnetStarLinear
        cls = JKOnetStarLinear
        config['train']['epochs'] = 1
        config['train']['eval_freq'] = 1
        config['energy']['linear']['potential'] = True
        config['energy']['linear']['internal'] = True
        config['energy']['linear']['interaction'] = True
        if solver == EnumMethod.JKO_NET_STAR_LINEAR_POTENTIAL:
            config['energy']['linear']['interaction'] = False
            config['energy']['linear']['internal'] = False
        elif solver == EnumMethod.JKO_NET_STAR_LINEAR_POTENTIAL_INTERNAL:
            config['energy']['linear']['interaction'] = False
        elif solver == EnumMethod.JKO_NET_STAR_LINEAR_INTERACTION:
            config['energy']['linear']['potential'] = False
            config['energy']['linear']['internal'] = False
    elif solver == EnumMethod.JKO_NET:
        from models.jkonet import JKOnet
        cls = JKOnet
    elif solver == EnumMethod.JKO_NET_VANILLA:
        from models.jkonet import JKOnetVanilla
        cls = JKOnetVanilla
    elif solver == EnumMethod.JKO_NET_MONGE_GAP:
        from models.jkonet import JKOnetMongeGap
        cls = JKOnetMongeGap
    else:
        raise NotImplementedError(f'Solver {solver} not implemented yet.')
                                  
    return cls(config, data_dim, dt)