import jax
from enum import Enum

class EnumMethod(Enum):
    """
    Available solvers.
    """
    JKO_NET_STAR = 'jkonet-star' # Solve with jkonet*, full generality.
    JKO_NET_STAR_POTENTIAL = 'jkonet-star-potential' # Fit only potential energy.
    JKO_NET_STAR_POTENTIAL_INTERNAL = 'jkonet-star-potential-internal' # Fit potential energy + wiener process.
    JKO_NET_STAR_LINEAR = 'jkonet-star-linear' # Solve with jkonet*, linear parametrization.
    JKO_NET_STAR_LINEAR_POTENTIAL = 'jkonet-star-linear-potential' # Solve with jkonet*, linear parametrization of only the potential and internal energies.
    JKO_NET_STAR_LINEAR_POTENTIAL_INTERNAL = 'jkonet-star-linear-potential-internal' # Solve with jkonet*, linear parametrization of potential and internal energies.
    JKO_NET_STAR_LINEAR_INTERACTION = 'jkonet-star-linear-interaction' # Solve with jkonet*, linear parametrization of interaction energy only.
    JKO_NET = 'jkonet' # Fit potential energy with JKOnet, see https://arxiv.org/abs/2106.06345.
    JKO_NET_VANILLA = 'jkonet-vanilla' # Fit potential energy with JKOnet, no ICNN
    JKO_NET_MONGE_GAP = 'jkonet-monge-gap' # Fit potential energy with JKOnet using Monge gap regularizer

    def __str__(self):
        return self.value

def get_model(
        solver: EnumMethod,
        config: dict,
        data_dim: int,
        dt: float):
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