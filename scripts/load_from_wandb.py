import re
from typing import Dict

wandb_config = {
    "entity": "passionfruit-ai",
    "project": "jkonet-star-camera-ready",
}

def parse_name(run_name: str) -> Dict[str, str]:
    """
    Parses the run name and extracts model configuration details.

    The run name is expected to follow the pattern:
    <method>.potential_<potential>_internal_<internal>_beta_<beta>_interaction_<interaction>
        _dt_<dt>_T_<T>_dim_<dim>_N_<N>_gmm_<gmm>_seed_<seed>_split_<split>


    Parameters
    ----------
    run_name : str
        The run name string containing the configuration parameters.

    Returns
    -------
    Dict[str, str]
        A dictionary with the extracted configuration details:

        - `method`: The method used in the run.
        - `potential`: Type of potential used.
        - `internal`: Type of internal process used.
        - `beta`: Beta parameter value.
        - `interaction`: Type of interaction used.
        - `dt`: Timestep size.
        - `T`: Total time or steps.
        - `dim`: Dimensionality.
        - `N`: Number of samples.
        - `gmm`: Gaussian Mixture Model parameter.
        - `seed`: Random seed used.
        - `split`: Data split parameter.
    """
    # Regular expression to capture the values
    pattern = re.compile(r'(?P<method_value>.*?).potential_(?P<potential_value>.*?)_internal_(?P<internal_value>.*?)_beta_(?P<beta>[0-9.]+)_interaction_(?P<interaction_value>.*?)_dt_(?P<dt>[0-9.]+)_T_(?P<T>[0-9.]+)_dim_(?P<dim>[0-9.]+)_N_(?P<N>[0-9.]+)_gmm_(?P<gmm>[0-9.]+)_seed_(?P<seed>[0-9.]+)(?:_split_(?P<split>[0-9.]+))?')

    # Search the pattern in the input string
    match = pattern.search(run_name)

    run_details = {}

    # Extract the values
    if match:
        run_details['method'] = match.group('method_value')
        run_details['potential'] = match.group('potential_value')
        run_details['internal'] = match.group('internal_value')
        run_details['beta'] = match.group('beta')
        run_details['interaction'] = match.group('interaction_value')
        run_details['dt'] = match.group('dt')
        run_details['T'] = match.group('T')
        run_details['dim'] = match.group('dim')
        run_details['N'] = match.group('N')
        run_details['gmm'] = match.group('gmm')
        run_details['seed'] = match.group('seed')
        run_details['split'] = match.group('split')

    sinkhorn_pattern = re.compile(r"sinkhorn_(\d+\.?\d*)")
    match = sinkhorn_pattern.search(run_name)
    if match:
        run_details['sinkhorn'] = match.group(1)
    else:
        run_details['sinkhorn'] = '0.0'


    return run_details
