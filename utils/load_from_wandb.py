import re

def parse_name(run_name: str):
    """
    Parse the run name and return the model name and the potential type
    """
    # Regular expression to capture the values
    pattern = re.compile(r'(?P<method_value>.*?).potential_(?P<potential_value>.*?)_internal_(?P<internal_value>.*?)_beta_(?P<beta>[0-9.]+)_interaction_(?P<interaction_value>.*?)_dt_(?P<dt>[0-9.]+)_T_(?P<T>[0-9.]+)_dim_(?P<dim>[0-9.]+)_N_(?P<N>[0-9.]+)_gmm_(?P<gmm>[0-9.]+)_seed_(?P<seed>[0-9.]+)_split_(?P<split>[0-9.]+)')

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

    return run_details
