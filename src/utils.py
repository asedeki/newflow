import difflib


def best_match_dict(parameters: dict, match_param_keys) -> dict:
    # # input(parameters.items())
    # # input(match_param_keys)
    _best_match_dict = {
        difflib.get_close_matches(key, match_param_keys, n=1)[0]: v
        for key, v in parameters.items()
    }
    return _best_match_dict
