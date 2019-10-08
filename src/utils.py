import difflib

from numba import jit, njit

# import future.concurrent


def best_match_dict(parameters: dict, match_param_keys) -> dict:
    _best_match_dict = dict()
    for param in parameters.keys():
        v = difflib.get_close_matches(param, match_param_keys, n=1)
        if v:
            _best_match_dict[v[0]] = parameters[param]

    return _best_match_dict


def best_match_list(inilist: list, match_list: list) -> list:
    _best_match_list = list()
    for param in inilist:
        v = difflib.get_close_matches(param, match_list, n=1)
        if v:
            _best_match_list.append(v[0])

    return _best_match_list


@njit(cache=True)
def rg_equations_interaction(dg1, dg2, dg3, self_g1, self_g2,
                             self_g3, loopsPeierls, loopsCooper):

    Np = loopsPeierls.shape[0]
    N2 = Np // 2
    # inds = (-N2, N2)
    inds = (0, Np)
    for k1 in range(inds[0], inds[1]):
        for k2 in range(inds[0], inds[1]):
            qc = (k1 + k2) % Np
            for k3 in range(inds[0], inds[1]):
                qp = (k3 - k2) % Np
                qpp = (k1 - k3) % Np
                k4 = (k1 + k2 - k3) % Np
                i = (k1, k2, k3)

                for kp in range(inds[0], inds[1]):
                    IP = loopsPeierls[k2, kp, qp]
                    IP2 = loopsPeierls[k2, kp, -qp]
                    IC = loopsCooper[k1, kp, qc]
                    IPP = loopsPeierls[k3, kp, qpp]

                    m1 = (k1, k2, kp)
                    m2 = (kp, (qc - kp) % Np, k3)
                    m3 = (k1, (kp - qp) % Np, kp)
                    m4 = (kp, k2, k3)
                    m5 = (k1, kp, (kp + qp) % Np)
                    m6 = (k2, (kp + qp) % Np, k3)
                    m7 = (k2, (kp + qp) % Np, kp)
                    m8 = ((kp + qp) % Np, k2, k3)
                    m9 = (k2, (kp + qpp) % Np, kp)
                    m10 = (k2, (kp + qpp) % Np, k4)
                    m11 = (k1, kp, (kp + qpp) % Np)
                    m12 = (k1, kp, k3)

                    dg1[i] += -0.5 * (
                        (self_g2[m1] * self_g1[m2]
                            + self_g1[m1] * self_g2[m2]
                         ) * IC
                        - (
                            self_g2[m3] * self_g1[m4]
                            + self_g1[m3] * self_g2[m4]
                            - 2 * self_g1[m3] * self_g1[m4]
                        ) * IP2
                    )
                    dg1[i] += 0.5 * (
                        self_g3[m5] * self_g3[m6]
                        + self_g3[m7] * self_g3[k1, kp, k4]
                        - 2.0 * self_g3[m5] * self_g3[m8]
                    ) * IP

                    dg2[i] += 0.5 * (
                        +(
                            - self_g2[m1] * self_g2[m2]
                            - self_g1[m1] * self_g1[m2]
                        ) * IC
                        + self_g2[m3] * self_g2[m4] * IP2
                    )
                    dg2[i] += 0.5 * self_g3[k1, kp, k4] * self_g3[m6] * IP

                    dg3[i] += 0.5 * (
                        self_g3[m5] * self_g2[m7]
                        + self_g3[k1, kp, k4] * self_g1[m7]
                        + self_g2[m5] * self_g3[m7]
                        + self_g1[m5] * self_g3[m6]
                        - 2 * self_g1[m7] * self_g3[m5]
                        - 2 * self_g3[m7] * self_g1[m5]
                    ) * IP
                    dg3[i] += 0.5 * (
                        self_g3[m12] * self_g2[m9]
                        + self_g3[m10] * self_g2[m11]
                    ) * IPP
