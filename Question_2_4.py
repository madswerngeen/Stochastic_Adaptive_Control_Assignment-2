import numpy as np
from numpy.polynomial import polynomial as P

def solve_diophantine(A, C, m):
    """
    Solves the Diophantine equation:
        C(q⁻¹) = A(q⁻¹) · G(q⁻¹) + q⁻ᵐ · S(q⁻¹)

    Parameters:
    A : list or array-like
        Coefficients of the polynomial A(q⁻¹)
    C : list or array-like
        Coefficients of the polynomial C(q⁻¹)
    m : int
        Delay (i.e., the number of leading terms to extract in G)

    Returns:
    S : list
        Coefficients of the polynomial S(q⁻¹)
    G : list
        Coefficients of the polynomial G(q⁻¹), length = m
    """
    
    G = []
    if len(C) < len(A): # case degree C < degree A
        S = list(C) + [0 for _ in range(len(A)-len(C))] # pad with zeros
    else: # case degree A < degree C
        S = C
        A = list(A) + [0 for _ in range(len(C)-len(A))] # pad with zeros

    for i in range(m):
        G.append(S[0])
        S = [S[j] - S[0] * A[j] for j in range(1, len(S))] + [0]

    S = S[:-1]
    return S, G

def get_GMV_poly(A, B, C, Ay=None, By=None, Hw=None, Au=None, Bu=None, rho=1.0):
    """
    Compute polynomials R(q⁻¹), Q(q⁻¹), S(q⁻¹) for the Generalized Minimum Variance (GMV) control law:
        R(q⁻¹) u_t = Q(q⁻¹) w_t - S(q⁻¹) y_t

    The GMV control law is derived using the Diophantine equation, and the resulting polynomials 
    are computed based on ARMAX system coefficients, transfer functions, and control parameters.

    Parameters:
    A, B, C : list or np.array
        Polynomials of the ARMAX model where A, B, and C represent coefficients 
        of the system equation (A q⁻¹ y_t = B q⁻¹ u_t + C q⁻¹ e_t).
    Ay, By : list, optional
        Polynomials of the reference system (typically Ay = 1, By = 1 for simple cases).
    Hw : list, optional
        Transfer function of the setpoint w_t (default is [1] which means no filtering).
    Au, Bu : list, optional
        Polynomials of the control input system (default values represent a first-order difference).
    rho : float, optional
        Control effort weighting parameter (default = 1.0), affecting the trade-off between control effort 
        and tracking error.

    Returns:
    R, Q, S : np.array
        Polynomials for the GMV controller, where:
        - R(q⁻¹) is the controller’s transfer function for the control input.
        - Q(q⁻¹) is the transfer function for the reference input w_t.
        - S(q⁻¹) is the transfer function for the output feedback y_t.
    """

    # Default settings for MV1a controller if not provided
    if Ay is None:
        Ay = [1]  # Default to no modification for Ay (i.e., Ay = 1)
    if By is None:
        By = [1]  # Default to no modification for By (i.e., By = 1)
    if Hw is None:
        Hw = [1]  # Default setpoint transfer function (no modification)
    if Bu is None:
        Bu = [1, -1]  # Default to (1 - q⁻¹) for control input transfer function
    if Au is None:
        Au = [1]  # Default to no modification for Au (i.e., Au = 1)

    # Solve the Diophantine equation: Aᵧ(q⁻¹) * A(q⁻¹) * G(q⁻¹) + q⁻¹ * S(q⁻¹) = Bᵧ(q⁻¹) * C(q⁻¹)
    S, G = solve_diophantine(P.polymul(Ay, A), P.polymul(By, C), 1)

    # Compute the polynomial R(q⁻¹): Au*B*G + (rho / B0) * C*Bu
    R = P.polyadd(P.polymul(Au, P.polymul(B, G)), (rho / B[0]) * P.polymul(C, Bu))

    # Compute the polynomial Q(q⁻¹): Au * (Hw * C)
    Q = P.polymul(Au, P.polymul(C, Hw))

    # Compute the polynomial S(q⁻¹): -Au * (S / Ay) [Note that remainder of poly-division is removed]
    S = -P.polymul(Au, P.polydiv(S, Ay)[0])
    
    return R, Q, S
    