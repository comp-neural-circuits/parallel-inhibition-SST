import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import eig

class VIP_circuit:
    def __init__(self, W, rE0=3.0, rP0=5.0, rS0=2.0, rV0=2.0, I_stim_E=0.3, I_stim_P=0.3, I_stim_S=0.0, I_stim_V=0.0, I_mod_E=0.0,I_mod_P=0.0,I_mod_S=0.0,I_mod_V=0.0, step_rX=0.01, power=2, mult_f=0.25):
        # Define parameters
        self.W = W
        self.wEE, self.wEP, self.wES, self.wEV, self.wPE, self.wPP, self.wPS, self.wPV, self.wSE, self.wSP, self.wSS, self.wSV, self.wVE, self.wVP, self.wVS, self.wVV = W.flatten()

        self.rE0 = rE0 #for weight sweep set-point
        self.rP0 = rP0
        self.rS0 = rS0  # SST initial firing rate
        self.rV0 = rV0  # VIP initial firing rate
        self.step_rX = step_rX

        self.I_stim_E = I_stim_E
        self.I_stim_P = I_stim_P
        self.I_stim_S = I_stim_S
        self.I_stim_V = I_stim_V
        self.I_mod_E = I_mod_E
        self.I_mod_P = I_mod_P
        self.I_mod_S = I_mod_S
        self.I_mod_V = I_mod_V

        # I/O function parameters
        self.power = power
        self.mult_f = mult_f

        # Define colormap
        self.hex_colors = ['#ffffff', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d']
        self.colors = [[int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5)] for hex_color in self.hex_colors]

        # Create custom colormap
        self.cmap = LinearSegmentedColormap.from_list('custom', self.colors, N=128)
        # self.cmap = 'plasma'
            # --- Add this helper mapping once in the class body ---
        self._w_order = [
        "wEE","wEP","wES","wEV",
        "wPE","wPP","wPS","wPV",
        "wSE","wSP","wSS","wSV",
        "wVE","wVP","wVS","wVV"
        ]
        self._w_index = {name:i for i, name in enumerate(self._w_order)}

    def _compute_block(self,
                    mode: str,
                    rE: float,
                    rP: float,
                    rS: float,
                    rV: float,
                    dE: float = 0.0,
                    dP: float = 0.0,
                    dS: float = 0.0,
                    dV: float = 0.0,
                    add_input: bool = False,):
        """
        General computation block used for both PRE and POST VIP cases.

        Parameters
        ----------
        rE, rP : float
            E and P rates at the grid point.
        dE, dP, dS, dV : float, optional
            Additive deltas to E,P,S,V (use zeros for PRE; use VIP-induced deltas for POST).
        add_v_input : bool, optional
            If True (POST), include +I_mod_V inside the V nonlinearity (matches your bV_mod).

        Returns
        -------
        out : dict
            {
            'gain': float,                       # L_EE * I_stim_E + L_EP * I_stim_P
            'max_ev_real': float,                # max Re eig(J)
            'max_ev_imag': float,                # max Im eig(J)
            'L_EV': float, 'L_PV': float,        # linear-response entries with 'V' column
            'L_SV': float, 'L_VV': float,
            'det': float,                        # determinant-like scalar
            'J': np.ndarray or None,             # Jacobian (None if singular)
            'singular': bool                     # True if |det| < 1e-10
            }
        Notes
        -----
        PRE:  call with d*=0, add_v_input=False.
        POST: call with d* = VIP deltas from PRE, add_v_input=True.
        The xE,xP,xS,xV are computed from baseline rS0,rV0 (as in your code),
        and then gains b* use shifted (r* + d*) in the linear terms, with +I_mod_V
        added to V only when add_v_input=True (exactly matching bV_mod).
        """     

        if mode == "rates":
            # Inputs (computed once from baselines)
            xE = (rE / self.mult_f)**(1 / self.power) - (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV)
            xP = (rP / self.mult_f)**(1 / self.power) - (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV)
            xS = (rS / self.mult_f)**(1 / self.power) - (self.wSE*rE - self.wSP*rP - self.wSS*rS - self.wSV*rV)
            xV = (rV / self.mult_f)**(1 / self.power) - (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV)

            # Effective rates (PRE: same as rE,rP,rS0,rV0; POST: shifted by VIP)
            rE_eff = rE + dE
            rP_eff = rP + dP
            rS_eff = rS + dS
            rV_eff = rV + dV

            # Gains (derivatives). Only V gets the extra +I_mod_V in POST.
            if add_input:
                add_E = self.I_mod_E
                add_P = self.I_mod_P
                add_S = self.I_mod_S
                add_V = self.I_mod_V
            else:
                add_E = 0.0
                add_P = 0.0
                add_S = 0.0
                add_V = 0.0

            bE = self.power * self.mult_f * (self.wEE*rE_eff - self.wEP*rP_eff - self.wES*rS_eff - self.wEV*rV_eff + xE + add_E)**(self.power - 1)
            bP = self.power * self.mult_f * (self.wPE*rE_eff - self.wPP*rP_eff - self.wPS*rS_eff - self.wPV*rV_eff + xP + add_P)**(self.power - 1)
            bS = self.power * self.mult_f * (self.wSE*rE_eff - self.wSP*rP_eff - self.wSS*rS_eff - self.wSV*rV_eff + xS + add_S)**(self.power - 1)
            bV = self.power * self.mult_f * (self.wVE*rE_eff - self.wVP*rP_eff - self.wVS*rS_eff - self.wVV*rV_eff + xV + add_V)**(self.power - 1)
            # print(self.wSE*rE_eff, - self.wSP*rP_eff, - self.wSV*rV_eff, + xS)
            # print(rP_eff, rS_eff)
        elif mode == "weights":
            bE = self.power * self.mult_f * ((rE + self.I_mod_E)/self.mult_f)**((self.power - 1)/self.power)
            bP = self.power * self.mult_f * ((rP + self.I_mod_P)/self.mult_f)**((self.power - 1)/self.power)
            bS = self.power * self.mult_f * ((rS + self.I_mod_S)/self.mult_f)**((self.power - 1)/self.power)
            bV = self.power * self.mult_f * ((rV + self.I_mod_V)/self.mult_f)**((self.power - 1)/self.power)

        # Big denominator (identical form; just uses current b* symbols)
        det_num = -self.wEE*self.wPP*self.wSS*self.wVV + self.wEE*self.wPP*self.wSV*self.wVS + self.wEE*self.wPS*self.wSP*self.wVV - self.wEE*self.wPS*self.wSV*self.wVP - self.wEE*self.wPV*self.wSP*self.wVS + self.wEE*self.wPV*self.wSS*self.wVP + self.wEP*self.wPE*self.wSS*self.wVV - self.wEP*self.wPE*self.wSV*self.wVS - self.wEP*self.wPS*self.wSE*self.wVV + self.wEP*self.wPS*self.wSV*self.wVE + self.wEP*self.wPV*self.wSE*self.wVS - self.wEP*self.wPV*self.wSS*self.wVE - self.wES*self.wPE*self.wSP*self.wVV + self.wES*self.wPE*self.wSV*self.wVP + self.wES*self.wPP*self.wSE*self.wVV - self.wES*self.wPP*self.wSV*self.wVE - self.wES*self.wPV*self.wSE*self.wVP + self.wES*self.wPV*self.wSP*self.wVE + self.wEV*self.wPE*self.wSP*self.wVS - self.wEV*self.wPE*self.wSS*self.wVP - self.wEV*self.wPP*self.wSE*self.wVS + self.wEV*self.wPP*self.wSS*self.wVE + self.wEV*self.wPS*self.wSE*self.wVP - self.wEV*self.wPS*self.wSP*self.wVE - self.wEE*self.wSS*self.wVV/bP + self.wEE*self.wSV*self.wVS/bP + self.wES*self.wSE*self.wVV/bP - self.wES*self.wSV*self.wVE/bP - self.wEV*self.wSE*self.wVS/bP + self.wEV*self.wSS*self.wVE/bP - self.wEE*self.wPP*self.wSS/bV + self.wEE*self.wPS*self.wSP/bV + self.wEP*self.wPE*self.wSS/bV - self.wEP*self.wPS*self.wSE/bV - self.wES*self.wPE*self.wSP/bV + self.wES*self.wPP*self.wSE/bV - self.wEE*self.wSS/(bV*bP) + self.wES*self.wSE/(bV*bP) - self.wEE*self.wPP*self.wVV/bS + self.wEE*self.wPV*self.wVP/bS + self.wEP*self.wPE*self.wVV/bS - self.wEP*self.wPV*self.wVE/bS - self.wEV*self.wPE*self.wVP/bS + self.wEV*self.wPP*self.wVE/bS - self.wEE*self.wVV/(bS*bP) + self.wEV*self.wVE/(bS*bP) - self.wEE*self.wPP/(bS*bV) + self.wEP*self.wPE/(bS*bV) - self.wEE/(bS*bV*bP) + self.wPP*self.wSS*self.wVV/bE - self.wPP*self.wSV*self.wVS/bE - self.wPS*self.wSP*self.wVV/bE + self.wPS*self.wSV*self.wVP/bE + self.wPV*self.wSP*self.wVS/bE - self.wPV*self.wSS*self.wVP/bE + self.wSS*self.wVV/(bE*bP) - self.wSV*self.wVS/(bE*bP) + self.wPP*self.wSS/(bE*bV) - self.wPS*self.wSP/(bE*bV) + self.wSS/(bE*bV*bP) + self.wPP*self.wVV/(bE*bS) - self.wPV*self.wVP/(bE*bS) + self.wVV/(bE*bS*bP) + self.wPP/(bE*bS*bV) + 1/(bE*bS*bV*bP)

        L_EE = (self.wPP*self.wSS*self.wVV - self.wPP*self.wSV*self.wVS - self.wPS*self.wSP*self.wVV + self.wPS*self.wSV*self.wVP + self.wPV*self.wSP*self.wVS - self.wPV*self.wSS*self.wVP + self.wSS*self.wVV/bP - self.wSV*self.wVS/bP + self.wPP*self.wSS/bV - self.wPS*self.wSP/bV + self.wSS/(bV*bP) + self.wPP*self.wVV/bS - self.wPV*self.wVP/bS + self.wVV/(bS*bP) + self.wPP/(bS*bV) + 1/(bS*bV*bP)) / det_num
        L_EP = (-self.wEP*self.wSS*self.wVV + self.wEP*self.wSV*self.wVS + self.wES*self.wSP*self.wVV - self.wES*self.wSV*self.wVP - self.wEV*self.wSP*self.wVS + self.wEV*self.wSS*self.wVP - self.wEP*self.wSS/bV + self.wES*self.wSP/bV - self.wEP*self.wVV/bS + self.wEV*self.wVP/bS - self.wEP/(bS*bV)) / det_num
        # L_EP = (self.wEP*self.wSV*self.wVS - self.wES*self.wSV*self.wVP + self.wES*self.wSP/bV - self.wEP/(bS*bV)) / det_num
        L_ES = (self.wEP*self.wPS*self.wVV - self.wEP*self.wPV*self.wVS - self.wES*self.wPP*self.wVV + self.wES*self.wPV*self.wVP + self.wEV*self.wPP*self.wVS - self.wEV*self.wPS*self.wVP - self.wES*self.wVV/bP + self.wEV*self.wVS/bP + self.wEP*self.wPS/bV - self.wES*self.wPP/bV - self.wES/(bV*bP)) / det_num
        L_EV = (-self.wEP*self.wPS*self.wSV + self.wEP*self.wPV*self.wSS + self.wES*self.wPP*self.wSV - self.wES*self.wPV*self.wSP - self.wEV*self.wPP*self.wSS + self.wEV*self.wPS*self.wSP + self.wES*self.wSV/bP - self.wEV*self.wSS/bP + self.wEP*self.wPV/bS - self.wEV*self.wPP/bS - self.wEV/(bS*bP)) / det_num
        L_PE = (self.wPE*self.wSS*self.wVV - self.wPE*self.wSV*self.wVS - self.wPS*self.wSE*self.wVV + self.wPS*self.wSV*self.wVE + self.wPV*self.wSE*self.wVS - self.wPV*self.wSS*self.wVE + self.wPE*self.wSS/bV - self.wPS*self.wSE/bV + self.wPE*self.wVV/bS - self.wPV*self.wVE/bS + self.wPE/(bS*bV)) / det_num
        L_PP = (-self.wEE*self.wSS*self.wVV + self.wEE*self.wSV*self.wVS + self.wES*self.wSE*self.wVV - self.wES*self.wSV*self.wVE - self.wEV*self.wSE*self.wVS + self.wEV*self.wSS*self.wVE - self.wEE*self.wSS/bV + self.wES*self.wSE/bV - self.wEE*self.wVV/bS + self.wEV*self.wVE/bS - self.wEE/(bS*bV) + self.wSS*self.wVV/bE - self.wSV*self.wVS/bE + self.wSS/(bE*bV) + self.wVV/(bE*bS) + 1/(bE*bS*bV)) / det_num
        L_PS = (self.wEE*self.wPS*self.wVV - self.wEE*self.wPV*self.wVS - self.wES*self.wPE*self.wVV + self.wES*self.wPV*self.wVE + self.wEV*self.wPE*self.wVS - self.wEV*self.wPS*self.wVE + self.wEE*self.wPS/bV - self.wES*self.wPE/bV - self.wPS*self.wVV/bE + self.wPV*self.wVS/bE - self.wPS/(bE*bV)) / det_num
        L_PV = (-self.wEE*self.wPS*self.wSV + self.wEE*self.wPV*self.wSS + self.wES*self.wPE*self.wSV - self.wES*self.wPV*self.wSE - self.wEV*self.wPE*self.wSS + self.wEV*self.wPS*self.wSE + self.wEE*self.wPV/bS - self.wEV*self.wPE/bS + self.wPS*self.wSV/bE - self.wPV*self.wSS/bE - self.wPV/(bE*bS)) / det_num
        L_SE = (-self.wPE*self.wSP*self.wVV + self.wPE*self.wSV*self.wVP + self.wPP*self.wSE*self.wVV - self.wPP*self.wSV*self.wVE - self.wPV*self.wSE*self.wVP + self.wPV*self.wSP*self.wVE + self.wSE*self.wVV/bP - self.wSV*self.wVE/bP - self.wPE*self.wSP/bV + self.wPP*self.wSE/bV + self.wSE/(bV*bP)) / det_num
        L_SP = (self.wEE*self.wSP*self.wVV - self.wEE*self.wSV*self.wVP - self.wEP*self.wSE*self.wVV + self.wEP*self.wSV*self.wVE + self.wEV*self.wSE*self.wVP - self.wEV*self.wSP*self.wVE + self.wEE*self.wSP/bV - self.wEP*self.wSE/bV - self.wSP*self.wVV/bE + self.wSV*self.wVP/bE - self.wSP/(bE*bV)) / det_num
        L_SS = (-self.wEE*self.wPP*self.wVV + self.wEE*self.wPV*self.wVP + self.wEP*self.wPE*self.wVV - self.wEP*self.wPV*self.wVE - self.wEV*self.wPE*self.wVP + self.wEV*self.wPP*self.wVE - self.wEE*self.wVV/bP + self.wEV*self.wVE/bP - self.wEE*self.wPP/bV + self.wEP*self.wPE/bV - self.wEE/(bV*bP) + self.wPP*self.wVV/bE - self.wPV*self.wVP/bE + self.wVV/(bE*bP) + self.wPP/(bE*bV) + 1/(bE*bV*bP)) / det_num
        L_SV = (self.wEE*self.wPP*self.wSV - self.wEE*self.wPV*self.wSP - self.wEP*self.wPE*self.wSV + self.wEP*self.wPV*self.wSE + self.wEV*self.wPE*self.wSP - self.wEV*self.wPP*self.wSE + self.wEE*self.wSV/bP - self.wEV*self.wSE/bP - self.wPP*self.wSV/bE + self.wPV*self.wSP/bE - self.wSV/(bE*bP)) / det_num
        L_VE = (self.wPE*self.wSP*self.wVS - self.wPE*self.wSS*self.wVP - self.wPP*self.wSE*self.wVS + self.wPP*self.wSS*self.wVE + self.wPS*self.wSE*self.wVP - self.wPS*self.wSP*self.wVE - self.wSE*self.wVS/bP + self.wSS*self.wVE/bP - self.wPE*self.wVP/bS + self.wPP*self.wVE/bS + self.wVE/(bS*bP)) / det_num
        L_VP = (-self.wEE*self.wSP*self.wVS + self.wEE*self.wSS*self.wVP + self.wEP*self.wSE*self.wVS - self.wEP*self.wSS*self.wVE - self.wES*self.wSE*self.wVP + self.wES*self.wSP*self.wVE + self.wEE*self.wVP/bS - self.wEP*self.wVE/bS + self.wSP*self.wVS/bE - self.wSS*self.wVP/bE - self.wVP/(bE*bS)) / det_num
        L_VS = (self.wEE*self.wPP*self.wVS - self.wEE*self.wPS*self.wVP - self.wEP*self.wPE*self.wVS + self.wEP*self.wPS*self.wVE + self.wES*self.wPE*self.wVP - self.wES*self.wPP*self.wVE + self.wEE*self.wVS/bP - self.wES*self.wVE/bP - self.wPP*self.wVS/bE + self.wPS*self.wVP/bE - self.wVS/(bE*bP)) / det_num
        L_VV = (-self.wEE*self.wPP*self.wSS + self.wEE*self.wPS*self.wSP + self.wEP*self.wPE*self.wSS - self.wEP*self.wPS*self.wSE - self.wES*self.wPE*self.wSP + self.wES*self.wPP*self.wSE - self.wEE*self.wSS/bP + self.wES*self.wSE/bP - self.wEE*self.wPP/bS + self.wEP*self.wPE/bS - self.wEE/(bS*bP) + self.wPP*self.wSS/bE - self.wPS*self.wSP/bE + self.wSS/(bE*bP) + self.wPP/(bE*bS) + 1/(bE*bS*bP)) / det_num

        # b = (self.wEP*self.wSV*self.wVS - self.wES*self.wSV*self.wVP - self.wEP/(bS*bV))
        # a = self.wES
        # c = (self.wEE - 1/bE)*self.wPS - self.wES*self.wEP
        # d = (self.wEE*self.wPP*self.wSV*self.wVS
        #     - self.wEE*self.wPS*self.wSV*self.wVP
        #     - self.wEP*self.wPE*self.wSV*self.wVS
        #     + self.wEP*self.wPS*self.wSV*self.wVE
        #     + self.wES*self.wPE*self.wSV*self.wVP
        #     - self.wES*self.wPP*self.wSV*self.wVE
        #     - self.wEP*self.wPS*self.wSE/bV
        #     + self.wES*self.wPP*self.wSE/bV
        #     - self.wEE*self.wPP/(bS*bV)
        #     + self.wEP*self.wPE/(bS*bV)
        #     + self.wEE*self.wSV*self.wVS/bP
        #     - self.wES*self.wSV*self.wVE/bP
        #     + self.wES*self.wSE/(bP*bV)
        #     - self.wEE/(bP*bS*bV)
        #     - self.wPP*self.wSV*self.wVS/bE
        #     + self.wPS*self.wSV*self.wVP/bE
        #     + self.wPP/(bE*bS*bV)
        #     - self.wSV*self.wVS/(bE*bP)
        #     + 1/(bE*bP*bS*bV)
        # )

        # analyze_mobius(a, b, c, d, kmin=0, kmax=1)
        if abs(det_num) < 1e-10:
            return dict(
                gain=np.nan, max_ev_real=np.nan, max_ev_imag=np.nan,
                all_L=np.nan,
                det=det_num, J=None, singular=True
            )

        # Jacobian with current b*
        J = np.array([
            [ self.wEE - 1/bE,     -self.wEP,          -self.wES,           -self.wEV ],
            [ self.wPE,            -self.wPP - 1/bP,   -self.wPS,           -self.wPV ],
            [ self.wSE,            -self.wSP,          -self.wSS - 1/bS,    -self.wSV ],
            [ self.wVE,            -self.wVP,          -self.wVS,           -self.wVV - 1/bV ]
        ], dtype=float)

        gain = L_EE * self.I_stim_E + L_EP * self.I_stim_P + L_ES * self.I_stim_S + L_EV * self.I_stim_V
        # print(+ self.wEP*self.wSV*self.wVS, - self.wES*self.wSV*self.wVP, + self.wES*self.wSP * (1/bV), - self.wEP/(bS*bV))
        # print(L_EP)
        # print(det_num)
        # print(gain)

        all_L = (L_EE, L_EP, L_ES, L_EV, L_PE, L_PP, L_PS, L_PV,
                 L_SE, L_SP, L_SS, L_SV, L_VE, L_VP, L_VS, L_VV)

        all_b = (bE, bP, bS, bV) 

        # print(( + self.wEP*self.wSV*self.wVS, - self.wES*self.wSV*self.wVP,  + self.wES*self.wSP/bV, - self.wEP/(bS*bV)))
        # print(+ self.wES, self.wSP, 1/bV, "\t", - self.wES, self.wSV, self.wVP)
        # if self.wES > (self.wEP*self.wPS)/(self.wPP + 1/bP):
        #     print("DisNet \t", self.wES, (self.wEP*self.wPS)/(self.wPP + 1/bP)) #dis VS inh
        # else:
        #     print("InhNet \t", self.wES, (self.wEP*self.wPS)/(self.wPP + 1/bP)) #dis VS inh

        # if self.wEE > 1/bE:
        #     print("ISN \t", self.wEE , 1/bE) #dis VS inh
        # else:
        #     print("non-ISN \t", self.wEE, 1/bE) #dis VS inh

        try:
            # evs = np.linalg.eigvals(J)
            # evs = np.linalg.eig(J)
            # leading = evs[np.argmax(evs.real)]
            # max_ev_real = float(np.max(np.real(leading)))
            # max_ev_imag = float(np.max(np.imag(leading)))
            # print(J)
            evs = eig(J)[0]
            # print(evs)
            max_ev_real = float(np.max(np.real(evs)))
            # print(max_ev_real)
            max_ev_imag = float(np.max(np.imag(evs)))
            # print(max_ev_real)
        except Exception:
            max_ev_real = np.nan
            max_ev_imag = np.nan

        return dict(
            gain=float(gain),
            max_ev_real=max_ev_real,
            max_ev_imag=max_ev_imag,
            all_L= all_L,
            all_b= all_b,
            det=float(det_num), J=J, singular=False
        )
        
    def _assign_weights(self, updates: dict):
        """
        In-place assign given weights in both attributes and the W array.
        """
        # update attributes
        for k, v in updates.items():
            setattr(self, k, float(v))
        # update W from attributes in the canonical order
        vals = [getattr(self, name) for name in self._w_order]
        self.W = np.array(vals, dtype=float).reshape(4, 4)
        
    def calculate_dynamics_general(
        self,
        mode="rates",
        grid_search=False,
        w1_name=None,
        w1_vals=None,
        w2_name=None,
        w2_vals=None
    ):
        """
        Unified dynamics computation across either firing-rate or weight parameter sweeps.

        Parameters
        ----------
        mode : {'rates', 'weights'}
            Determines what is being varied: rates (rE,rP) or network weights.
        grid_search : bool
            If True, perform a 2D sweep over rates; otherwise compute a single point.
        max_rE, max_rP : float
            Used for 'rates' mode when grid_search=True or as fixed values otherwise.
        w1_name : str
            Weight name to sweep in 'weights' mode.
        w1_vals : array-like
            Values for the weight parameter sweep.
        rE_fixed, rP_fixed : float
            Used for 'weights' mode as fixed operating point.

        Returns
        -------
        out : dict
            Dictionary with fields:
            ['gain','gain_mod','maxEV','maxEV_mod','maxImEV','maxImEV_mod',
            'dE','dP','dS','dV'] and parameter vectors (rE_vec/rP_vec or w1_vals).
        """
        # ---------------------------------------------------------------------
        # 1. Define parameter domain
        # ---------------------------------------------------------------------
        if mode == "rates":
            if grid_search:
                max_rEP = max(self.rE0, self.rP0)
                self.rE_vec = np.arange(max_rEP, 0.5 - self.step_rX, -self.step_rX)
                self.rP_vec = np.arange(0.5, max_rEP + self.step_rX, self.step_rX)
                single_point = False
            else:
                self.rE_vec = np.array([self.rE0])
                self.rP_vec = np.array([self.rP0])
                single_point = True

        elif mode == "weights":
            if w1_name not in self._w_index:
                raise ValueError(f"Invalid weight name {w1_name}.")
            w1_vals = np.asarray(w1_vals, dtype=float)

            if w2_name is None:
                single_point = len(w1_vals) == 1
            else:
                if w2_name not in self._w_index:
                    raise ValueError(f"Invalid weight name {w2_name}.")
                w2_vals = np.asarray(w2_vals, dtype=float)
                single_point = False
        else:
            raise ValueError("mode must be either 'rates' or 'weights'.")

        # ---------------------------------------------------------------------
        # 2. Prepare storage
        # ---------------------------------------------------------------------
        if mode == "rates":
            nE, nP = len(self.rE_vec), len(self.rP_vec)
            shape = (nE, nP)
        else:
            if w2_name is None:
                shape = (len(w1_vals),)
            else:
                shape = (len(w1_vals), len(w2_vals))

        gain         = np.full(shape, np.nan)
        gain_mod     = np.full(shape, np.nan)
        maxEV        = np.full(shape, np.nan)
        maxEV_mod    = np.full(shape, np.nan)
        maxImEV      = np.full(shape, np.nan)
        maxImEV_mod  = np.full(shape, np.nan)
        dE           = np.full(shape, np.nan)
        dP           = np.full(shape, np.nan)
        dS           = np.full(shape, np.nan)
        dV           = np.full(shape, np.nan)
        bE           = np.full(shape, np.nan)
        bP           = np.full(shape, np.nan)
        bS           = np.full(shape, np.nan)
        bV           = np.full(shape, np.nan)

        # ---------------------------------------------------------------------
        # 3. Main loop
        # ---------------------------------------------------------------------
        orig_weights = {name: getattr(self, name) for name in self._w_order}
        rS, rV = self.rS0, self.rV0

        if mode == "rates":
            for i, rE in enumerate(self.rE_vec):
                for j, rP in enumerate(self.rP_vec):
                    res_pre = self._compute_block(mode,rE, rP, rS, rV, add_input=False)
                    if res_pre["singular"]:
                        continue

                    gain[i, j] = res_pre["gain"]
                    maxEV[i, j] = res_pre["max_ev_real"]
                    maxImEV[i, j] = res_pre["max_ev_imag"]

                    # --- Compute VIP deltas ---
                    L = res_pre["all_L"]
                    (L_EE, L_EP, L_ES, L_EV,
                    L_PE, L_PP, L_PS, L_PV,
                    L_SE, L_SP, L_SS, L_SV,
                    L_VE, L_VP, L_VS, L_VV) = L

                    dE[i, j] = L_EE*self.I_mod_E + L_EP*self.I_mod_P + L_ES*self.I_mod_S + L_EV*self.I_mod_V
                    dP[i, j] = L_PE*self.I_mod_E + L_PP*self.I_mod_P + L_PS*self.I_mod_S + L_PV*self.I_mod_V
                    dS[i, j] = L_SE*self.I_mod_E + L_SP*self.I_mod_P + L_SS*self.I_mod_S + L_SV*self.I_mod_V
                    dV[i, j] = L_VE*self.I_mod_E + L_VP*self.I_mod_P + L_VS*self.I_mod_S + L_VV*self.I_mod_V

                    res_post = self._compute_block(mode,rE, rP, rS, rV, dE[i, j], dP[i, j], dS[i, j], dV[i, j], add_input=True)
                    if not res_post["singular"]:
                        gain_mod[i, j]    = res_post["gain"]
                        maxEV_mod[i, j]   = res_post["max_ev_real"]
                        maxImEV_mod[i, j] = res_post["max_ev_imag"]

                if not single_point:
                    perc = np.ceil(100 * i / len(self.rE_vec))
                    if not perc % 5:
                        print(f"Completed: {perc}%", end="\r")

        else:  # mode == 'weights'
            rE_fixed = self.rE0
            rP_fixed = self.rP0

            if w2_name is None:
                for i, w1 in enumerate(w1_vals):
                    self._assign_weights({w1_name: w1})
                    res = self._compute_block(mode, rE_fixed, rP_fixed, self.rS0, self.rV0)

                    if res["singular"]:
                        continue

                    gain[i]    = res["gain"]
                    maxEV[i]   = res["max_ev_real"]
                    maxImEV[i] = res["max_ev_imag"]
                    bE[i], bP[i], bS[i], bV[i] = res["all_b"]

            else:
                for i, w1 in enumerate(w1_vals):
                    for j, w2 in enumerate(w2_vals):
                        self._assign_weights({w1_name: w1, w2_name: w2})
                        res = self._compute_block(mode, rE_fixed, rP_fixed, self.rS0, self.rV0)

                        if res["singular"]:
                            continue

                        gain[i, j]    = res["gain"]
                        maxEV[i, j]   = res["max_ev_real"]
                        maxImEV[i, j] = res["max_ev_imag"]
                        bE[i, j], bP[i, j], bS[i, j], bV[i, j] = res["all_b"]

            self._assign_weights(orig_weights)
            all_b = dict(bE=bE, bP=bP, bS=bS, bV=bV)

        # ---------------------------------------------------------------------
        # 4. Assemble result
        # ---------------------------------------------------------------------
        if mode == "rates" and single_point:
            return dict(
                gain=gain[0, 0], gain_mod=gain_mod[0, 0],
                maxEV=maxEV[0, 0], maxEV_mod=maxEV_mod[0, 0],
                maxImEV=maxImEV[0, 0], maxImEV_mod=maxImEV_mod[0, 0],
                dE=dE[0, 0], dP=dP[0, 0], dS=dS[0, 0], dV=dV[0, 0]
            )
        elif mode == "rates":
            return dict(
                rE_vec=self.rE_vec, rP_vec=self.rP_vec,
                gain=gain, gain_mod=gain_mod,
                maxEV=maxEV, maxEV_mod=maxEV_mod,
                maxImEV=maxImEV, maxImEV_mod=maxImEV_mod,
                dE=dE, dP=dP, dS=dS, dV=dV
            )
        else:
            out = dict(
                gain=gain,
                maxEV=maxEV,
                maxImEV=maxImEV,
                all_b=all_b
            )
            out["w1_vals"] = w1_vals
            if w2_name is not None:
                out["w2_vals"] = w2_vals
            return out

    def plot_heatmaps_with_arrows_and_scatter(self, case_name, folder_name, iso_stab, iso_gain, step_arrows=100, nr_points=1000):
        """Plot heatmaps with arrow overlays and scatter plots for a given case"""

        # Calculate dynamics
        res = self.calculate_dynamics_general(mode="rates", grid_search=True)

        gain_num        = res["gain"]
        gain_num_mod    = res["gain_mod"]
        maxEVs_num      = res["maxEV"]
        maxEVs_num_mod  = res["maxEV_mod"]
        maxImEVs_num    = res["maxImEV"]
        maxImEVs_num_mod= res["maxImEV_mod"]
        modV_rE_num     = res["dE"]
        modV_rP_num     = res["dP"]
        modV_rS_num     = res["dS"]
        modV_rV_num     = res["dV"]
        # Create stability mask
        mask_threshold = -0.05
        maxEVs_num_set0 = np.minimum(maxEVs_num, 0)
        mask01 = np.copy(maxEVs_num)
        mask01[mask01 > mask_threshold] = 0
        mask01[mask01 < mask_threshold] = 1

        # Normalize stability
        norm_maxEVs_num_set0 = mask01 * maxEVs_num_set0
        min_val_EV = np.nanmin(norm_maxEVs_num_set0[norm_maxEVs_num_set0 != 0])
        if min_val_EV != 0:
            norm_maxEVs_num_set0 = norm_maxEVs_num_set0 / min_val_EV

        max_val_gain = np.nanmax(gain_num[mask01 != 0])
        min_val_gain = np.nanmin(gain_num[mask01 != 0])

        norm_gain_num_masked = mask01 * ((gain_num - min_val_gain) / (max_val_gain - min_val_gain))

        # Create coordinate matrices
        rE_mat, rP_mat = np.meshgrid(self.rE_vec, self.rP_vec, indexing='ij')

        # Normalize arrow directions
        masked_modV_rP = mask01 * modV_rP_num
        masked_modV_rE = mask01 * modV_rE_num

        # Avoid division by zero
        magnitude = np.sqrt(masked_modV_rP**2 + masked_modV_rE**2)
        magnitude[magnitude == 0] = 1

        norm_masked_modV_rP = masked_modV_rP / magnitude
        norm_masked_modV_rE = masked_modV_rE / magnitude

        # Set arrows to zero where mask is zero
        norm_masked_modV_rP[mask01 == 0] = 0
        norm_masked_modV_rE[mask01 == 0] = 0

        # Calculate differences for scatter plots
        maxEVs_num_set0_2 = np.minimum(maxEVs_num, 0)
        mask01_2 = maxEVs_num.copy()
        mask01_2[mask01_2 > mask_threshold] = np.nan
        mask01_2[mask01_2 < mask_threshold] = 1

        maxEVs_num_set0_2_mod = np.minimum(maxEVs_num_mod, 0)
        mask01_2_mod = maxEVs_num_mod.copy()
        mask01_2_mod[mask01_2_mod > mask_threshold] = np.nan
        mask01_2_mod[mask01_2_mod < mask_threshold] = 1

        diff_maxEVs_num_set0_2_mod = (mask01_2 * maxEVs_num_set0_2 -
                                       mask01_2_mod * maxEVs_num_set0_2_mod)

        gain_num_set0_2 = mask01_2 * gain_num
        gain_num_set0_2_mod = mask01_2_mod * gain_num_mod
        diff_gain_num_set0_2_mod = gain_num_set0_2_mod - gain_num_set0_2

        # Plot random subset of points
        subset_indices = np.random.choice(diff_maxEVs_num_set0_2_mod.size, nr_points, replace=False)
        subset_i, subset_j = np.unravel_index(subset_indices, diff_maxEVs_num_set0_2_mod.shape)

        # print(oscMetric_num[distEVs_num == 0])
    #===============================================================================================
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()
        fig.suptitle(f'{case_name} case')

    ##### Plot 1: Gain with arrows
        im1 = ax1.imshow(norm_gain_num_masked, cmap=self.cmap, aspect='auto',
                         extent=[self.rP_vec[0], self.rP_vec[-1], self.rE_vec[-1], self.rE_vec[0]])

        # Add arrows to gain plot
        ax1.quiver(rP_mat[::step_arrows, ::step_arrows],
                   rE_mat[::step_arrows, ::step_arrows],
                   norm_masked_modV_rP[::step_arrows, ::step_arrows],
                   norm_masked_modV_rE[::step_arrows, ::step_arrows],
                   angles='xy', color='black', alpha=0.7)

        # Plot gain isolines
        cont1 = ax1.contour(rP_mat, rE_mat, norm_gain_num_masked, levels=iso_gain, linewidths=1.5)

        ax1.set_title(f'Gain (normalized)')
        ax1.set_xlabel('rP (Hz)')
        ax1.set_ylabel('rE (Hz)')

    ##### Plot 2: Stability with arrows
        im2 = ax2.imshow(norm_maxEVs_num_set0, cmap=self.cmap, aspect='auto',
                         extent=[self.rP_vec[0], self.rP_vec[-1], self.rE_vec[-1], self.rE_vec[0]])

        # Add arrows to stability plot
        ax2.quiver(rP_mat[::step_arrows, ::step_arrows],
                   rE_mat[::step_arrows, ::step_arrows],
                   norm_masked_modV_rP[::step_arrows, ::step_arrows],
                   norm_masked_modV_rE[::step_arrows, ::step_arrows],
                   angles='xy', color='black', alpha=0.7)

        # Plot stability isolines
        cont2 = ax2.contour(rP_mat, rE_mat, norm_maxEVs_num_set0, levels=iso_stab, linewidths=1.5)

        ax2.set_title(f'Stability (normalized)')
        ax2.set_xlabel('rP (Hz)')
        ax2.set_ylabel('rE (Hz)')

    ##### Plot 3: Scatter plot for deltas of gain and stability
        for i, j in zip(subset_i, subset_j):
            if not (np.isnan(diff_maxEVs_num_set0_2_mod[i, j]) or np.isnan(diff_gain_num_set0_2_mod[i, j])):
                ax3.scatter(diff_maxEVs_num_set0_2_mod[i, j], diff_gain_num_set0_2_mod[i, j],
                          s=1, c='k', alpha=0.5)

        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax3.set_title('mod VIP')
        ax3.set_xlabel(r'$\Delta$ stability')
        ax3.set_ylabel(r'$\Delta$ gain')
        ax3.set_xlim([-0.5, 0.5])
        ax3.set_ylim([-4, 4])


    ##### Plot 4: Info about parameters
        matrix_str = f'Random Points: {nr_points}\n\n' + f'SST init. rate: {self.rS0}\nVIP init. rate: {self.rV0}\n\n' + f'Stimulation E: {self.I_stim_E}\nStimulation PV: {self.I_stim_P}\nModulation VIP: {self.I_mod_V}\n\n' + 'Weight Matrix:\n' + '\n'.join(['  '.join([f'{val:.2f}' for val in row]) for row in self.W])
        ax4.axis('off')  # turn off axes
        ax4.text(0.5, 0.5, matrix_str, fontsize=12, ha='center', va='center', family='monospace')

        return fig


    def simulate_dynamics(self, tauE=10, tauP=10, tauS=10, tauV=10,
                        dt=0.01, end_sim=750,
                        stim_times=(50, 350)):
        # Initialize rates
        rE, rP, rS, rV = self.rE0, self.rP0, self.rS0, self.rV0

        # Compute baseline x*
        xE0 = (rE / self.mult_f)**(1 / self.power) - (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV)
        xP0 = (rP / self.mult_f)**(1 / self.power) - (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV)
        xS0 = (rS / self.mult_f)**(1 / self.power) - (self.wSE*rE - self.wSP*rP - self.wSS*rS - self.wSV*rV)
        xV0 = (rV / self.mult_f)**(1 / self.power) - (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV)
        

        # Initialize dynamic variables
        time = np.arange(0, end_sim + dt, dt)
        rE_trace = np.zeros_like(time)
        rP_trace = np.zeros_like(time)
        rS_trace = np.zeros_like(time)
        rV_trace = np.zeros_like(time)

        # Initial conditions
        rE_trace[0], rP_trace[0], rS_trace[0], rV_trace[0] = rE, rP, rS, rV
        xE_fb, xP_fb, xS_fb, xV_fb = 0.0, 0.0, 0.0, 0.0
        xE_ff, xP_ff, xS_ff, xV_ff = 0.0, 0.0, 0.0, 0.0

        # Simulation loop
        for i in range(1, len(time)):
            t = time[i]

            # Modulation onset
            if np.isclose(t, stim_times[0], atol=dt/2):
                xE_fb = self.I_mod_E
                xP_fb = self.I_mod_P
                xS_fb = self.I_mod_S
                xV_fb = self.I_mod_V

            # Stimulation onset (E,P)
            if np.isclose(t, stim_times[1], atol=dt/2):
                xE_ff = self.I_stim_E
                xP_ff = self.I_stim_P
                xS_ff = self.I_stim_S
                xV_ff = self.I_stim_V

            # Update via Euler method
            rE_dot = (-rE + self.mult_f * (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV + xE0 + xE_ff + xE_fb)**self.power) / tauE
            rP_dot = (-rP + self.mult_f * (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV + xP0 + xP_ff + xP_fb)**self.power) / tauP
            rS_dot = (-rS + self.mult_f * (self.wSE*rE - self.wSP*rP - self.wSS*rS - self.wSV*rV + xS0 + xS_ff + xS_fb)**self.power) / tauS
            rV_dot = (-rV + self.mult_f * (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV + xV0 + xV_ff + xV_fb)**self.power) / tauV

            rE += dt * rE_dot
            rP += dt * rP_dot
            rS += dt * rS_dot
            rV += dt * rV_dot

            rE = max(rE, 0.0) #rectify
            rP = max(rP, 0.0)
            rS = max(rS, 0.0)
            rV = max(rV, 0.0)

            # Store
            rE_trace[i] = rE
            rP_trace[i] = rP
            rS_trace[i] = rS
            rV_trace[i] = rV

        # Return traces
        return dict(time=time, rE=rE_trace, rP=rP_trace, rS=rS_trace, rV=rV_trace)





class VIP_circuit_double_plasticity:
    def __init__(self, W, rE0=3.0, rP0=5.0, rS0=2.0, rV0=2.0, I_stim_E=0.3, I_stim_P=0.3, I_stim_S=0.0, I_stim_V=0.0, I_mod_E=0.0,I_mod_P=0.0,I_mod_S=0.0,I_mod_V=0.0, step_rX=0.01, power=1, mult_f=1):
        # Define parameters
        self.W = W
        
        self.wEE, self.wEP, self.wES, self.wEV, self.wPE, self.wPP, self.wPS, self.wPV, self.wSE, self.wSP, self.wSS, self.wSV, self.wVE, self.wVP, self.wVS, self.wVV = W.flatten()

        self.rE0 = rE0 #for weight sweep set-point
        self.rP0 = rP0
        self.rS0 = rS0  # SST initial firing rate
        self.rV0 = rV0  # VIP initial firing rate
        self.step_rX = step_rX

        self.I_stim_E = I_stim_E
        self.I_stim_P = I_stim_P
        self.I_stim_S = I_stim_S
        self.I_stim_V = I_stim_V
        self.I_mod_E = I_mod_E
        self.I_mod_P = I_mod_P
        self.I_mod_S = I_mod_S
        self.I_mod_V = I_mod_V

        # I/O function parameters
        self.power = power
        self.mult_f = mult_f

        # --- STP params (analytical) ---
        self.tau_d = 0.100    # STD recovery (ms)
        self.U_d   = 1     # STD utilization

        self.tau_f = 0.400    # STF recovery (ms)
        self.U_f     = 1     # STF baseline utilization
        self.U_max = 2.0     # STF upper cap


        # Define colormap
        self.hex_colors = ['#ffffff', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d']
        self.colors = [[int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5)] for hex_color in self.hex_colors]

        # Create custom colormap
        self.cmap = LinearSegmentedColormap.from_list('custom', self.colors, N=128)
        # self.cmap = 'plasma'
            # --- Add this helper mapping once in the class body ---
        self._w_order = [
        "wEE","wEP","wES","wEV",
        "wPE","wPP","wPS","wPV",
        "wSE","wSP","wSS","wSV",
        "wVE","wVP","wVS","wVV"
        ]
        self._w_index = {name:i for i, name in enumerate(self._w_order)}


    def _compute_block(self,
                    mode: str,
                    rE: float,
                    rP: float,
                    rS: float,
                    rV: float,
                    dE: float = 0.0,
                    dP: float = 0.0,
                    dS: float = 0.0,
                    dV: float = 0.0,
                    add_input: bool = False,):
        """
        General computation block used for both PRE and POST VIP cases.

        Parameters
        ----------
        rE, rP : float
            E and P rates at the grid point.
        dE, dP, dS, dV : float, optional
            Additive deltas to E,P,S,V (use zeros for PRE; use VIP-induced deltas for POST).
        add_v_input : bool, optional
            If True (POST), include +I_mod_V inside the V nonlinearity (matches your bV_mod).

        Returns
        -------
        out : dict
            {
            'gain': float,                       # L_EE * I_stim_E + L_EP * I_stim_P
            'max_ev_real': float,                # max Re eig(J)
            'max_ev_imag': float,                # max Im eig(J)
            'L_EV': float, 'L_PV': float,        # linear-response entries with 'V' column
            'L_SV': float, 'L_VV': float,
            'det': float,                        # determinant-like scalar
            'J': np.ndarray or None,             # Jacobian (None if singular)
            'singular': bool                     # True if |det| < 1e-10
            }
        Notes
        -----
        PRE:  call with d*=0, add_v_input=False.
        POST: call with d* = VIP deltas from PRE, add_v_input=True.
        The xE,xP,xS,xV are computed from baseline rS0,rV0 (as in your code),
        and then gains b* use shifted (r* + d*) in the linear terms, with +I_mod_V
        added to V only when add_v_input=True (exactly matching bV_mod).
        """     

        rE_eff = rE + dE
        rP_eff = rP + dP
        rS_eff = rS + dS
        rV_eff = rV + dV

        if add_input:
            add_E, add_P, add_S, add_V = self.I_mod_E, self.I_mod_P, self.I_mod_S, self.I_mod_V
        else:
            add_E = add_P = add_S = add_V = 0.0

        newton_before = False
        if newton_before:
            def residual(r_vec):
                rE_, rP_, rS_, rV_ = r_vec
                pSP = 1 / (1 + self.U_d * rP_ * self.tau_d)
                pSV = (1 + self.U_f * self.U_max * rS_ * self.tau_f) / (1 + self.U_f * rS_ * self.tau_f)

                xE = (rE_ / self.mult_f)**(1 / self.power) - (self.wEE*rE_ - self.wEP*rP_ - self.wES*rS_ - self.wEV*rV_)
                xP = (rP_ / self.mult_f)**(1 / self.power) - (self.wPE*rE_ - self.wPP*rP_ - self.wPS*rS_ - self.wPV*rV_)
                xS = (rS_ / self.mult_f)**(1 / self.power) - (self.wSE*rE_ - pSP*self.wSP*rP_ - self.wSS*rS_ - pSV*self.wSV*rV_)
                xV = (rV_ / self.mult_f)**(1 / self.power) - (self.wVE*rE_ - self.wVP*rP_ - self.wVS*rS_ - self.wVV*rV_)

                gE = self.wEE*rE_ - self.wEP*rP_ - self.wES*rS_ - self.wEV*rV_ + xE + add_E
                gP = self.wPE*rE_ - self.wPP*rP_ - self.wPS*rS_ - self.wPV*rV_ + xP + add_P
                gS = self.wSE*rE_ - pSP*self.wSP*rP_ - self.wSS*rS_ - pSV*self.wSV*rV_ + xS + add_S
                gV = self.wVE*rE_ - self.wVP*rP_ - self.wVS*rS_ - self.wVV*rV_ + xV + add_V

                phiE = self.mult_f * max(gE, 0)**self.power
                phiP = self.mult_f * max(gP, 0)**self.power
                phiS = self.mult_f * max(gS, 0)**self.power
                phiV = self.mult_f * max(gV, 0)**self.power

                return np.array([rE_ - phiE, rP_ - phiP, rS_ - phiS, rV_ - phiV])

            r_vec = np.array([rE_eff, rP_eff, rS_eff, rV_eff])
            for _ in range(10):  # a few Newton iterations to find steady-state
                F = residual(r_vec)
                if np.linalg.norm(F) < 1e-9:
                    break
                # Finite-difference Jacobian
                J_fd = np.zeros((4, 4))
                eps = 1e-6
                for j in range(4):
                    r_pert = r_vec.copy()
                    r_pert[j] += eps
                    J_fd[:, j] = (residual(r_pert) - F) / eps
                try:
                    step = np.linalg.solve(J_fd, F)
                except np.linalg.LinAlgError:
                    break
                r_vec -= step
                r_vec = np.clip(r_vec, 0, None)

            rE_eff, rP_eff, rS_eff, rV_eff = r_vec  # overwrite steady-state values

        pSP = 1 / (1 + self.U_d * rP_eff * self.tau_d)
        pSV = (1 + self.U_f * self.U_max * rV_eff * self.tau_f)/(1 + self.U_f * rV_eff * self.tau_f)
        DpSP = - pSP**2 * self.U_d * self.tau_d
        DpSV = (self.U_f * (self.U_max-1) * self.tau_f) / ( (1 + self.U_f * rV_eff * self.tau_f)**2 )
        # pSV = 1
        # DpSV = 0

        xE = (rE / self.mult_f)**(1 / self.power) - (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV)
        xP = (rP / self.mult_f)**(1 / self.power) - (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV)
        xS = (rS / self.mult_f)**(1 / self.power) - (self.wSE*rE - pSP*self.wSP*rP - self.wSS*rS - pSV*self.wSV*rV)
        xV = (rV / self.mult_f)**(1 / self.power) - (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV)


        if mode == "rates":
            bE = self.power * self.mult_f * (self.wEE*rE_eff - self.wEP*rP_eff - self.wES*rS_eff - self.wEV*rV_eff + xE + add_E)**(self.power - 1)
            bP = self.power * self.mult_f * (self.wPE*rE_eff - self.wPP*rP_eff - self.wPS*rS_eff - self.wPV*rV_eff + xP + add_P)**(self.power - 1)
            bS = self.power * self.mult_f * (self.wSE*rE_eff - pSP*self.wSP*rP_eff - self.wSS*rS_eff - pSV*self.wSV*rV_eff + xS + add_S)**(self.power - 1)
            bV = self.power * self.mult_f * (self.wVE*rE_eff - self.wVP*rP_eff - self.wVS*rS_eff - self.wVV*rV_eff + xV + add_V)**(self.power - 1)
            # print(self.wSE*rE_eff, - self.wSP*rP_eff, - self.wSV*rV_eff, + xS)
            # print(rP_eff, rS_eff)
        elif mode == "weights":
            bE = self.power * self.mult_f * ((rE + self.I_mod_E)/self.mult_f)**((self.power - 1)/self.power)
            bP = self.power * self.mult_f * ((rP + self.I_mod_P)/self.mult_f)**((self.power - 1)/self.power)
            bS = self.power * self.mult_f * ((rS + self.I_mod_S)/self.mult_f)**((self.power - 1)/self.power)
            bV = self.power * self.mult_f * ((rV + self.I_mod_V)/self.mult_f)**((self.power - 1)/self.power)
    

        det_num = (DpSP*bE*bP*bS*bV*rP*self.wEE*self.wPS*self.wSP*self.wVV - DpSP*bE*bP*bS*bV*rP*self.wEE*self.wPV*self.wSP*self.wVS - DpSP*bE*bP*bS*bV*rP*self.wES*self.wPE*self.wSP*self.wVV + DpSP*bE*bP*bS*bV*rP*self.wES*self.wPV*self.wSP*self.wVE + DpSP*bE*bP*bS*bV*rP*self.wEV*self.wPE*self.wSP*self.wVS - DpSP*bE*bP*bS*bV*rP*self.wEV*self.wPS*self.wSP*self.wVE + DpSP*bE*bP*bS*rP*self.wEE*self.wPS*self.wSP - DpSP*bE*bP*bS*rP*self.wES*self.wPE*self.wSP - DpSP*bP*bS*bV*rP*self.wPS*self.wSP*self.wVV + DpSP*bP*bS*bV*rP*self.wPV*self.wSP*self.wVS - DpSP*bP*bS*rP*self.wPS*self.wSP + DpSV*bE*bP*bS*bV*rV*self.wEE*self.wPP*self.wSV*self.wVS - DpSV*bE*bP*bS*bV*rV*self.wEE*self.wPS*self.wSV*self.wVP - DpSV*bE*bP*bS*bV*rV*self.wEP*self.wPE*self.wSV*self.wVS + DpSV*bE*bP*bS*bV*rV*self.wEP*self.wPS*self.wSV*self.wVE + DpSV*bE*bP*bS*bV*rV*self.wES*self.wPE*self.wSV*self.wVP - DpSV*bE*bP*bS*bV*rV*self.wES*self.wPP*self.wSV*self.wVE + DpSV*bE*bS*bV*rV*self.wEE*self.wSV*self.wVS - DpSV*bE*bS*bV*rV*self.wES*self.wSV*self.wVE - DpSV*bP*bS*bV*rV*self.wPP*self.wSV*self.wVS + DpSV*bP*bS*bV*rV*self.wPS*self.wSV*self.wVP - DpSV*bS*bV*rV*self.wSV*self.wVS + bE*bP*bS*bV*pSP*self.wEE*self.wPS*self.wSP*self.wVV - bE*bP*bS*bV*pSP*self.wEE*self.wPV*self.wSP*self.wVS - bE*bP*bS*bV*pSP*self.wES*self.wPE*self.wSP*self.wVV + bE*bP*bS*bV*pSP*self.wES*self.wPV*self.wSP*self.wVE + bE*bP*bS*bV*pSP*self.wEV*self.wPE*self.wSP*self.wVS - bE*bP*bS*bV*pSP*self.wEV*self.wPS*self.wSP*self.wVE + bE*bP*bS*bV*pSV*self.wEE*self.wPP*self.wSV*self.wVS - bE*bP*bS*bV*pSV*self.wEE*self.wPS*self.wSV*self.wVP - bE*bP*bS*bV*pSV*self.wEP*self.wPE*self.wSV*self.wVS + bE*bP*bS*bV*pSV*self.wEP*self.wPS*self.wSV*self.wVE + bE*bP*bS*bV*pSV*self.wES*self.wPE*self.wSV*self.wVP - bE*bP*bS*bV*pSV*self.wES*self.wPP*self.wSV*self.wVE - bE*bP*bS*bV*self.wEE*self.wPP*self.wSS*self.wVV + bE*bP*bS*bV*self.wEE*self.wPV*self.wSS*self.wVP + bE*bP*bS*bV*self.wEP*self.wPE*self.wSS*self.wVV - bE*bP*bS*bV*self.wEP*self.wPS*self.wSE*self.wVV + bE*bP*bS*bV*self.wEP*self.wPV*self.wSE*self.wVS - bE*bP*bS*bV*self.wEP*self.wPV*self.wSS*self.wVE + bE*bP*bS*bV*self.wES*self.wPP*self.wSE*self.wVV - bE*bP*bS*bV*self.wES*self.wPV*self.wSE*self.wVP - bE*bP*bS*bV*self.wEV*self.wPE*self.wSS*self.wVP - bE*bP*bS*bV*self.wEV*self.wPP*self.wSE*self.wVS + bE*bP*bS*bV*self.wEV*self.wPP*self.wSS*self.wVE + bE*bP*bS*bV*self.wEV*self.wPS*self.wSE*self.wVP + bE*bP*bS*pSP*self.wEE*self.wPS*self.wSP - bE*bP*bS*pSP*self.wES*self.wPE*self.wSP - bE*bP*bS*self.wEE*self.wPP*self.wSS + bE*bP*bS*self.wEP*self.wPE*self.wSS - bE*bP*bS*self.wEP*self.wPS*self.wSE + bE*bP*bS*self.wES*self.wPP*self.wSE - bE*bP*bV*self.wEE*self.wPP*self.wVV + bE*bP*bV*self.wEE*self.wPV*self.wVP + bE*bP*bV*self.wEP*self.wPE*self.wVV - bE*bP*bV*self.wEP*self.wPV*self.wVE - bE*bP*bV*self.wEV*self.wPE*self.wVP + bE*bP*bV*self.wEV*self.wPP*self.wVE - bE*bP*self.wEE*self.wPP + bE*bP*self.wEP*self.wPE + bE*bS*bV*pSV*self.wEE*self.wSV*self.wVS - bE*bS*bV*pSV*self.wES*self.wSV*self.wVE - bE*bS*bV*self.wEE*self.wSS*self.wVV + bE*bS*bV*self.wES*self.wSE*self.wVV - bE*bS*bV*self.wEV*self.wSE*self.wVS + bE*bS*bV*self.wEV*self.wSS*self.wVE - bE*bS*self.wEE*self.wSS + bE*bS*self.wES*self.wSE - bE*bV*self.wEE*self.wVV + bE*bV*self.wEV*self.wVE - bE*self.wEE - bP*bS*bV*pSP*self.wPS*self.wSP*self.wVV + bP*bS*bV*pSP*self.wPV*self.wSP*self.wVS - bP*bS*bV*pSV*self.wPP*self.wSV*self.wVS + bP*bS*bV*pSV*self.wPS*self.wSV*self.wVP + bP*bS*bV*self.wPP*self.wSS*self.wVV - bP*bS*bV*self.wPV*self.wSS*self.wVP - bP*bS*pSP*self.wPS*self.wSP + bP*bS*self.wPP*self.wSS + bP*bV*self.wPP*self.wVV - bP*bV*self.wPV*self.wVP + bP*self.wPP - bS*bV*pSV*self.wSV*self.wVS + bS*bV*self.wSS*self.wVV + bS*self.wSS + bV*self.wVV + 1)/(bE*bP*bS*bV)

        L_EE = (-DpSP*rP*self.wPS*self.wSP*self.wVV + DpSP*rP*self.wPV*self.wSP*self.wVS - DpSP*rP*self.wPS*self.wSP/bV - DpSV*rV*self.wPP*self.wSV*self.wVS + DpSV*rV*self.wPS*self.wSV*self.wVP - DpSV*rV*self.wSV*self.wVS/bP - pSP*self.wPS*self.wSP*self.wVV + pSP*self.wPV*self.wSP*self.wVS - pSV*self.wPP*self.wSV*self.wVS + pSV*self.wPS*self.wSV*self.wVP + self.wPP*self.wSS*self.wVV - self.wPV*self.wSS*self.wVP - pSP*self.wPS*self.wSP/bV + self.wPP*self.wSS/bV + self.wPP*self.wVV/bS - self.wPV*self.wVP/bS + self.wPP/(bS*bV) - pSV*self.wSV*self.wVS/bP + self.wSS*self.wVV/bP + self.wSS/(bP*bV) + self.wVV/(bP*bS) + 1/(bP*bS*bV)) / det_num
        L_EP = (DpSP*rP*self.wES*self.wSP*self.wVV - DpSP*rP*self.wEV*self.wSP*self.wVS + DpSP*rP*self.wES*self.wSP/bV + DpSV*rV*self.wEP*self.wSV*self.wVS - DpSV*rV*self.wES*self.wSV*self.wVP + pSP*self.wES*self.wSP*self.wVV - pSP*self.wEV*self.wSP*self.wVS + pSV*self.wEP*self.wSV*self.wVS - pSV*self.wES*self.wSV*self.wVP - self.wEP*self.wSS*self.wVV + self.wEV*self.wSS*self.wVP + pSP*self.wES*self.wSP/bV - self.wEP*self.wSS/bV - self.wEP*self.wVV/bS + self.wEV*self.wVP/bS - self.wEP/(bS*bV)) / det_num
        L_ES = (self.wEP*self.wPS*self.wVV - self.wEP*self.wPV*self.wVS - self.wES*self.wPP*self.wVV + self.wES*self.wPV*self.wVP + self.wEV*self.wPP*self.wVS - self.wEV*self.wPS*self.wVP + self.wEP*self.wPS/bV - self.wES*self.wPP/bV - self.wES*self.wVV/bP + self.wEV*self.wVS/bP - self.wES/(bP*bV)) / det_num
        L_EV = (-DpSP*rP*self.wES*self.wPV*self.wSP + DpSP*rP*self.wEV*self.wPS*self.wSP - DpSV*rV*self.wEP*self.wPS*self.wSV + DpSV*rV*self.wES*self.wPP*self.wSV + DpSV*rV*self.wES*self.wSV/bP - pSP*self.wES*self.wPV*self.wSP + pSP*self.wEV*self.wPS*self.wSP - pSV*self.wEP*self.wPS*self.wSV + pSV*self.wES*self.wPP*self.wSV + self.wEP*self.wPV*self.wSS - self.wEV*self.wPP*self.wSS + self.wEP*self.wPV/bS - self.wEV*self.wPP/bS + pSV*self.wES*self.wSV/bP - self.wEV*self.wSS/bP - self.wEV/(bP*bS)) / det_num
        L_PE = (-DpSV*rV*self.wPE*self.wSV*self.wVS + DpSV*rV*self.wPS*self.wSV*self.wVE - pSV*self.wPE*self.wSV*self.wVS + pSV*self.wPS*self.wSV*self.wVE + self.wPE*self.wSS*self.wVV - self.wPS*self.wSE*self.wVV + self.wPV*self.wSE*self.wVS - self.wPV*self.wSS*self.wVE + self.wPE*self.wSS/bV - self.wPS*self.wSE/bV + self.wPE*self.wVV/bS - self.wPV*self.wVE/bS + self.wPE/(bS*bV)) / det_num
        L_PP = (DpSV*rV*self.wEE*self.wSV*self.wVS - DpSV*rV*self.wES*self.wSV*self.wVE - DpSV*rV*self.wSV*self.wVS/bE + pSV*self.wEE*self.wSV*self.wVS - pSV*self.wES*self.wSV*self.wVE - self.wEE*self.wSS*self.wVV + self.wES*self.wSE*self.wVV - self.wEV*self.wSE*self.wVS + self.wEV*self.wSS*self.wVE - self.wEE*self.wSS/bV + self.wES*self.wSE/bV - self.wEE*self.wVV/bS + self.wEV*self.wVE/bS - self.wEE/(bS*bV) - pSV*self.wSV*self.wVS/bE + self.wSS*self.wVV/bE + self.wSS/(bE*bV) + self.wVV/(bE*bS) + 1/(bE*bS*bV)) / det_num
        L_PS = (self.wEE*self.wPS*self.wVV - self.wEE*self.wPV*self.wVS - self.wES*self.wPE*self.wVV + self.wES*self.wPV*self.wVE + self.wEV*self.wPE*self.wVS - self.wEV*self.wPS*self.wVE + self.wEE*self.wPS/bV - self.wES*self.wPE/bV - self.wPS*self.wVV/bE + self.wPV*self.wVS/bE - self.wPS/(bE*bV)) / det_num
        L_PV = (-DpSV*rV*self.wEE*self.wPS*self.wSV + DpSV*rV*self.wES*self.wPE*self.wSV + DpSV*rV*self.wPS*self.wSV/bE - pSV*self.wEE*self.wPS*self.wSV + pSV*self.wES*self.wPE*self.wSV + self.wEE*self.wPV*self.wSS - self.wES*self.wPV*self.wSE - self.wEV*self.wPE*self.wSS + self.wEV*self.wPS*self.wSE + self.wEE*self.wPV/bS - self.wEV*self.wPE/bS + pSV*self.wPS*self.wSV/bE - self.wPV*self.wSS/bE - self.wPV/(bE*bS)) / det_num
        L_SE = (-DpSP*rP*self.wPE*self.wSP*self.wVV + DpSP*rP*self.wPV*self.wSP*self.wVE - DpSP*rP*self.wPE*self.wSP/bV + DpSV*rV*self.wPE*self.wSV*self.wVP - DpSV*rV*self.wPP*self.wSV*self.wVE - DpSV*rV*self.wSV*self.wVE/bP - pSP*self.wPE*self.wSP*self.wVV + pSP*self.wPV*self.wSP*self.wVE + pSV*self.wPE*self.wSV*self.wVP - pSV*self.wPP*self.wSV*self.wVE + self.wPP*self.wSE*self.wVV - self.wPV*self.wSE*self.wVP - pSP*self.wPE*self.wSP/bV + self.wPP*self.wSE/bV - pSV*self.wSV*self.wVE/bP + self.wSE*self.wVV/bP + self.wSE/(bP*bV)) / det_num
        L_SP = (DpSP*rP*self.wEE*self.wSP*self.wVV - DpSP*rP*self.wEV*self.wSP*self.wVE + DpSP*rP*self.wEE*self.wSP/bV - DpSP*rP*self.wSP*self.wVV/bE - DpSP*rP*self.wSP/(bE*bV) - DpSV*rV*self.wEE*self.wSV*self.wVP + DpSV*rV*self.wEP*self.wSV*self.wVE + DpSV*rV*self.wSV*self.wVP/bE + pSP*self.wEE*self.wSP*self.wVV - pSP*self.wEV*self.wSP*self.wVE - pSV*self.wEE*self.wSV*self.wVP + pSV*self.wEP*self.wSV*self.wVE - self.wEP*self.wSE*self.wVV + self.wEV*self.wSE*self.wVP + pSP*self.wEE*self.wSP/bV - self.wEP*self.wSE/bV - pSP*self.wSP*self.wVV/bE + pSV*self.wSV*self.wVP/bE - pSP*self.wSP/(bE*bV)) / det_num
        L_SS = (-self.wEE*self.wPP*self.wVV + self.wEE*self.wPV*self.wVP + self.wEP*self.wPE*self.wVV - self.wEP*self.wPV*self.wVE - self.wEV*self.wPE*self.wVP + self.wEV*self.wPP*self.wVE - self.wEE*self.wPP/bV + self.wEP*self.wPE/bV - self.wEE*self.wVV/bP + self.wEV*self.wVE/bP - self.wEE/(bP*bV) + self.wPP*self.wVV/bE - self.wPV*self.wVP/bE + self.wPP/(bE*bV) + self.wVV/(bE*bP) + 1/(bE*bP*bV)) / det_num
        L_SV = (-DpSP*rP*self.wEE*self.wPV*self.wSP + DpSP*rP*self.wEV*self.wPE*self.wSP + DpSP*rP*self.wPV*self.wSP/bE + DpSV*rV*self.wEE*self.wPP*self.wSV - DpSV*rV*self.wEP*self.wPE*self.wSV + DpSV*rV*self.wEE*self.wSV/bP - DpSV*rV*self.wPP*self.wSV/bE - DpSV*rV*self.wSV/(bE*bP) - pSP*self.wEE*self.wPV*self.wSP + pSP*self.wEV*self.wPE*self.wSP + pSV*self.wEE*self.wPP*self.wSV - pSV*self.wEP*self.wPE*self.wSV + self.wEP*self.wPV*self.wSE - self.wEV*self.wPP*self.wSE + pSV*self.wEE*self.wSV/bP - self.wEV*self.wSE/bP + pSP*self.wPV*self.wSP/bE - pSV*self.wPP*self.wSV/bE - pSV*self.wSV/(bE*bP)) / det_num
        L_VE = (DpSP*rP*self.wPE*self.wSP*self.wVS - DpSP*rP*self.wPS*self.wSP*self.wVE + pSP*self.wPE*self.wSP*self.wVS - pSP*self.wPS*self.wSP*self.wVE - self.wPE*self.wSS*self.wVP - self.wPP*self.wSE*self.wVS + self.wPP*self.wSS*self.wVE + self.wPS*self.wSE*self.wVP - self.wPE*self.wVP/bS + self.wPP*self.wVE/bS - self.wSE*self.wVS/bP + self.wSS*self.wVE/bP + self.wVE/(bP*bS)) / det_num
        L_VP = (-DpSP*rP*self.wEE*self.wSP*self.wVS + DpSP*rP*self.wES*self.wSP*self.wVE + DpSP*rP*self.wSP*self.wVS/bE - pSP*self.wEE*self.wSP*self.wVS + pSP*self.wES*self.wSP*self.wVE + self.wEE*self.wSS*self.wVP + self.wEP*self.wSE*self.wVS - self.wEP*self.wSS*self.wVE - self.wES*self.wSE*self.wVP + self.wEE*self.wVP/bS - self.wEP*self.wVE/bS + pSP*self.wSP*self.wVS/bE - self.wSS*self.wVP/bE - self.wVP/(bE*bS)) / det_num
        L_VS = (self.wEE*self.wPP*self.wVS - self.wEE*self.wPS*self.wVP - self.wEP*self.wPE*self.wVS + self.wEP*self.wPS*self.wVE + self.wES*self.wPE*self.wVP - self.wES*self.wPP*self.wVE + self.wEE*self.wVS/bP - self.wES*self.wVE/bP - self.wPP*self.wVS/bE + self.wPS*self.wVP/bE - self.wVS/(bE*bP)) / det_num
        L_VV = (DpSP*rP*self.wEE*self.wPS*self.wSP - DpSP*rP*self.wES*self.wPE*self.wSP - DpSP*rP*self.wPS*self.wSP/bE + pSP*self.wEE*self.wPS*self.wSP - pSP*self.wES*self.wPE*self.wSP - self.wEE*self.wPP*self.wSS + self.wEP*self.wPE*self.wSS - self.wEP*self.wPS*self.wSE + self.wES*self.wPP*self.wSE - self.wEE*self.wPP/bS + self.wEP*self.wPE/bS - self.wEE*self.wSS/bP + self.wES*self.wSE/bP - self.wEE/(bP*bS) - pSP*self.wPS*self.wSP/bE + self.wPP*self.wSS/bE + self.wPP/(bE*bS) + self.wSS/(bE*bP) + 1/(bE*bP*bS)) / det_num

        if abs(det_num) < 1e-10:
            return dict(
                gain=np.nan, max_ev_real=np.nan, max_ev_imag=np.nan,
                all_L=np.nan,
                det=det_num, J=None, singular=True
            )

        J = np.array([[self.wEE - 1/bE, -self.wEP, -self.wES, -self.wEV], 
                          [self.wPE, -self.wPP - 1/bP, -self.wPS, -self.wPV], 
                          [self.wSE, DpSP*rP*self.wSP - pSP*self.wSP, -self.wSS - 1/bS, DpSV*rV*self.wSV - pSV*self.wSV], 
                          [self.wVE, -self.wVP, -self.wVS, -self.wVV - 1/bV]],  dtype=float)


        gain = L_EE * self.I_stim_E + L_EP * self.I_stim_P + L_ES * self.I_stim_S + L_EV * self.I_stim_V
        # print(+ self.wEP*self.wSV*self.wVS, - self.wES*self.wSV*self.wVP, + self.wES*self.wSP * (1/bV), - self.wEP/(bS*bV))
        # print(L_EP)
        # print(det_num)
        # print(gain)

        all_L = (L_EE, L_EP, L_ES, L_EV, L_PE, L_PP, L_PS, L_PV,
                 L_SE, L_SP, L_SS, L_SV, L_VE, L_VP, L_VS, L_VV)

        all_b = (bE, bP, bS, bV) 

        # print(( + self.wEP*self.wSV*self.wVS, - self.wES*self.wSV*self.wVP,  + self.wES*self.wSP/bV, - self.wEP/(bS*bV)))
        # print(+ self.wES, self.wSP, 1/bV, "\t", - self.wES, self.wSV, self.wVP)
        # if self.wES > (self.wEP*self.wPS)/(self.wPP + 1/bP):
        #     print("DisNet \t", self.wES, (self.wEP*self.wPS)/(self.wPP + 1/bP)) #dis VS inh
        # else:
        #     print("InhNet \t", self.wES, (self.wEP*self.wPS)/(self.wPP + 1/bP)) #dis VS inh

        try:
            # evs = eig(np.linalg.inv(J))[0]
            evs = eig(J)[0]
            max_ev_real = float(np.max(np.real(evs)))
            max_ev_imag = float(np.max(np.imag(evs)))
        except Exception:
            max_ev_real = np.nan
            max_ev_imag = np.nan

        return dict(
            gain=float(gain),
            max_ev_real=max_ev_real,
            max_ev_imag=max_ev_imag,
            all_L= all_L,
            all_b= all_b,
            det=float(det_num), J=J, singular=False
        )
        
    def _assign_weights(self, updates: dict):
        """
        In-place assign given weights in both attributes and the W array.
        """
        # update attributes
        for k, v in updates.items():
            setattr(self, k, float(v))
        # update W from attributes in the canonical order
        vals = [getattr(self, name) for name in self._w_order]
        self.W = np.array(vals, dtype=float).reshape(4, 4)
        
    def calculate_linear_theory_general(
        self,
        mode="rates",
        grid_search=False,
        w1_name=None,
        w1_vals=None,
        w2_name=None,
        w2_vals=None
    ):
        """
        Unified dynamics computation across either firing-rate or weight parameter sweeps.

        Parameters
        ----------
        mode : {'rates', 'weights'}
            Determines what is being varied: rates (rE,rP) or network weights.
        grid_search : bool
            If True, perform a 2D sweep over rates; otherwise compute a single point.
        max_rE, max_rP : float
            Used for 'rates' mode when grid_search=True or as fixed values otherwise.
        w1_name : str
            Weight name to sweep in 'weights' mode.
        w1_vals : array-like
            Values for the weight parameter sweep.
        rE_fixed, rP_fixed : float
            Used for 'weights' mode as fixed operating point.

        Returns
        -------
        out : dict
            Dictionary with fields:
            ['gain','gain_mod','maxEV','maxEV_mod','maxImEV','maxImEV_mod',
            'dE','dP','dS','dV'] and parameter vectors (rE_vec/rP_vec or w1_vals).
        """
        # ---------------------------------------------------------------------
        # 1. Define parameter domain
        # ---------------------------------------------------------------------
        if mode == "rates":
            if grid_search:
                max_rEP = max(self.rE0, self.rP0)
                self.rE_vec = np.arange(max_rEP, 0.5 - self.step_rX, -self.step_rX)
                self.rP_vec = np.arange(0.5, max_rEP + self.step_rX, self.step_rX)
                single_point = False
            else:
                self.rE_vec = np.array([self.rE0])
                self.rP_vec = np.array([self.rP0])
                single_point = True

        elif mode == "weights":
            if w1_name not in self._w_index:
                raise ValueError(f"Invalid weight name {w1_name}.")
            w1_vals = np.asarray(w1_vals, dtype=float)

            if w2_name is None:
                single_point = len(w1_vals) == 1
            else:
                if w2_name not in self._w_index:
                    raise ValueError(f"Invalid weight name {w2_name}.")
                w2_vals = np.asarray(w2_vals, dtype=float)
                single_point = False
        else:
            raise ValueError("mode must be either 'rates' or 'weights'.")

        # ---------------------------------------------------------------------
        # 2. Prepare storage
        # ---------------------------------------------------------------------
        if mode == "rates":
            nE, nP = len(self.rE_vec), len(self.rP_vec)
            shape = (nE, nP)
        else:
            if w2_name is None:
                shape = (len(w1_vals),)
            else:
                shape = (len(w1_vals), len(w2_vals))

        gain         = np.full(shape, np.nan)
        gain_mod     = np.full(shape, np.nan)
        maxEV        = np.full(shape, np.nan)
        maxEV_mod    = np.full(shape, np.nan)
        maxImEV      = np.full(shape, np.nan)
        maxImEV_mod  = np.full(shape, np.nan)
        dE           = np.full(shape, np.nan)
        dP           = np.full(shape, np.nan)
        dS           = np.full(shape, np.nan)
        dV           = np.full(shape, np.nan)
        bE           = np.full(shape, np.nan)
        bP           = np.full(shape, np.nan)
        bS           = np.full(shape, np.nan)
        bV           = np.full(shape, np.nan)

        # ---------------------------------------------------------------------
        # 3. Main loop
        # ---------------------------------------------------------------------
        orig_weights = {name: getattr(self, name) for name in self._w_order}
        rS, rV = self.rS0, self.rV0

        if mode == "rates":
            for i, rE in enumerate(self.rE_vec):
                for j, rP in enumerate(self.rP_vec):
                    res_pre = self._compute_block(mode,rE, rP, rS, rV, add_input=False)
                    if res_pre["singular"]:
                        continue

                    gain[i, j] = res_pre["gain"]
                    maxEV[i, j] = res_pre["max_ev_real"]
                    maxImEV[i, j] = res_pre["max_ev_imag"]

                    # --- Compute VIP deltas ---
                    L = res_pre["all_L"]
                    (L_EE, L_EP, L_ES, L_EV,
                    L_PE, L_PP, L_PS, L_PV,
                    L_SE, L_SP, L_SS, L_SV,
                    L_VE, L_VP, L_VS, L_VV) = L

                    dE[i, j] = L_EE*self.I_mod_E + L_EP*self.I_mod_P + L_ES*self.I_mod_S + L_EV*self.I_mod_V
                    dP[i, j] = L_PE*self.I_mod_E + L_PP*self.I_mod_P + L_PS*self.I_mod_S + L_PV*self.I_mod_V
                    dS[i, j] = L_SE*self.I_mod_E + L_SP*self.I_mod_P + L_SS*self.I_mod_S + L_SV*self.I_mod_V
                    dV[i, j] = L_VE*self.I_mod_E + L_VP*self.I_mod_P + L_VS*self.I_mod_S + L_VV*self.I_mod_V

                    res_post = self._compute_block(mode,rE, rP, rS, rV, dE[i, j], dP[i, j], dS[i, j], dV[i, j], add_input=True)
                    if not res_post["singular"]:
                        gain_mod[i, j]    = res_post["gain"]
                        maxEV_mod[i, j]   = res_post["max_ev_real"]
                        maxImEV_mod[i, j] = res_post["max_ev_imag"]

                if not single_point:
                    perc = np.ceil(100 * i / len(self.rE_vec))
                    if not perc % 5:
                        print(f"Completed: {perc}%", end="\r")

        else:  # mode == 'weights'
            rE_fixed = self.rE0
            rP_fixed = self.rP0

            if w2_name is None:
                for i, w1 in enumerate(w1_vals):
                    self._assign_weights({w1_name: w1})
                    res = self._compute_block(mode, rE_fixed, rP_fixed, self.rS0, self.rV0)

                    if res["singular"]:
                        continue

                    gain[i]    = res["gain"]
                    maxEV[i]   = res["max_ev_real"]
                    maxImEV[i] = res["max_ev_imag"]
                    bE[i], bP[i], bS[i], bV[i] = res["all_b"]

            else:
                for i, w1 in enumerate(w1_vals):
                    for j, w2 in enumerate(w2_vals):
                        self._assign_weights({w1_name: w1, w2_name: w2})
                        res = self._compute_block(mode, rE_fixed, rP_fixed, self.rS0, self.rV0)

                        if res["singular"]:
                            continue

                        gain[i, j]    = res["gain"]
                        maxEV[i, j]   = res["max_ev_real"]
                        maxImEV[i, j] = res["max_ev_imag"]
                        bE[i, j], bP[i, j], bS[i, j], bV[i, j] = res["all_b"]

            self._assign_weights(orig_weights)
            all_b = dict(bE=bE, bP=bP, bS=bS, bV=bV)

        # ---------------------------------------------------------------------
        # 4. Assemble result
        # ---------------------------------------------------------------------
        if mode == "rates" and single_point:
            return dict(
                gain=gain[0, 0], gain_mod=gain_mod[0, 0],
                maxEV=maxEV[0, 0], maxEV_mod=maxEV_mod[0, 0],
                maxImEV=maxImEV[0, 0], maxImEV_mod=maxImEV_mod[0, 0],
                dE=dE[0, 0], dP=dP[0, 0], dS=dS[0, 0], dV=dV[0, 0]
            )
        elif mode == "rates":
            return dict(
                rE_vec=self.rE_vec, rP_vec=self.rP_vec,
                gain=gain, gain_mod=gain_mod,
                maxEV=maxEV, maxEV_mod=maxEV_mod,
                maxImEV=maxImEV, maxImEV_mod=maxImEV_mod,
                dE=dE, dP=dP, dS=dS, dV=dV
            )
        else:
            out = dict(
                gain=gain,
                maxEV=maxEV,
                maxImEV=maxImEV,
                all_b=all_b
            )
            out["w1_vals"] = w1_vals
            if w2_name is not None:
                out["w2_vals"] = w2_vals
            return out

    def plot_heatmaps_with_arrows_and_scatter(self, case_name, folder_name, iso_stab, iso_gain, step_arrows=100, nr_points=1000):
        """Plot heatmaps with arrow overlays and scatter plots for a given case"""

        # Calculate dynamics
        res = self.calculate_linear_theory_general(mode="rates", grid_search=True)

        gain_num        = res["gain"]
        gain_num_mod    = res["gain_mod"]
        maxEVs_num      = res["maxEV"]
        maxEVs_num_mod  = res["maxEV_mod"]
        maxImEVs_num    = res["maxImEV"]
        maxImEVs_num_mod= res["maxImEV_mod"]
        modV_rE_num     = res["dE"]
        modV_rP_num     = res["dP"]
        modV_rS_num     = res["dS"]
        modV_rV_num     = res["dV"]
        # Create stability mask
        mask_threshold = -0.05 # Threshold for stability masking
        maxEVs_num_set0 = np.minimum(maxEVs_num, 0)
        mask01 = np.copy(maxEVs_num)
        mask01[mask01 > mask_threshold] = 0
        mask01[mask01 < mask_threshold] = 1

        # Normalize stability
        norm_maxEVs_num_set0 = mask01 * maxEVs_num_set0
        min_val_EV = np.nanmin(norm_maxEVs_num_set0[norm_maxEVs_num_set0 != 0])
        if min_val_EV != 0:
            norm_maxEVs_num_set0 = norm_maxEVs_num_set0 / min_val_EV

        max_val_gain = np.nanmax(gain_num[mask01 != 0])
        min_val_gain = np.nanmin(gain_num[mask01 != 0])

        norm_gain_num_masked = mask01 * ((gain_num - min_val_gain) / (max_val_gain - min_val_gain))

        # Create coordinate matrices
        rE_mat, rP_mat = np.meshgrid(self.rE_vec, self.rP_vec, indexing='ij')

        # Normalize arrow directions
        masked_modV_rP = mask01 * modV_rP_num
        masked_modV_rE = mask01 * modV_rE_num

        # Avoid division by zero
        magnitude = np.sqrt(masked_modV_rP**2 + masked_modV_rE**2)
        magnitude[magnitude == 0] = 1

        norm_masked_modV_rP = masked_modV_rP / magnitude
        norm_masked_modV_rE = masked_modV_rE / magnitude

        # Set arrows to zero where mask is zero
        norm_masked_modV_rP[mask01 == 0] = 0
        norm_masked_modV_rE[mask01 == 0] = 0

        # Calculate differences for scatter plots
        maxEVs_num_set0_2 = np.minimum(maxEVs_num, 0)
        mask01_2 = maxEVs_num.copy()
        mask01_2[mask01_2 > mask_threshold] = np.nan
        mask01_2[mask01_2 < mask_threshold] = 1

        maxEVs_num_set0_2_mod = np.minimum(maxEVs_num_mod, 0)
        mask01_2_mod = maxEVs_num_mod.copy()
        mask01_2_mod[mask01_2_mod > mask_threshold] = np.nan
        mask01_2_mod[mask01_2_mod < mask_threshold] = 1

        diff_maxEVs_num_set0_2_mod = (mask01_2 * maxEVs_num_set0_2 -
                                       mask01_2_mod * maxEVs_num_set0_2_mod)

        gain_num_set0_2 = mask01_2 * gain_num
        gain_num_set0_2_mod = mask01_2_mod * gain_num_mod
        diff_gain_num_set0_2_mod = gain_num_set0_2_mod - gain_num_set0_2

        # Plot random subset of points
        subset_indices = np.random.choice(diff_maxEVs_num_set0_2_mod.size, nr_points, replace=False)
        subset_i, subset_j = np.unravel_index(subset_indices, diff_maxEVs_num_set0_2_mod.shape)

        # print(oscMetric_num[distEVs_num == 0])
    #===============================================================================================
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()
        fig.suptitle(f'{case_name} case')

    ##### Plot 1: Gain with arrows
        im1 = ax1.imshow(norm_gain_num_masked, cmap=self.cmap, aspect='auto',
                         extent=[self.rP_vec[0], self.rP_vec[-1], self.rE_vec[-1], self.rE_vec[0]])

        # Add arrows to gain plot
        ax1.quiver(rP_mat[::step_arrows, ::step_arrows],
                   rE_mat[::step_arrows, ::step_arrows],
                   norm_masked_modV_rP[::step_arrows, ::step_arrows],
                   norm_masked_modV_rE[::step_arrows, ::step_arrows],
                   angles='xy', color='black', alpha=0.7)

        # Plot gain isolines
        cont1 = ax1.contour(rP_mat, rE_mat, norm_gain_num_masked, levels=iso_gain, linewidths=1.5)

        ax1.set_title(f'Gain (normalized)')
        ax1.set_xlabel('rP (Hz)')
        ax1.set_ylabel('rE (Hz)')

    ##### Plot 2: Stability with arrows
        im2 = ax2.imshow(norm_maxEVs_num_set0, cmap=self.cmap, aspect='auto',
                         extent=[self.rP_vec[0], self.rP_vec[-1], self.rE_vec[-1], self.rE_vec[0]])

        # Add arrows to stability plot
        ax2.quiver(rP_mat[::step_arrows, ::step_arrows],
                   rE_mat[::step_arrows, ::step_arrows],
                   norm_masked_modV_rP[::step_arrows, ::step_arrows],
                   norm_masked_modV_rE[::step_arrows, ::step_arrows],
                   angles='xy', color='black', alpha=0.7)

        # Plot stability isolines
        cont2 = ax2.contour(rP_mat, rE_mat, norm_maxEVs_num_set0, levels=iso_stab, linewidths=1.5)

        ax2.set_title(f'Stability (normalized)')
        ax2.set_xlabel('rP (Hz)')
        ax2.set_ylabel('rE (Hz)')

    ##### Plot 3: Scatter plot for deltas of gain and stability
        for i, j in zip(subset_i, subset_j):
            if not (np.isnan(diff_maxEVs_num_set0_2_mod[i, j]) or np.isnan(diff_gain_num_set0_2_mod[i, j])):
                ax3.scatter(diff_maxEVs_num_set0_2_mod[i, j], diff_gain_num_set0_2_mod[i, j],
                          s=1, c='k', alpha=0.5)

        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax3.set_title('mod VIP')
        ax3.set_xlabel(r'$\Delta$ stability')
        ax3.set_ylabel(r'$\Delta$ gain')
        ax3.set_xlim([-0.5, 0.5])
        ax3.set_ylim([-4, 4])


    ##### Plot 4: Info about parameters
        matrix_str = f'Random Points: {nr_points}\n\n' + f'SST init. rate: {self.rS0}\nVIP init. rate: {self.rV0}\n\n' + f'Stimulation E: {self.I_stim_E}\nStimulation PV: {self.I_stim_P}\nModulation VIP: {self.I_mod_V}\n\n' + 'Weight Matrix:\n' + '\n'.join(['  '.join([f'{val:.2f}' for val in row]) for row in self.W])
        ax4.axis('off')  # turn off axes
        ax4.text(0.5, 0.5, matrix_str, fontsize=12, ha='center', va='center', family='monospace')

        return fig


    def simulate_dynamics(self, tauE=10, tauP=10, tauS=10, tauV=10,
                        dt=0.01, end_sim=1000,
                        stim_times=(50, 350, 700)):
        # ----------------------------------------------------------------
        # Initialize rates with fixed-point refinement (Newton iterations)
        # ----------------------------------------------------------------
        rE, rP, rS, rV = self.rE0, self.rP0, self.rS0, self.rV0

        newton_baseline = False
        if newton_baseline:
            def residual(r_vec):
                rE_, rP_, rS_, rV_ = r_vec
                # steady-state STP factors
                pSP = 1 / (1 + self.U_d * self.tau_d * rP_)
                pSV = (1 + self.U_f * self.tau_f * self.U_max * rV_) / (1 + self.U_f * self.tau_f * rV_)
                # pSV = 1

                # steady-state condition: r = mult_f * f(g)
                xE = (rE_ / self.mult_f)**(1/self.power) - (self.wEE*rE_ - self.wEP*rP_ - self.wES*rS_ - self.wEV*rV_)
                xP = (rP_ / self.mult_f)**(1/self.power) - (self.wPE*rE_ - self.wPP*rP_ - self.wPS*rS_ - self.wPV*rV_)
                xS = (rS_ / self.mult_f)**(1/self.power) - (self.wSE*rE_ - pSP*self.wSP*rP_ - self.wSS*rS_ - pSV*self.wSV*rV_)
                xV = (rV_ / self.mult_f)**(1/self.power) - (self.wVE*rE_ - self.wVP*rP_ - self.wVS*rS_ - self.wVV*rV_)
                gE = self.wEE*rE_ - self.wEP*rP_ - self.wES*rS_ - self.wEV*rV_ + xE
                gP = self.wPE*rE_ - self.wPP*rP_ - self.wPS*rS_ - self.wPV*rV_ + xP
                gS = self.wSE*rE_ - pSP*self.wSP*rP_ - self.wSS*rS_ - pSV*self.wSV*rV_ + xS
                gV = self.wVE*rE_ - self.wVP*rP_ - self.wVS*rS_ - self.wVV*rV_ + xV
                phi = self.mult_f * np.maximum([gE,gP,gS,gV],0.0)**self.power
                return r_vec - phi

            # Newton iterations for initial steady-state
            r_vec = np.array([rE, rP, rS, rV])
            for _ in range(10):
                F = residual(r_vec)
                if np.linalg.norm(F) < 1e-9:
                    break
                eps = 1e-5
                J_fd = np.zeros((4,4))
                for j in range(4):
                    r_pert = r_vec.copy()
                    r_pert[j] += eps
                    J_fd[:,j] = (residual(r_pert) - F)/eps
                try:
                    step = np.linalg.solve(J_fd, F)
                except np.linalg.LinAlgError:
                    break
                r_vec -= step
                r_vec = np.clip(r_vec, 0, None)
            rE, rP, rS, rV = r_vec  # refined baseline

        # ----------------------------------------------------------------
        # Compute baseline x*
        # ----------------------------------------------------------------
        xE0 = (rE / self.mult_f)**(1 / self.power) - (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV)
        xP0 = (rP / self.mult_f)**(1 / self.power) - (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV)
        xS0 = (rS / self.mult_f)**(1 / self.power) - (self.wSE*rE - self.wSP*rP - self.wSS*rS - self.wSV*rV)
        xV0 = (rV / self.mult_f)**(1 / self.power) - (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV)

        # steady-state STP initialization
        # d_SP = 1 / (1 + self.U_d * self.tau_d * rP)
        d_SP = 1
        # f_SV = (1 + self.U_f * self.tau_f * self.U_max * rV) / (1 + self.U_f * self.tau_f * rV)
        f_SV = 1

        # ----------------------------------------------------------------
        # Initialize traces
        # ----------------------------------------------------------------
        time = np.arange(0, end_sim + dt, dt)
        rE_trace = np.zeros_like(time)
        rP_trace = np.zeros_like(time)
        rS_trace = np.zeros_like(time)
        rV_trace = np.zeros_like(time)
        d_SP_trace = np.ones_like(time)
        f_SV_trace = np.ones_like(time)

        # Initial conditions
        rE_trace[0], rP_trace[0], rS_trace[0], rV_trace[0], d_SP_trace[0], f_SV_trace[0] = rE, rP, rS, rV, d_SP, f_SV
        xE_fb, xP_fb, xS_fb, xV_fb = 0.0, 0.0, 0.0, 0.0
        xE_ff, xP_ff, xS_ff, xV_ff = 0.0, 0.0, 0.0, 0.0

        stp_active = False  # flag for STP on/off

        # Decide which events are available
        if len(stim_times) == 3:
            t_stp, t_mod, t_stim = stim_times
            use_modulation = True
        elif len(stim_times) == 2:
            t_stp, t_stim = stim_times
            use_modulation = False
        else:
            raise ValueError("stim_times must have length 2 or 3")

        for i in range(1, len(time)):
            t = time[i]

            # STP activation onset
            if np.isclose(t, t_stp, atol=dt/2):
                stp_active = True

            # modulation onset (only if present)
            if use_modulation and np.isclose(t, t_mod, atol=dt/2):
                xE_fb, xP_fb, xS_fb, xV_fb = self.I_mod_E, self.I_mod_P, self.I_mod_S, self.I_mod_V

            # stimulation onset
            if np.isclose(t, t_stim, atol=dt/2):
                xE_ff, xP_ff, xS_ff, xV_ff = self.I_stim_E, self.I_stim_P, self.I_stim_S, self.I_stim_V


            # Euler integration
            rE_dot = (-rE + self.mult_f * (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV + xE0 + xE_ff + xE_fb)**self.power) / tauE
            rP_dot = (-rP + self.mult_f * (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV + xP0 + xP_ff + xP_fb)**self.power) / tauP
            rS_dot = (-rS + self.mult_f * (self.wSE*rE - d_SP*self.wSP*rP - self.wSS*rS - f_SV*self.wSV*rV + xS0 + xS_ff + xS_fb)**self.power) / tauS
            rV_dot = (-rV + self.mult_f * (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV + xV0 + xV_ff + xV_fb)**self.power) / tauV

            # STP variable updates only after activation
            if stp_active:
                d_SP_dot = ((1 - d_SP) / self.tau_d) - self.U_d * d_SP * rP
                f_SV_dot = (1 - f_SV) / self.tau_f + self.U_f * (self.U_max - f_SV) * rV
                # f_SV_dot = 0
            else:
                d_SP_dot, f_SV_dot = 0, 0  # frozen

            # integrate all variables
            rE += dt * rE_dot
            rP += dt * rP_dot
            rS += dt * rS_dot
            rV += dt * rV_dot
            d_SP += dt * d_SP_dot
            f_SV += dt * f_SV_dot

            rE = max(rE, 0.0) #rectify
            rP = max(rP, 0.0)
            rS = max(rS, 0.0)
            rV = max(rV, 0.0)

            # store traces
            rE_trace[i], rP_trace[i], rS_trace[i], rV_trace[i] = rE, rP, rS, rV
            d_SP_trace[i], f_SV_trace[i] = d_SP, f_SV

        return dict(time=time, rE=rE_trace, rP=rP_trace, rS=rS_trace,
                    rV=rV_trace, d_SP=d_SP_trace, f_SV=f_SV_trace)

class VIP_circuit_neuromod:
    """
    E-PV-SST-VIP circuit with neuromodulated inhibition of SST (Gupta et al.).

    PV->SST: Tsodyks-Markram depression with release probability U_SP,
        pSP(rP) = U_SP / (1 + U_SP*rP*tau_SP).
        ACh lowers U_SP -> first-pulse amplitude down, PPR up (both observed).
    VIP->SST: static synapse with gain 1 at every arousal level. NE doubles
        the first-pulse amplitude but halves the PPR (Fig 3M-P), so by the
        second pulse (~50 ms) the effective strength is back at baseline:
        no steady-state effect, hence NE is not modeled.
    Arousal axis theta in [0, 1] via set_arousal(): the circuit changes only
        through U_SP(theta) decreasing (ACh side). theta also sets the input
        mix: stimulus = (1-theta)*FF + theta*FB (see gain_mix in
        sweep_arousal).
    FF drive targets E + PV (I_FF_E, I_FF_P); FB drive targets VIP
        (I_FB_V, optionally I_FB_E). gain_FF / gain_FB are the corresponding
        linear-response gains of E.

    The linear-theory expressions (det, L, Jacobian) are copied verbatim from
    VIP_circuit_double_plasticity; they are generic in pSP/DpSP/pSV/DpSV, so
    only the synaptic-factor definitions differ.
    """
    def __init__(self, W, rE0=3.0, rP0=5.0, rS0=2.0, rV0=2.0, I_stim_E=0.3, I_stim_P=0.3, I_stim_S=0.0, I_stim_V=0.0, I_mod_E=0.0,I_mod_P=0.0,I_mod_S=0.0,I_mod_V=0.0, step_rX=0.01, power=1, mult_f=1, theta=0.0, I_FF_E=1.0, I_FF_P=1.0, I_FB_V=1.0, I_FB_E=0.0):
        # Define parameters
        self.W = W

        self.wEE, self.wEP, self.wES, self.wEV, self.wPE, self.wPP, self.wPS, self.wPV, self.wSE, self.wSP, self.wSS, self.wSV, self.wVE, self.wVP, self.wVS, self.wVV = W.flatten()

        self.rE0 = rE0 #for weight sweep set-point
        self.rP0 = rP0
        self.rS0 = rS0  # SST initial firing rate
        self.rV0 = rV0  # VIP initial firing rate
        self.step_rX = step_rX

        self.I_stim_E = I_stim_E
        self.I_stim_P = I_stim_P
        self.I_stim_S = I_stim_S
        self.I_stim_V = I_stim_V
        self.I_mod_E = I_mod_E
        self.I_mod_P = I_mod_P
        self.I_mod_S = I_mod_S
        self.I_mod_V = I_mod_V

        # FF drive -> E + PV; FB drive -> VIP (optionally E)
        self.I_FF_E = I_FF_E
        self.I_FF_P = I_FF_P
        self.I_FB_V = I_FB_V
        self.I_FB_E = I_FB_E

        # I/O function parameters
        self.power = power
        self.mult_f = mult_f

        # --- neuromodulated synapses onto SST ---
        # PV->SST: TM depression; ACh lowers the release probability U_SP
        self.tau_SP = 0.100      # s, depression recovery
        self.U_SP_base = 0.5     # low arousal
        self.U_SP_high = 0.15    # high arousal (ACh)
        # VIP->SST: static, gain 1 at all theta (NE transient cancels by 50 ms,
        # Fig 3M-P: amplitude x2 * PPR x0.5 -> no steady-state effect)
        # slow alpha5-like postsynaptic filter on VIP->SST (simulation only)
        self.tau_syn_slow = 50.0  # ms
        self.set_arousal(theta)

        # --- weight name -> index mapping ---
        self._w_order = [
        "wEE","wEP","wES","wEV",
        "wPE","wPP","wPS","wPV",
        "wSE","wSP","wSS","wSV",
        "wVE","wVP","wVS","wVV"
        ]
        self._w_index = {name:i for i, name in enumerate(self._w_order)}

    def set_arousal(self, theta):
        """Configure the arousal state theta in [0, 1]:
        I(theta) = (1-theta)*FF + theta*FB, with FF = E+PV input carried
        through the PV->SST synapse in state U_SP(theta), FB = VIP input.
        Sets the synaptic state U_SP (the only circuit change) and the
        arousal-routed input amplitudes I_arousal_* (channel amplitudes
        I_FF_*/I_FB_* scaled by the shares 1-theta / theta)."""
        self.theta = float(theta)
        self.U_SP = (1 - self.theta) * self.U_SP_base + self.theta * self.U_SP_high
        self.I_arousal_E = (1 - self.theta) * self.I_FF_E + self.theta * self.I_FB_E
        self.I_arousal_P = (1 - self.theta) * self.I_FF_P
        self.I_arousal_V = self.theta * self.I_FB_V

    def _compute_block(self,
                    mode: str,
                    rE: float,
                    rP: float,
                    rS: float,
                    rV: float,
                    dE: float = 0.0,
                    dP: float = 0.0,
                    dS: float = 0.0,
                    dV: float = 0.0,
                    add_input: bool = False,):
        """
        Linear theory at one operating point: linear-response matrix L, gains
        (gain, gain_FF, gain_FB), Jacobian and its leading eigenvalue.
        Same math as VIP_circuit_double_plasticity._compute_block except for
        the synaptic-factor definitions pSP/DpSP/pSV/DpSV (see class docstring).
        """
        rE_eff = rE + dE
        rP_eff = rP + dP
        rS_eff = rS + dS
        rV_eff = rV + dV

        if add_input:
            add_E, add_P, add_S, add_V = self.I_mod_E, self.I_mod_P, self.I_mod_S, self.I_mod_V
        else:
            add_E = add_P = add_S = add_V = 0.0

        # PV->SST: TM depression, U_SP in the numerator so that a U change moves
        # amplitude and PPR together (ACh: both down/up as observed)
        pSP = self.U_SP / (1 + self.U_SP * rP_eff * self.tau_SP)
        DpSP = - pSP**2 * self.tau_SP
        # VIP->SST: static, gain 1 (NE not modeled; no steady-state effect)
        pSV = 1.0
        DpSV = 0.0

        xE = (rE / self.mult_f)**(1 / self.power) - (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV)
        xP = (rP / self.mult_f)**(1 / self.power) - (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV)
        xS = (rS / self.mult_f)**(1 / self.power) - (self.wSE*rE - pSP*self.wSP*rP - self.wSS*rS - pSV*self.wSV*rV)
        xV = (rV / self.mult_f)**(1 / self.power) - (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV)


        if mode == "rates":
            bE = self.power * self.mult_f * (self.wEE*rE_eff - self.wEP*rP_eff - self.wES*rS_eff - self.wEV*rV_eff + xE + add_E)**(self.power - 1)
            bP = self.power * self.mult_f * (self.wPE*rE_eff - self.wPP*rP_eff - self.wPS*rS_eff - self.wPV*rV_eff + xP + add_P)**(self.power - 1)
            bS = self.power * self.mult_f * (self.wSE*rE_eff - pSP*self.wSP*rP_eff - self.wSS*rS_eff - pSV*self.wSV*rV_eff + xS + add_S)**(self.power - 1)
            bV = self.power * self.mult_f * (self.wVE*rE_eff - self.wVP*rP_eff - self.wVS*rS_eff - self.wVV*rV_eff + xV + add_V)**(self.power - 1)
            # print(self.wSE*rE_eff, - self.wSP*rP_eff, - self.wSV*rV_eff, + xS)
            # print(rP_eff, rS_eff)
        elif mode == "weights":
            bE = self.power * self.mult_f * ((rE + self.I_mod_E)/self.mult_f)**((self.power - 1)/self.power)
            bP = self.power * self.mult_f * ((rP + self.I_mod_P)/self.mult_f)**((self.power - 1)/self.power)
            bS = self.power * self.mult_f * ((rS + self.I_mod_S)/self.mult_f)**((self.power - 1)/self.power)
            bV = self.power * self.mult_f * ((rV + self.I_mod_V)/self.mult_f)**((self.power - 1)/self.power)
    

        det_num = (DpSP*bE*bP*bS*bV*rP*self.wEE*self.wPS*self.wSP*self.wVV - DpSP*bE*bP*bS*bV*rP*self.wEE*self.wPV*self.wSP*self.wVS - DpSP*bE*bP*bS*bV*rP*self.wES*self.wPE*self.wSP*self.wVV + DpSP*bE*bP*bS*bV*rP*self.wES*self.wPV*self.wSP*self.wVE + DpSP*bE*bP*bS*bV*rP*self.wEV*self.wPE*self.wSP*self.wVS - DpSP*bE*bP*bS*bV*rP*self.wEV*self.wPS*self.wSP*self.wVE + DpSP*bE*bP*bS*rP*self.wEE*self.wPS*self.wSP - DpSP*bE*bP*bS*rP*self.wES*self.wPE*self.wSP - DpSP*bP*bS*bV*rP*self.wPS*self.wSP*self.wVV + DpSP*bP*bS*bV*rP*self.wPV*self.wSP*self.wVS - DpSP*bP*bS*rP*self.wPS*self.wSP + DpSV*bE*bP*bS*bV*rV*self.wEE*self.wPP*self.wSV*self.wVS - DpSV*bE*bP*bS*bV*rV*self.wEE*self.wPS*self.wSV*self.wVP - DpSV*bE*bP*bS*bV*rV*self.wEP*self.wPE*self.wSV*self.wVS + DpSV*bE*bP*bS*bV*rV*self.wEP*self.wPS*self.wSV*self.wVE + DpSV*bE*bP*bS*bV*rV*self.wES*self.wPE*self.wSV*self.wVP - DpSV*bE*bP*bS*bV*rV*self.wES*self.wPP*self.wSV*self.wVE + DpSV*bE*bS*bV*rV*self.wEE*self.wSV*self.wVS - DpSV*bE*bS*bV*rV*self.wES*self.wSV*self.wVE - DpSV*bP*bS*bV*rV*self.wPP*self.wSV*self.wVS + DpSV*bP*bS*bV*rV*self.wPS*self.wSV*self.wVP - DpSV*bS*bV*rV*self.wSV*self.wVS + bE*bP*bS*bV*pSP*self.wEE*self.wPS*self.wSP*self.wVV - bE*bP*bS*bV*pSP*self.wEE*self.wPV*self.wSP*self.wVS - bE*bP*bS*bV*pSP*self.wES*self.wPE*self.wSP*self.wVV + bE*bP*bS*bV*pSP*self.wES*self.wPV*self.wSP*self.wVE + bE*bP*bS*bV*pSP*self.wEV*self.wPE*self.wSP*self.wVS - bE*bP*bS*bV*pSP*self.wEV*self.wPS*self.wSP*self.wVE + bE*bP*bS*bV*pSV*self.wEE*self.wPP*self.wSV*self.wVS - bE*bP*bS*bV*pSV*self.wEE*self.wPS*self.wSV*self.wVP - bE*bP*bS*bV*pSV*self.wEP*self.wPE*self.wSV*self.wVS + bE*bP*bS*bV*pSV*self.wEP*self.wPS*self.wSV*self.wVE + bE*bP*bS*bV*pSV*self.wES*self.wPE*self.wSV*self.wVP - bE*bP*bS*bV*pSV*self.wES*self.wPP*self.wSV*self.wVE - bE*bP*bS*bV*self.wEE*self.wPP*self.wSS*self.wVV + bE*bP*bS*bV*self.wEE*self.wPV*self.wSS*self.wVP + bE*bP*bS*bV*self.wEP*self.wPE*self.wSS*self.wVV - bE*bP*bS*bV*self.wEP*self.wPS*self.wSE*self.wVV + bE*bP*bS*bV*self.wEP*self.wPV*self.wSE*self.wVS - bE*bP*bS*bV*self.wEP*self.wPV*self.wSS*self.wVE + bE*bP*bS*bV*self.wES*self.wPP*self.wSE*self.wVV - bE*bP*bS*bV*self.wES*self.wPV*self.wSE*self.wVP - bE*bP*bS*bV*self.wEV*self.wPE*self.wSS*self.wVP - bE*bP*bS*bV*self.wEV*self.wPP*self.wSE*self.wVS + bE*bP*bS*bV*self.wEV*self.wPP*self.wSS*self.wVE + bE*bP*bS*bV*self.wEV*self.wPS*self.wSE*self.wVP + bE*bP*bS*pSP*self.wEE*self.wPS*self.wSP - bE*bP*bS*pSP*self.wES*self.wPE*self.wSP - bE*bP*bS*self.wEE*self.wPP*self.wSS + bE*bP*bS*self.wEP*self.wPE*self.wSS - bE*bP*bS*self.wEP*self.wPS*self.wSE + bE*bP*bS*self.wES*self.wPP*self.wSE - bE*bP*bV*self.wEE*self.wPP*self.wVV + bE*bP*bV*self.wEE*self.wPV*self.wVP + bE*bP*bV*self.wEP*self.wPE*self.wVV - bE*bP*bV*self.wEP*self.wPV*self.wVE - bE*bP*bV*self.wEV*self.wPE*self.wVP + bE*bP*bV*self.wEV*self.wPP*self.wVE - bE*bP*self.wEE*self.wPP + bE*bP*self.wEP*self.wPE + bE*bS*bV*pSV*self.wEE*self.wSV*self.wVS - bE*bS*bV*pSV*self.wES*self.wSV*self.wVE - bE*bS*bV*self.wEE*self.wSS*self.wVV + bE*bS*bV*self.wES*self.wSE*self.wVV - bE*bS*bV*self.wEV*self.wSE*self.wVS + bE*bS*bV*self.wEV*self.wSS*self.wVE - bE*bS*self.wEE*self.wSS + bE*bS*self.wES*self.wSE - bE*bV*self.wEE*self.wVV + bE*bV*self.wEV*self.wVE - bE*self.wEE - bP*bS*bV*pSP*self.wPS*self.wSP*self.wVV + bP*bS*bV*pSP*self.wPV*self.wSP*self.wVS - bP*bS*bV*pSV*self.wPP*self.wSV*self.wVS + bP*bS*bV*pSV*self.wPS*self.wSV*self.wVP + bP*bS*bV*self.wPP*self.wSS*self.wVV - bP*bS*bV*self.wPV*self.wSS*self.wVP - bP*bS*pSP*self.wPS*self.wSP + bP*bS*self.wPP*self.wSS + bP*bV*self.wPP*self.wVV - bP*bV*self.wPV*self.wVP + bP*self.wPP - bS*bV*pSV*self.wSV*self.wVS + bS*bV*self.wSS*self.wVV + bS*self.wSS + bV*self.wVV + 1)/(bE*bP*bS*bV)

        L_EE = (-DpSP*rP*self.wPS*self.wSP*self.wVV + DpSP*rP*self.wPV*self.wSP*self.wVS - DpSP*rP*self.wPS*self.wSP/bV - DpSV*rV*self.wPP*self.wSV*self.wVS + DpSV*rV*self.wPS*self.wSV*self.wVP - DpSV*rV*self.wSV*self.wVS/bP - pSP*self.wPS*self.wSP*self.wVV + pSP*self.wPV*self.wSP*self.wVS - pSV*self.wPP*self.wSV*self.wVS + pSV*self.wPS*self.wSV*self.wVP + self.wPP*self.wSS*self.wVV - self.wPV*self.wSS*self.wVP - pSP*self.wPS*self.wSP/bV + self.wPP*self.wSS/bV + self.wPP*self.wVV/bS - self.wPV*self.wVP/bS + self.wPP/(bS*bV) - pSV*self.wSV*self.wVS/bP + self.wSS*self.wVV/bP + self.wSS/(bP*bV) + self.wVV/(bP*bS) + 1/(bP*bS*bV)) / det_num
        L_EP = (DpSP*rP*self.wES*self.wSP*self.wVV - DpSP*rP*self.wEV*self.wSP*self.wVS + DpSP*rP*self.wES*self.wSP/bV + DpSV*rV*self.wEP*self.wSV*self.wVS - DpSV*rV*self.wES*self.wSV*self.wVP + pSP*self.wES*self.wSP*self.wVV - pSP*self.wEV*self.wSP*self.wVS + pSV*self.wEP*self.wSV*self.wVS - pSV*self.wES*self.wSV*self.wVP - self.wEP*self.wSS*self.wVV + self.wEV*self.wSS*self.wVP + pSP*self.wES*self.wSP/bV - self.wEP*self.wSS/bV - self.wEP*self.wVV/bS + self.wEV*self.wVP/bS - self.wEP/(bS*bV)) / det_num
        L_ES = (self.wEP*self.wPS*self.wVV - self.wEP*self.wPV*self.wVS - self.wES*self.wPP*self.wVV + self.wES*self.wPV*self.wVP + self.wEV*self.wPP*self.wVS - self.wEV*self.wPS*self.wVP + self.wEP*self.wPS/bV - self.wES*self.wPP/bV - self.wES*self.wVV/bP + self.wEV*self.wVS/bP - self.wES/(bP*bV)) / det_num
        L_EV = (-DpSP*rP*self.wES*self.wPV*self.wSP + DpSP*rP*self.wEV*self.wPS*self.wSP - DpSV*rV*self.wEP*self.wPS*self.wSV + DpSV*rV*self.wES*self.wPP*self.wSV + DpSV*rV*self.wES*self.wSV/bP - pSP*self.wES*self.wPV*self.wSP + pSP*self.wEV*self.wPS*self.wSP - pSV*self.wEP*self.wPS*self.wSV + pSV*self.wES*self.wPP*self.wSV + self.wEP*self.wPV*self.wSS - self.wEV*self.wPP*self.wSS + self.wEP*self.wPV/bS - self.wEV*self.wPP/bS + pSV*self.wES*self.wSV/bP - self.wEV*self.wSS/bP - self.wEV/(bP*bS)) / det_num
        L_PE = (-DpSV*rV*self.wPE*self.wSV*self.wVS + DpSV*rV*self.wPS*self.wSV*self.wVE - pSV*self.wPE*self.wSV*self.wVS + pSV*self.wPS*self.wSV*self.wVE + self.wPE*self.wSS*self.wVV - self.wPS*self.wSE*self.wVV + self.wPV*self.wSE*self.wVS - self.wPV*self.wSS*self.wVE + self.wPE*self.wSS/bV - self.wPS*self.wSE/bV + self.wPE*self.wVV/bS - self.wPV*self.wVE/bS + self.wPE/(bS*bV)) / det_num
        L_PP = (DpSV*rV*self.wEE*self.wSV*self.wVS - DpSV*rV*self.wES*self.wSV*self.wVE - DpSV*rV*self.wSV*self.wVS/bE + pSV*self.wEE*self.wSV*self.wVS - pSV*self.wES*self.wSV*self.wVE - self.wEE*self.wSS*self.wVV + self.wES*self.wSE*self.wVV - self.wEV*self.wSE*self.wVS + self.wEV*self.wSS*self.wVE - self.wEE*self.wSS/bV + self.wES*self.wSE/bV - self.wEE*self.wVV/bS + self.wEV*self.wVE/bS - self.wEE/(bS*bV) - pSV*self.wSV*self.wVS/bE + self.wSS*self.wVV/bE + self.wSS/(bE*bV) + self.wVV/(bE*bS) + 1/(bE*bS*bV)) / det_num
        L_PS = (self.wEE*self.wPS*self.wVV - self.wEE*self.wPV*self.wVS - self.wES*self.wPE*self.wVV + self.wES*self.wPV*self.wVE + self.wEV*self.wPE*self.wVS - self.wEV*self.wPS*self.wVE + self.wEE*self.wPS/bV - self.wES*self.wPE/bV - self.wPS*self.wVV/bE + self.wPV*self.wVS/bE - self.wPS/(bE*bV)) / det_num
        L_PV = (-DpSV*rV*self.wEE*self.wPS*self.wSV + DpSV*rV*self.wES*self.wPE*self.wSV + DpSV*rV*self.wPS*self.wSV/bE - pSV*self.wEE*self.wPS*self.wSV + pSV*self.wES*self.wPE*self.wSV + self.wEE*self.wPV*self.wSS - self.wES*self.wPV*self.wSE - self.wEV*self.wPE*self.wSS + self.wEV*self.wPS*self.wSE + self.wEE*self.wPV/bS - self.wEV*self.wPE/bS + pSV*self.wPS*self.wSV/bE - self.wPV*self.wSS/bE - self.wPV/(bE*bS)) / det_num
        L_SE = (-DpSP*rP*self.wPE*self.wSP*self.wVV + DpSP*rP*self.wPV*self.wSP*self.wVE - DpSP*rP*self.wPE*self.wSP/bV + DpSV*rV*self.wPE*self.wSV*self.wVP - DpSV*rV*self.wPP*self.wSV*self.wVE - DpSV*rV*self.wSV*self.wVE/bP - pSP*self.wPE*self.wSP*self.wVV + pSP*self.wPV*self.wSP*self.wVE + pSV*self.wPE*self.wSV*self.wVP - pSV*self.wPP*self.wSV*self.wVE + self.wPP*self.wSE*self.wVV - self.wPV*self.wSE*self.wVP - pSP*self.wPE*self.wSP/bV + self.wPP*self.wSE/bV - pSV*self.wSV*self.wVE/bP + self.wSE*self.wVV/bP + self.wSE/(bP*bV)) / det_num
        L_SP = (DpSP*rP*self.wEE*self.wSP*self.wVV - DpSP*rP*self.wEV*self.wSP*self.wVE + DpSP*rP*self.wEE*self.wSP/bV - DpSP*rP*self.wSP*self.wVV/bE - DpSP*rP*self.wSP/(bE*bV) - DpSV*rV*self.wEE*self.wSV*self.wVP + DpSV*rV*self.wEP*self.wSV*self.wVE + DpSV*rV*self.wSV*self.wVP/bE + pSP*self.wEE*self.wSP*self.wVV - pSP*self.wEV*self.wSP*self.wVE - pSV*self.wEE*self.wSV*self.wVP + pSV*self.wEP*self.wSV*self.wVE - self.wEP*self.wSE*self.wVV + self.wEV*self.wSE*self.wVP + pSP*self.wEE*self.wSP/bV - self.wEP*self.wSE/bV - pSP*self.wSP*self.wVV/bE + pSV*self.wSV*self.wVP/bE - pSP*self.wSP/(bE*bV)) / det_num
        L_SS = (-self.wEE*self.wPP*self.wVV + self.wEE*self.wPV*self.wVP + self.wEP*self.wPE*self.wVV - self.wEP*self.wPV*self.wVE - self.wEV*self.wPE*self.wVP + self.wEV*self.wPP*self.wVE - self.wEE*self.wPP/bV + self.wEP*self.wPE/bV - self.wEE*self.wVV/bP + self.wEV*self.wVE/bP - self.wEE/(bP*bV) + self.wPP*self.wVV/bE - self.wPV*self.wVP/bE + self.wPP/(bE*bV) + self.wVV/(bE*bP) + 1/(bE*bP*bV)) / det_num
        L_SV = (-DpSP*rP*self.wEE*self.wPV*self.wSP + DpSP*rP*self.wEV*self.wPE*self.wSP + DpSP*rP*self.wPV*self.wSP/bE + DpSV*rV*self.wEE*self.wPP*self.wSV - DpSV*rV*self.wEP*self.wPE*self.wSV + DpSV*rV*self.wEE*self.wSV/bP - DpSV*rV*self.wPP*self.wSV/bE - DpSV*rV*self.wSV/(bE*bP) - pSP*self.wEE*self.wPV*self.wSP + pSP*self.wEV*self.wPE*self.wSP + pSV*self.wEE*self.wPP*self.wSV - pSV*self.wEP*self.wPE*self.wSV + self.wEP*self.wPV*self.wSE - self.wEV*self.wPP*self.wSE + pSV*self.wEE*self.wSV/bP - self.wEV*self.wSE/bP + pSP*self.wPV*self.wSP/bE - pSV*self.wPP*self.wSV/bE - pSV*self.wSV/(bE*bP)) / det_num
        L_VE = (DpSP*rP*self.wPE*self.wSP*self.wVS - DpSP*rP*self.wPS*self.wSP*self.wVE + pSP*self.wPE*self.wSP*self.wVS - pSP*self.wPS*self.wSP*self.wVE - self.wPE*self.wSS*self.wVP - self.wPP*self.wSE*self.wVS + self.wPP*self.wSS*self.wVE + self.wPS*self.wSE*self.wVP - self.wPE*self.wVP/bS + self.wPP*self.wVE/bS - self.wSE*self.wVS/bP + self.wSS*self.wVE/bP + self.wVE/(bP*bS)) / det_num
        L_VP = (-DpSP*rP*self.wEE*self.wSP*self.wVS + DpSP*rP*self.wES*self.wSP*self.wVE + DpSP*rP*self.wSP*self.wVS/bE - pSP*self.wEE*self.wSP*self.wVS + pSP*self.wES*self.wSP*self.wVE + self.wEE*self.wSS*self.wVP + self.wEP*self.wSE*self.wVS - self.wEP*self.wSS*self.wVE - self.wES*self.wSE*self.wVP + self.wEE*self.wVP/bS - self.wEP*self.wVE/bS + pSP*self.wSP*self.wVS/bE - self.wSS*self.wVP/bE - self.wVP/(bE*bS)) / det_num
        L_VS = (self.wEE*self.wPP*self.wVS - self.wEE*self.wPS*self.wVP - self.wEP*self.wPE*self.wVS + self.wEP*self.wPS*self.wVE + self.wES*self.wPE*self.wVP - self.wES*self.wPP*self.wVE + self.wEE*self.wVS/bP - self.wES*self.wVE/bP - self.wPP*self.wVS/bE + self.wPS*self.wVP/bE - self.wVS/(bE*bP)) / det_num
        L_VV = (DpSP*rP*self.wEE*self.wPS*self.wSP - DpSP*rP*self.wES*self.wPE*self.wSP - DpSP*rP*self.wPS*self.wSP/bE + pSP*self.wEE*self.wPS*self.wSP - pSP*self.wES*self.wPE*self.wSP - self.wEE*self.wPP*self.wSS + self.wEP*self.wPE*self.wSS - self.wEP*self.wPS*self.wSE + self.wES*self.wPP*self.wSE - self.wEE*self.wPP/bS + self.wEP*self.wPE/bS - self.wEE*self.wSS/bP + self.wES*self.wSE/bP - self.wEE/(bP*bS) - pSP*self.wPS*self.wSP/bE + self.wPP*self.wSS/bE + self.wPP/(bE*bS) + self.wSS/(bE*bP) + 1/(bE*bP*bS)) / det_num

        if abs(det_num) < 1e-10:
            return dict(
                gain=np.nan, gain_FF=np.nan, gain_FB=np.nan,
                max_ev_real=np.nan, max_ev_imag=np.nan,
                all_L=np.nan,
                det=det_num, J=None, singular=True
            )

        # S-row STP couplings via the chain rule, d/dr[p(r)*w*r] = (p + Dp*r)*w,
        # consistent with the det/L expressions above. (The parent class's J
        # carries the Dp term with the opposite sign, which is inconsistent
        # with its own L matrix.)
        J = np.array([[self.wEE - 1/bE, -self.wEP, -self.wES, -self.wEV],
                          [self.wPE, -self.wPP - 1/bP, -self.wPS, -self.wPV],
                          [self.wSE, -(pSP + DpSP*rP)*self.wSP, -self.wSS - 1/bS, -(pSV + DpSV*rV)*self.wSV],
                          [self.wVE, -self.wVP, -self.wVS, -self.wVV - 1/bV]],  dtype=float)


        gain = L_EE * self.I_stim_E + L_EP * self.I_stim_P + L_ES * self.I_stim_S + L_EV * self.I_stim_V
        gain_FF = L_EE * self.I_FF_E + L_EP * self.I_FF_P
        gain_FB = L_EV * self.I_FB_V + L_EE * self.I_FB_E
        # print(+ self.wEP*self.wSV*self.wVS, - self.wES*self.wSV*self.wVP, + self.wES*self.wSP * (1/bV), - self.wEP/(bS*bV))
        # print(L_EP)
        # print(det_num)
        # print(gain)

        all_L = (L_EE, L_EP, L_ES, L_EV, L_PE, L_PP, L_PS, L_PV,
                 L_SE, L_SP, L_SS, L_SV, L_VE, L_VP, L_VS, L_VV)

        all_b = (bE, bP, bS, bV) 

        # print(( + self.wEP*self.wSV*self.wVS, - self.wES*self.wSV*self.wVP,  + self.wES*self.wSP/bV, - self.wEP/(bS*bV)))
        # print(+ self.wES, self.wSP, 1/bV, "\t", - self.wES, self.wSV, self.wVP)
        # if self.wES > (self.wEP*self.wPS)/(self.wPP + 1/bP):
        #     print("DisNet \t", self.wES, (self.wEP*self.wPS)/(self.wPP + 1/bP)) #dis VS inh
        # else:
        #     print("InhNet \t", self.wES, (self.wEP*self.wPS)/(self.wPP + 1/bP)) #dis VS inh

        evs = eig(J)[0]
        max_ev_real = float(np.max(np.real(evs)))
        max_ev_imag = float(np.max(np.imag(evs)))

        return dict(
            gain=float(gain),
            gain_FF=float(gain_FF),
            gain_FB=float(gain_FB),
            max_ev_real=max_ev_real,
            max_ev_imag=max_ev_imag,
            all_L= all_L,
            all_b= all_b,
            det=float(det_num), J=J, singular=False
        )

    def _assign_weights(self, updates: dict):
        """
        In-place assign given weights in both attributes and the W array.
        """
        # update attributes
        for k, v in updates.items():
            setattr(self, k, float(v))
        # update W from attributes in the canonical order
        vals = [getattr(self, name) for name in self._w_order]
        self.W = np.array(vals, dtype=float).reshape(4, 4)

    def calculate_linear_theory_general(
        self,
        mode="rates",
        grid_search=False,
        w1_name=None,
        w1_vals=None,
        w2_name=None,
        w2_vals=None
    ):
        """
        Unified dynamics computation across either firing-rate or weight parameter sweeps.

        Parameters
        ----------
        mode : {'rates', 'weights'}
            Determines what is being varied: rates (rE,rP) or network weights.
        grid_search : bool
            If True, perform a 2D sweep over rates; otherwise compute a single point.
        max_rE, max_rP : float
            Used for 'rates' mode when grid_search=True or as fixed values otherwise.
        w1_name : str
            Weight name to sweep in 'weights' mode.
        w1_vals : array-like
            Values for the weight parameter sweep.
        rE_fixed, rP_fixed : float
            Used for 'weights' mode as fixed operating point.

        Returns
        -------
        out : dict
            Dictionary with fields:
            ['gain','gain_mod','maxEV','maxEV_mod','maxImEV','maxImEV_mod',
            'dE','dP','dS','dV'] and parameter vectors (rE_vec/rP_vec or w1_vals).
        """
        # ---------------------------------------------------------------------
        # 1. Define parameter domain
        # ---------------------------------------------------------------------
        if mode == "rates":
            if grid_search:
                max_rEP = max(self.rE0, self.rP0)
                self.rE_vec = np.arange(max_rEP, 0.5 - self.step_rX, -self.step_rX)
                self.rP_vec = np.arange(0.5, max_rEP + self.step_rX, self.step_rX)
                single_point = False
            else:
                self.rE_vec = np.array([self.rE0])
                self.rP_vec = np.array([self.rP0])
                single_point = True

        elif mode == "weights":
            if w1_name not in self._w_index:
                raise ValueError(f"Invalid weight name {w1_name}.")
            w1_vals = np.asarray(w1_vals, dtype=float)

            if w2_name is None:
                single_point = len(w1_vals) == 1
            else:
                if w2_name not in self._w_index:
                    raise ValueError(f"Invalid weight name {w2_name}.")
                w2_vals = np.asarray(w2_vals, dtype=float)
                single_point = False
        else:
            raise ValueError("mode must be either 'rates' or 'weights'.")

        # ---------------------------------------------------------------------
        # 2. Prepare storage
        # ---------------------------------------------------------------------
        if mode == "rates":
            nE, nP = len(self.rE_vec), len(self.rP_vec)
            shape = (nE, nP)
        else:
            if w2_name is None:
                shape = (len(w1_vals),)
            else:
                shape = (len(w1_vals), len(w2_vals))

        gain         = np.full(shape, np.nan)
        gain_FF      = np.full(shape, np.nan)
        gain_FB      = np.full(shape, np.nan)
        gain_mod     = np.full(shape, np.nan)
        maxEV        = np.full(shape, np.nan)
        maxEV_mod    = np.full(shape, np.nan)
        maxImEV      = np.full(shape, np.nan)
        maxImEV_mod  = np.full(shape, np.nan)
        dE           = np.full(shape, np.nan)
        dP           = np.full(shape, np.nan)
        dS           = np.full(shape, np.nan)
        dV           = np.full(shape, np.nan)
        bE           = np.full(shape, np.nan)
        bP           = np.full(shape, np.nan)
        bS           = np.full(shape, np.nan)
        bV           = np.full(shape, np.nan)

        # ---------------------------------------------------------------------
        # 3. Main loop
        # ---------------------------------------------------------------------
        orig_weights = {name: getattr(self, name) for name in self._w_order}
        rS, rV = self.rS0, self.rV0

        if mode == "rates":
            for i, rE in enumerate(self.rE_vec):
                for j, rP in enumerate(self.rP_vec):
                    res_pre = self._compute_block(mode,rE, rP, rS, rV, add_input=False)
                    if res_pre["singular"]:
                        continue

                    gain[i, j] = res_pre["gain"]
                    gain_FF[i, j] = res_pre["gain_FF"]
                    gain_FB[i, j] = res_pre["gain_FB"]
                    maxEV[i, j] = res_pre["max_ev_real"]
                    maxImEV[i, j] = res_pre["max_ev_imag"]

                    # --- Compute VIP deltas ---
                    L = res_pre["all_L"]
                    (L_EE, L_EP, L_ES, L_EV,
                    L_PE, L_PP, L_PS, L_PV,
                    L_SE, L_SP, L_SS, L_SV,
                    L_VE, L_VP, L_VS, L_VV) = L

                    dE[i, j] = L_EE*self.I_mod_E + L_EP*self.I_mod_P + L_ES*self.I_mod_S + L_EV*self.I_mod_V
                    dP[i, j] = L_PE*self.I_mod_E + L_PP*self.I_mod_P + L_PS*self.I_mod_S + L_PV*self.I_mod_V
                    dS[i, j] = L_SE*self.I_mod_E + L_SP*self.I_mod_P + L_SS*self.I_mod_S + L_SV*self.I_mod_V
                    dV[i, j] = L_VE*self.I_mod_E + L_VP*self.I_mod_P + L_VS*self.I_mod_S + L_VV*self.I_mod_V

                    res_post = self._compute_block(mode,rE, rP, rS, rV, dE[i, j], dP[i, j], dS[i, j], dV[i, j], add_input=True)
                    if not res_post["singular"]:
                        gain_mod[i, j]    = res_post["gain"]
                        maxEV_mod[i, j]   = res_post["max_ev_real"]
                        maxImEV_mod[i, j] = res_post["max_ev_imag"]

                if not single_point:
                    perc = np.ceil(100 * i / len(self.rE_vec))
                    if not perc % 5:
                        print(f"Completed: {perc}%", end="\r")

        else:  # mode == 'weights'
            rE_fixed = self.rE0
            rP_fixed = self.rP0

            if w2_name is None:
                for i, w1 in enumerate(w1_vals):
                    self._assign_weights({w1_name: w1})
                    res = self._compute_block(mode, rE_fixed, rP_fixed, self.rS0, self.rV0)

                    if res["singular"]:
                        continue

                    gain[i]    = res["gain"]
                    gain_FF[i] = res["gain_FF"]
                    gain_FB[i] = res["gain_FB"]
                    maxEV[i]   = res["max_ev_real"]
                    maxImEV[i] = res["max_ev_imag"]
                    bE[i], bP[i], bS[i], bV[i] = res["all_b"]

            else:
                for i, w1 in enumerate(w1_vals):
                    for j, w2 in enumerate(w2_vals):
                        self._assign_weights({w1_name: w1, w2_name: w2})
                        res = self._compute_block(mode, rE_fixed, rP_fixed, self.rS0, self.rV0)

                        if res["singular"]:
                            continue

                        gain[i, j]    = res["gain"]
                        gain_FF[i, j] = res["gain_FF"]
                        gain_FB[i, j] = res["gain_FB"]
                        maxEV[i, j]   = res["max_ev_real"]
                        maxImEV[i, j] = res["max_ev_imag"]
                        bE[i, j], bP[i, j], bS[i, j], bV[i, j] = res["all_b"]

            self._assign_weights(orig_weights)
            all_b = dict(bE=bE, bP=bP, bS=bS, bV=bV)

        # ---------------------------------------------------------------------
        # 4. Assemble result
        # ---------------------------------------------------------------------
        if mode == "rates" and single_point:
            return dict(
                gain=gain[0, 0], gain_mod=gain_mod[0, 0],
                gain_FF=gain_FF[0, 0], gain_FB=gain_FB[0, 0],
                maxEV=maxEV[0, 0], maxEV_mod=maxEV_mod[0, 0],
                maxImEV=maxImEV[0, 0], maxImEV_mod=maxImEV_mod[0, 0],
                dE=dE[0, 0], dP=dP[0, 0], dS=dS[0, 0], dV=dV[0, 0]
            )
        elif mode == "rates":
            return dict(
                rE_vec=self.rE_vec, rP_vec=self.rP_vec,
                gain=gain, gain_mod=gain_mod,
                gain_FF=gain_FF, gain_FB=gain_FB,
                maxEV=maxEV, maxEV_mod=maxEV_mod,
                maxImEV=maxImEV, maxImEV_mod=maxImEV_mod,
                dE=dE, dP=dP, dS=dS, dV=dV
            )
        else:
            out = dict(
                gain=gain,
                gain_FF=gain_FF,
                gain_FB=gain_FB,
                maxEV=maxEV,
                maxImEV=maxImEV,
                all_b=all_b
            )
            out["w1_vals"] = w1_vals
            if w2_name is not None:
                out["w2_vals"] = w2_vals
            return out

    def sweep_arousal(self, theta_vals):
        """
        Sweep the arousal axis at the fixed rate set-point (rE0..rV0).
        Returns gain_FF, gain_FB, their ratio, the arousal-routed mixed gain
        gain_mix = (1-theta)*gain_FF + theta*gain_FB (the stimulus shifts from
        FF to FB along the axis), and the leading eigenvalue (real and
        imaginary part) per theta. Restores the current theta.
        """
        theta_vals = np.asarray(theta_vals, dtype=float)
        theta_orig = self.theta

        gain_FF = np.full(len(theta_vals), np.nan)
        gain_FB = np.full(len(theta_vals), np.nan)
        maxEV   = np.full(len(theta_vals), np.nan)
        maxImEV = np.full(len(theta_vals), np.nan)

        for i, th in enumerate(theta_vals):
            self.set_arousal(th)
            res = self._compute_block("weights", self.rE0, self.rP0, self.rS0, self.rV0)
            if res["singular"]:
                continue
            gain_FF[i] = res["gain_FF"]
            gain_FB[i] = res["gain_FB"]
            maxEV[i]   = res["max_ev_real"]
            maxImEV[i] = res["max_ev_imag"]

        self.set_arousal(theta_orig)
        return dict(theta=theta_vals, gain_FF=gain_FF, gain_FB=gain_FB,
                    ratio=gain_FB / gain_FF,
                    gain_mix=(1 - theta_vals) * gain_FF + theta_vals * gain_FB,
                    maxEV=maxEV, maxImEV=maxImEV)

    def simulate_dynamics(self, tauE=10, tauP=10, tauS=10, tauV=10,
                        dt=0.01, end_sim=1000,
                        t_stim_FF=None, t_stim_FB=None, t_stim_arousal=None):
        """
        Euler simulation with PV->SST depression (state x_SP), the static
        gain-1 VIP->SST synapse, and a slow postsynaptic filter s_SV
        (alpha5-like) on the VIP->SST current.

        Time is in ms and rates in Hz; tau_SP is in seconds and converted
        internally (the depletion drive uses rP/1000 spikes per ms), so STP
        transients evolve on the physiological ~100 ms timescale. Steady
        states match the analytic pSP = U_SP/(1 + U_SP*rP*tau_SP).

        FF step (I_FF_E -> E, I_FF_P -> PV) turns on at t_stim_FF;
        FB step (I_FB_V -> VIP, I_FB_E -> E) turns on at t_stim_FB;
        the arousal-routed step I(theta) = (1-theta)*FF + theta*FB
        (amplitudes I_arousal_* from set_arousal) turns on at
        t_stim_arousal. Pass None to omit an input.
        """
        rE, rP, rS, rV = self.rE0, self.rP0, self.rS0, self.rV0

        # steady-state synaptic states at the set-point
        x_SP = 1 / (1 + self.U_SP * rP * self.tau_SP)
        s_SV = rV
        tau_SP_ms = 1000.0 * self.tau_SP

        # baseline offsets computed WITH the synaptic factors so that
        # (rE0, rP0, rS0, rV0) is a fixed point of the simulated system
        xE0 = (rE / self.mult_f)**(1 / self.power) - (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV)
        xP0 = (rP / self.mult_f)**(1 / self.power) - (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV)
        xS0 = (rS / self.mult_f)**(1 / self.power) - (self.wSE*rE - self.U_SP*x_SP*self.wSP*rP - self.wSS*rS - self.wSV*s_SV)
        xV0 = (rV / self.mult_f)**(1 / self.power) - (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV)

        time = np.arange(0, end_sim + dt, dt)
        rE_trace = np.zeros_like(time)
        rP_trace = np.zeros_like(time)
        rS_trace = np.zeros_like(time)
        rV_trace = np.zeros_like(time)
        x_SP_trace = np.zeros_like(time)
        s_SV_trace = np.zeros_like(time)

        rE_trace[0], rP_trace[0], rS_trace[0], rV_trace[0] = rE, rP, rS, rV
        x_SP_trace[0], s_SV_trace[0] = x_SP, s_SV

        xE_ff, xP_ff = 0.0, 0.0
        xE_fb, xV_fb = 0.0, 0.0

        for i in range(1, len(time)):
            t = time[i]

            if t_stim_FF is not None and np.isclose(t, t_stim_FF, atol=dt/2):
                xE_ff, xP_ff = self.I_FF_E, self.I_FF_P
            if t_stim_FB is not None and np.isclose(t, t_stim_FB, atol=dt/2):
                xE_fb, xV_fb = self.I_FB_E, self.I_FB_V
            if t_stim_arousal is not None and np.isclose(t, t_stim_arousal, atol=dt/2):
                xE_ff, xP_ff = self.I_arousal_E, self.I_arousal_P
                xV_fb = self.I_arousal_V

            # Euler integration
            rE_dot = (-rE + self.mult_f * (self.wEE*rE - self.wEP*rP - self.wES*rS - self.wEV*rV + xE0 + xE_ff + xE_fb)**self.power) / tauE
            rP_dot = (-rP + self.mult_f * (self.wPE*rE - self.wPP*rP - self.wPS*rS - self.wPV*rV + xP0 + xP_ff)**self.power) / tauP
            rS_dot = (-rS + self.mult_f * (self.wSE*rE - self.U_SP*x_SP*self.wSP*rP - self.wSS*rS - self.wSV*s_SV + xS0)**self.power) / tauS
            rV_dot = (-rV + self.mult_f * (self.wVE*rE - self.wVP*rP - self.wVS*rS - self.wVV*rV + xV0 + xV_fb)**self.power) / tauV

            x_SP_dot = (1 - x_SP) / tau_SP_ms - self.U_SP * x_SP * (rP / 1000.0)
            s_SV_dot = (-s_SV + rV) / self.tau_syn_slow

            rE += dt * rE_dot
            rP += dt * rP_dot
            rS += dt * rS_dot
            rV += dt * rV_dot
            x_SP += dt * x_SP_dot
            s_SV += dt * s_SV_dot

            rE = max(rE, 0.0) #rectify
            rP = max(rP, 0.0)
            rS = max(rS, 0.0)
            rV = max(rV, 0.0)

            rE_trace[i], rP_trace[i], rS_trace[i], rV_trace[i] = rE, rP, rS, rV
            x_SP_trace[i], s_SV_trace[i] = x_SP, s_SV

        return dict(time=time, rE=rE_trace, rP=rP_trace, rS=rS_trace,
                    rV=rV_trace, x_SP=x_SP_trace, s_SV=s_SV_trace)
