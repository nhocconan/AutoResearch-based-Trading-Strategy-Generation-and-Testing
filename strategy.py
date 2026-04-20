#!/usr/bin/env python3
# 4h_ParabolicSAR_Trend_Follower
# Hypothesis: Parabolic SAR (0.02, 0.2) identifies trend direction and trailing stop.
# Long when price > SAR, short when price < SAR. Position size 0.25.
# Works in both bull and bear by following the trend with dynamic stop-loss.
# Low trade frequency due to trend persistence, reducing fee drag.

name = "4h_ParabolicSAR_Trend_Follower"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize arrays
    psar = np.full_like(close, np.nan)
    bull = np.full_like(close, False)  # True for uptrend
    af = np.full_like(close, af_start)
    ep = np.full_like(close, np.nan)  # Extreme point
    
    # Set initial values
    psar[0] = low[0]
    bull[0] = True
    af[0] = af_start
    ep[0] = high[0]
    
    # Calculate PSAR
    for i in range(1, n):
        if bull[i-1]:  # Was in uptrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for reversal
            if low[i] <= psar[i]:
                bull[i] = False  # Reverse to downtrend
                psar[i] = ep[i-1]  # SAR = prior EP
                af[i] = af_start
                ep[i] = low[i]   # New EP = low
            else:
                bull[i] = True   # Stay in uptrend
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Was in downtrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for reversal
            if high[i] >= psar[i]:
                bull[i] = True   # Reverse to uptrend
                psar[i] = ep[i-1]  # SAR = prior EP
                af[i] = af_start
                ep[i] = high[i]  # New EP = high
            else:
                bull[i] = False  # Stay in downtrend
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if np.isnan(psar[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            if bull[i]:  # Uptrend
                signals[i] = 0.25
                position = 1
            else:  # Downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if not bull[i]:  # Trend reversed to downtrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if bull[i]:  # Trend reversed to uptrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals