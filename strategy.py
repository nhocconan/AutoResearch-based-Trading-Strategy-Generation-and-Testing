#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume Confirmation
Hypothesis: Weekly pivot levels (R4/S4) act as strong support/resistance in both bull and bear markets.
Breakouts beyond these levels with volume continuation capture strong momentum moves.
Weekly pivot provides structure that works across regimes, while volume filters false breakouts.
Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (once before loop)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    # R3 = H + 2*(P-L), S3 = L - 2*(H-P)
    # R4 = R3 + (H-L), S4 = S3 - (H-L)
    H_w = df_w['high'].values
    L_w = df_w['low'].values
    C_w = df_w['close'].values
    
    P_w = (H_w + L_w + C_w) / 3.0
    R1_w = 2*P_w - L_w
    S1_w = 2*P_w - H_w
    range_w = H_w - L_w
    R2_w = P_w + range_w
    S2_w = P_w - range_w
    R3_w = H_w + 2*(P_w - L_w)
    S3_w = L_w - 2*(H_w - P_w)
    R4_w = R3_w + range_w
    S4_w = S3_w - range_w
    
    # Align weekly pivot levels to 6h (wait for weekly close)
    P_w_a = align_htf_to_ltf(prices, df_w, P_w)
    R4_w_a = align_htf_to_ltf(prices, df_w, R4_w)
    S4_w_a = align_htf_to_ltf(prices, df_w, S4_w)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(P_w_a[i]) or np.isnan(R4_w_a[i]) or np.isnan(S4_w_a[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long breakout: price closes above R4 with volume
            if price > R4_w_a[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S4 with volume
            elif price < S4_w_a[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to weekly pivot or opposite S4
            if price < P_w_a[i] or price > R4_w_a[i] * 1.02:  # slight buffer above R4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly pivot or opposite R4
            if price > P_w_a[i] or price < S4_w_a[i] * 0.98:  # slight buffer below S4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_R4S4_Breakout_Volume"
timeframe = "6h"
leverage = 1.0