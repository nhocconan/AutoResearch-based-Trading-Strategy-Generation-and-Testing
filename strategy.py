#!/usr/bin/env python3
# 1D_WILLIAMS_ALLIGATOR_TREND_FOLLOW
# Hypothesis: Williams Alligator (Jaw=TEETH=LIPS) on 1w timeframe identifies trend direction and strength.
# In strong trends, the three SMAs are well-separated and aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend).
# We enter on 1d timeframe when price crosses above/below all three lines with volume confirmation.
# Works in bull markets (uptrend alignment) and bear markets (downtrend alignment).
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years).

name = "1D_WILLIAMS_ALLIGATOR_TREND_FOLLOW"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Weekly data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams Alligator: three SMAs
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # Using regular SMA as approximation for SMMA (common in practice)
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 1d timeframe (weekly -> daily)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Volume confirmation: 20-period volume SMA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA and Alligator values
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > all three lines (Jaw > Teeth > Lips) AND volume > average
            if (close[i] > jaw_aligned[i] and 
                close[i] > teeth_aligned[i] and 
                close[i] > lips_aligned[i] and
                jaw_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and
                volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < all three lines (Lips < Teeth < Jaw) AND volume > average
            elif (close[i] < jaw_aligned[i] and 
                  close[i] < teeth_aligned[i] and 
                  close[i] < lips_aligned[i] and
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Teeth OR volume drops significantly
            if (close[i] < teeth_aligned[i] or 
                volume[i] < 0.5 * vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Teeth OR volume drops significantly
            if (close[i] > teeth_aligned[i] or 
                volume[i] < 0.5 * vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals