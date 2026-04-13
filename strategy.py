#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Volume_Trend
Hypothesis: Trade reversals at Camarilla pivot levels (H3/L3) on 4h using 12h trend filter and volume confirmation.
Camarilla levels from the prior 12h period act as intraday support/resistance. Price approaching H3/L3 with
volume exhaustion often reverses. The 12h EMA20 ensures we trade in direction of higher timeframe trend,
avoiding counter-trend whipsaws. Target: 25-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation (using prior completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 12h bar (H-L range)
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2 (inner band)
    # H6 = C + 1.5*(H-L)/2, L6 = C - 1.5*(H-L)/2 (outer band)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Use prior completed 12h bar to avoid look-ahead
    range_12h = high_12h - low_12h
    camarilla_h4 = close_12h + 1.1 * range_12h / 2.0
    camarilla_l4 = close_12h - 1.1 * range_12h / 2.0
    camarilla_h6 = close_12h + 1.5 * range_12h / 2.0
    camarilla_l6 = close_12h - 1.5 * range_12h / 2.0
    
    # Align Camarilla levels to 4h (using prior completed 12h bar)
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    h6_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h6)
    l6_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l6)
    
    # Get 12h EMA20 for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume condition: current volume < 0.5x 20-period average (low volume = exhaustion)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_exhaustion = volume < (vol_ma_20 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h6_aligned[i]) or np.isnan(l6_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_exhaustion[i])):
            signals[i] = 0.0
            continue
        
        # Long: price near L4/L6 with volume exhaustion and above 12h EMA20
        near_support = (close[i] <= l4_aligned[i] * 1.002) or (close[i] <= l6_aligned[i] * 1.002)
        long_condition = near_support and volume_exhaustion[i] and (close[i] > ema_20_aligned[i])
        
        # Short: price near H4/H6 with volume exhaustion and below 12h EMA20
        near_resistance = (close[i] >= h4_aligned[i] * 0.998) or (close[i] >= h6_aligned[i] * 0.998)
        short_condition = near_resistance and volume_exhaustion[i] and (close[i] < ema_20_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_12h_Camarilla_Pivot_Volume_Trend"
timeframe = "4h"
leverage = 1.0