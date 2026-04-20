# #!/usr/bin/env python3
# 1d_1W_Camarilla_R1S1_Breakout_Volume_Spike
# Hypothesis: Daily chart captures 24h price action. Weekly pivot points (R1/S1) act as key support/resistance levels.
# Breakouts above R1 or below S1 with volume confirmation indicate strong momentum. Works in both bull and bear markets:
# - Bull: breakouts above R1 continue uptrend
# - Bear: breakdowns below S1 continue downtrend
# Volume filter ensures breakouts are not false signals. Target: 15-25 trades/year.

name = "1d_1W_Camarilla_R1S1_Breakout_Volume_Spike"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = pivot_1w + (high_1w - low_1w) * 1.1 / 12
    s1_1w = pivot_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Calculate volume average for spike detection (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0 * 20-day average volume
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breaks above weekly R1 with volume spike
            if close[i] > r1_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: breaks below weekly S1 with volume spike
            elif close[i] < s1_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly S1 (reversal to downside)
            if close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly R1 (reversal to upside)
            if close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals