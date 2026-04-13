# Your solution
#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_Volume
Hypothesis: Breakout above R4 or below S4 of weekly Camarilla levels with volume expansion.
Weekly Camarilla provides stronger support/resistance than daily, reducing false breakouts.
Volume confirmation ensures institutional participation. Works in both bull (breakouts up)
and bear (breakdowns down) markets. Target: 8-15 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_val = high - low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    C = close
    H = high
    L = low
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    R1_1w, R2_1w, R3_1w, R4_1w, S1_1w, S2_1w, S3_1w, S4_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    
    # Align all data to daily timeframe
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_1w_aligned[i]) or np.isnan(S4_1w_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above R4 with volume expansion
        long_condition = (close[i] > R4_1w_aligned[i]) and volume_expansion[i]
        
        # Short: breakdown below S4 with volume expansion
        short_condition = (close[i] < S4_1w_aligned[i]) and volume_expansion[i]
        
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

name = "1d_1w_Camarilla_Breakout_Volume"
timeframe = "1d"
leverage = 1.0