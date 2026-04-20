# US-based AI Researcher
#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Conservative_v1
Concept: 4h breakout at daily Camarilla R1/S1 levels with volume confirmation and volume filter.
- Long: Close above R1 + volume > 1.5x avg volume (20-period) + price > VWAP (4h)
- Short: Close below S1 + volume > 1.5x avg volume (20-period) + price < VWAP (4h)
- Exit: Close crosses VWAP (4h) in opposite direction
- Position sizing: 0.25
- Target: 20-50 trades/year (80-200 total over 4 years)
- Works in bull/bear: Camarilla levels adapt to volatility, volume confirms breakout strength, VWAP filters false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_R1_S1_Breakout_Volume_Conservative_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h: VWAP calculation (typical price * volume) / cumulative volume ===
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap_numerator = (typical_price * prices['volume']).cumsum()
    vwap_denominator = prices['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap = vwap.values
    
    # === 4h: Volume average (20-period) ===
    volume_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # === Daily: Calculate Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vwap_val = vwap[i]
        volume_ma_val = volume_ma[i]
        close_val = prices['close'].iloc[i]
        volume_val = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vwap_val) or 
            np.isnan(volume_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R1 + volume > 1.5x avg volume + price > VWAP
            if (close_val > r1_val and 
                volume_val > 1.5 * volume_ma_val and 
                close_val > vwap_val):
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 + volume > 1.5x avg volume + price < VWAP
            elif (close_val < s1_val and 
                  volume_val > 1.5 * volume_ma_val and 
                  close_val < vwap_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close crosses below VWAP
            if close_val < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close crosses above VWAP
            if close_val > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals