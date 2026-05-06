#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels with volume spike confirmation
# Long when price breaks above 1w Camarilla R4 level AND daily volume > 2.0 * avg_volume(20)
# Short when price breaks below 1w Camarilla S4 level AND daily volume > 2.0 * avg_volume(20)
# Exit when price returns to 1w Camarilla midpoint (PP)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Camarilla R4/S4 represent strong breakout levels from 1w structure
# Volume spike filters for institutional participation in breakouts
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "1d_1wCamarilla_R4_S4_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 completed weekly bars for Camarilla
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (based on previous week)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R4 = PP + (High - Low) * 1.1/2
    # S4 = PP - (High - Low) * 1.1/2
    pp = (high_1w + low_1w + close_1w) / 3.0
    r4 = pp + (high_1w - low_1w) * 1.1 / 2.0
    s4 = pp - (high_1w - low_1w) * 1.1 / 2.0
    
    # Align 1w Camarilla levels to 1d timeframe (wait for completed 1w bar)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R4 level with volume spike
            if (close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S4 level with volume spike
            elif (close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1w Camarilla pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1w Camarilla pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals