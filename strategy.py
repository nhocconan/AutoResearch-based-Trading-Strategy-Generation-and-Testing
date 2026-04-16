# 12h_Pivot_R1_S1_Breakout_Volume_Confirmation
# Hypothesis: Price breaking above the daily R1 pivot or below the daily S1 pivot with volume confirmation on 12h timeframe captures institutional breakout moves. Uses 1d pivot levels as key support/resistance and requires volume > 1.5x 20-period average for confirmation. Works in both bull and bear markets by trading breakouts in either direction. Target: 50-150 total trades over 4 years (12-37/year).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Pivot Points (Camarilla style) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate pivot points from previous day
    pp = (df_1d['high'].shift(1) + df_1d['low'].shift(1) + df_1d['close'].shift(1)) / 3
    r1 = pp + (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 6
    s1 = pp - (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 6
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure pivot calculation has previous day data
    warmup = 24  # Need at least 1 day of 12h data
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Close position when price returns to pivot zone ===
        if position == 1:  # Long position
            # Exit when price crosses back below R1 (failed breakout)
            if price < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above S1 (failed breakdown)
            if price > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation
            if price > r1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume confirmation
            elif price < s1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0