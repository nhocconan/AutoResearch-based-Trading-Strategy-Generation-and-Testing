# 6h_1d_Aroon_Oscillator_Volume_Filter
# Hypothesis: Aroon oscillator identifies trend strength and direction on 1d timeframe. 
# Combined with volume confirmation on 6h, it captures strong trends while avoiding chop.
# Works in bull/bear by following 1d trend direction. Target: 15-30 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Aroon_Oscillator_Volume_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Aroon oscillator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 6-period volume moving average for confirmation
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    # Calculate Aroon oscillator on 1d (25-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Aroon Up: ((25 - days since 25-day high) / 25) * 100
    aroon_up = np.full(len(high_1d), np.nan)
    for i in range(24, len(high_1d)):
        lookback = high_1d[i-24:i+1]
        high_idx = len(lookback) - 1 - np.argmax(lookback[::-1])  # index of highest value
        aroon_up[i] = ((24 - high_idx) / 24) * 100
    
    # Aroon Down: ((25 - days since 25-day low) / 25) * 100
    aroon_down = np.full(len(low_1d), np.nan)
    for i in range(24, len(low_1d)):
        lookback = low_1d[i-24:i+1]
        low_idx = len(lookback) - 1 - np.argmin(lookback[::-1])  # index of lowest value
        aroon_down[i] = ((24 - low_idx) / 24) * 100
    
    # Aroon Oscillator = Aroon Up - Aroon Down
    aroon_osc = aroon_up - aroon_down
    
    # Align Aroon oscillator to 6h timeframe
    aroon_osc_aligned = align_htf_to_ltf(prices, df_1d, aroon_osc)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(25, n):
        # Skip if any required data is invalid
        if np.isnan(aroon_osc_aligned[i]) or np.isnan(vol_ma_6[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 6-period average
        volume_filter = volume[i] > 1.5 * vol_ma_6[i]
        
        # Aroon oscillator signals: >50 = strong uptrend, <-50 = strong downtrend
        strong_uptrend = aroon_osc_aligned[i] > 50
        strong_downtrend = aroon_osc_aligned[i] < -50
        
        # Entry conditions
        long_entry = strong_uptrend and volume_filter
        short_entry = strong_downtrend and volume_filter
        
        # Exit conditions: trend weakening or reversal
        long_exit = aroon_osc_aligned[i] < 0  # Oscillator falls below zero
        short_exit = aroon_osc_aligned[i] > 0  # Oscillator rises above zero
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals