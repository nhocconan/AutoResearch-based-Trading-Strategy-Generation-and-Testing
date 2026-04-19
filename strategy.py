#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h MA trend + volume spike
# Donchian breakout provides clear entry/exit signals, 12h MA filters for trend direction
# Volume spike confirms breakout strength, reducing false signals
# Designed for 4h timeframe to capture medium-term trends with controlled trade frequency
# Entry: Price breaks above Donchian upper band (20-period) + 12h MA up + volume spike
# Exit: Price breaks below Donchian lower band (20-period) OR 12h MA down
# Uses strict conditions to limit trades (~20-40/year) and avoid overtrading
name = "4h_Donchian_MA12_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h MA trend (using 12h data)
    df_12h = get_htf_data(prices, '12h')
    ma_12h = pd.Series(df_12h['close']).rolling(window=20, min_periods=20).mean().values
    ma_12h_aligned = align_htf_to_ltf(prices, df_12h, ma_12h)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 12h MA up + volume spike
            if (close[i] > donchian_high[i] and 
                ma_12h_aligned[i] > ma_12h_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 12h MA down + volume spike
            elif (close[i] < donchian_low[i] and 
                  ma_12h_aligned[i] < ma_12h_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low OR 12h MA down
            if (close[i] < donchian_low[i]) or (ma_12h_aligned[i] < ma_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high OR 12h MA up
            if (close[i] > donchian_high[i]) or (ma_12h_aligned[i] > ma_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals