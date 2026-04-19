#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h Supertrend trend filter + volume spike confirmation.
# Donchian captures breakout trends, Supertrend confirms direction, volume filter ensures strength.
# Designed for 4h to capture medium-term trends with low trade frequency (<30/year).
# Entry: Price breaks above Donchian upper band + Supertrend uptrend + volume spike.
# Exit: Price breaks below Donchian lower band OR Supertrend downtrend.
# Uses strict conditions to limit trades and avoid overtrading.
name = "4h_Donchian_Supertrend_Volume"
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
    
    # Donchian Channel (20-period)
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Supertrend (12h timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate ATR for Supertrend
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close_12h)
    trend = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upper_band[i-1]:
            trend[i] = 1
        elif close_12h[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if trend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if trend[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_12h, trend)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(trend_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper + Supertrend uptrend + volume spike
            if (close[i] > dc_upper[i] and 
                trend_aligned[i] == 1 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + Supertrend downtrend + volume spike
            elif (close[i] < dc_lower[i] and 
                  trend_aligned[i] == -1 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian lower OR Supertrend downtrend
            if (close[i] < dc_lower[i]) or (trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian upper OR Supertrend uptrend
            if (close[i] > dc_upper[i]) or (trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals