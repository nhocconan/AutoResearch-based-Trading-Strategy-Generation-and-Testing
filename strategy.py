#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
# Donchian breakouts capture momentum; volume confirms strength; 12h EMA ensures alignment with higher timeframe trend.
# Works in bull markets by riding breakouts, in bear by catching breakdowns with trend filter reducing false signals.
# Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
name = "6h_Donchian20_Volume_EMA12"
timeframe = "6h"
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
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with volume and above 12h EMA
            if (close[i] > high_20[i] and 
                volume_confirm[i] and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and below 12h EMA
            elif (close[i] < low_20[i] and 
                  volume_confirm[i] and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian or falls below 12h EMA
            if (close[i] < low_20[i]) or (close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian or rises above 12h EMA
            if (close[i] > high_20[i]) or (close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals