#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and daily ATR filter.
# Works in bull markets by capturing breakouts and in bear markets by avoiding false breakouts
# via volume confirmation and ATR-based regime filter. Targets 20-40 trades/year.
name = "4h_Donchian20_Volume_ATRFilter_v1"
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
    
    # Daily ATR for regime filter (ATR(14) < 20-period median = low volatility regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    low_volatility = atr_1d < atr_median_1d  # low volatility regime
    
    # Align to 4h timeframe
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(low_volatility_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation in low volatility regime
            if (close[i] > highest_high[i] and 
                volume_confirm[i] and 
                low_volatility_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation in low volatility regime
            elif (close[i] < lowest_low[i] and 
                  volume_confirm[i] and 
                  low_volatility_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches Donchian low or volatility increases
            if (close[i] < lowest_low[i]) or (not low_volatility_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches Donchian high or volatility increases
            if (close[i] > highest_high[i]) or (not low_volatility_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals