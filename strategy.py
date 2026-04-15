#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF context
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_12h = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    lowest_12h = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align 12h Donchian to 4h timeframe
    highest_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_12h)
    lowest_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_12h)
    
    # Calculate 12h ATR for volatility filter
    prev_close_12h = np.concatenate([[close_12h[0]], close_12h[:-1]])
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - prev_close_12h),
                                   np.abs(low_12h - prev_close_12h)))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_ratio_12h = atr_12h / close_12h
    
    # Align 12h ATR ratio to 4h timeframe
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume / (vol_ma_12h + 1e-10)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_12h_aligned[i]) or np.isnan(lowest_12h_aligned[i]) or
            np.isnan(atr_ratio_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout logic with volume and volatility confirmation
        # Long when price breaks above 12h Donchian high with volume spike and moderate volatility
        if (close[i] > highest_12h_aligned[i] and
            vol_ratio_12h_aligned[i] > 1.5 and  # Volume confirmation
            atr_ratio_12h_aligned[i] > 0.008 and  # Minimum volatility
            atr_ratio_12h_aligned[i] < 0.025):   # Maximum volatility filter
            signals[i] = 0.25
        # Short when price breaks below 12h Donchian low with volume spike and moderate volatility
        elif (close[i] < lowest_12h_aligned[i] and
              vol_ratio_12h_aligned[i] > 1.5 and  # Volume confirmation
              atr_ratio_12h_aligned[i] > 0.008 and  # Minimum volatility
              atr_ratio_12h_aligned[i] < 0.025):   # Maximum volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian_Breakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0