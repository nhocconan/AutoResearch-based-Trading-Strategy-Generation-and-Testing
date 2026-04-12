#!/usr/bin/env python3
"""
6h_12h_1d_Donchian_EMA_Vol_Filter
Hypothesis: Use 12h Donchian breakout with 1d EMA trend filter and volume confirmation on 6h timeframe.
Enter long when price breaks above 12h Donchian upper (20) in uptrend (1d close > EMA50) with volume > 1.5x average.
Enter short when price breaks below 12h Donchian lower (20) in downtrend (1d close < EMA50) with volume > 1.5x average.
Exit on trend reversal or price retracement to Donchian middle. Uses 0.25 position sizing.
Designed to capture strong directional moves in both bull and bear markets while avoiding whipsaws.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Donchian_EMA_Vol_Filter"
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
    
    # === 12H DONCHIAN CHANNEL (20) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align to 6h timeframe
    upper_6h = align_htf_to_ltf(prices, df_12h, donchian_upper)
    lower_6h = align_htf_to_ltf(prices, df_12h, donchian_lower)
    middle_6h = align_htf_to_ltf(prices, df_12h, donchian_middle)
    
    # === 1D EMA TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or np.isnan(middle_6h[i]) or 
            np.isnan(ema50_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above upper in uptrend with volume filter
        long_signal = (close[i] > upper_6h[i] and 
                      close[i] > ema50_6h[i] and  # Use current 6h close vs 1d EMA (aligned)
                      vol_ratio[i] > 1.5)
        
        # Short: break below lower in downtrend with volume filter
        short_signal = (close[i] < lower_6h[i] and 
                       close[i] < ema50_6h[i] and  # Use current 6h close vs 1d EMA (aligned)
                       vol_ratio[i] > 1.5)
        
        # Exit: trend reversal or retracement to middle
        exit_long = (position == 1 and 
                    (close[i] <= ema50_6h[i] or  # Trend reversed
                     close[i] <= middle_6h[i]))
        exit_short = (position == -1 and 
                     (close[i] >= ema50_6h[i] or  # Trend reversed
                      close[i] >= middle_6h[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals