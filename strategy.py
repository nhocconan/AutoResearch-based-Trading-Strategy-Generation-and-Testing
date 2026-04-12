#!/usr/bin/env python3
"""
4h_12h_1d_Turtle_Channel_Breakout_v1
Hypothesis: Use 4-hour Donchian channel (20-period) breakout with 12-hour EMA trend filter and volume confirmation.
Enter long when price breaks above 20-period Donchian high in uptrend (12h close > EMA20) with volume > 1.8x average.
Enter short when price breaks below 20-period Donchian low in downtrend (12h close < EMA20) with volume > 1.8x average.
Exit on trend reversal or price retracement to midpoint of Donchian channel.
Designed to capture strong directional moves in both bull and bear markets while avoiding whipsaws.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_Turtle_Channel_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4-HOUR DONCHIAN CHANNEL (20-period) ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # === 12-HOUR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_4h = align_htf_to_ltf(prices, df_12h, ema20_12h)
    close_12h_4h = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema20_12h_4h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        trend_up = close_12h_4h[i] > ema20_12h_4h[i]
        trend_down = close_12h_4h[i] < ema20_12h_4h[i]
        
        # Long: break above Donchian high in uptrend with volume surge
        long_signal = (trend_up and 
                      close[i] > high_20[i] * 1.001 and  # Break above high
                      vol_ratio[i] > 1.8)
        
        # Short: break below Donchian low in downtrend with volume surge
        short_signal = (trend_down and 
                       close[i] < low_20[i] * 0.999 and  # Break below low
                       vol_ratio[i] > 1.8)
        
        # Exit: trend reversal or price retracement to midpoint
        exit_long = (position == 1 and 
                    (not trend_up or close[i] <= mid_20[i]))
        exit_short = (position == -1 and 
                     (not trend_down or close[i] >= mid_20[i]))
        
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