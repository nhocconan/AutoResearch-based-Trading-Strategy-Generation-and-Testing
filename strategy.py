#!/usr/bin/env python3
"""
12h_1d_Price_Channel_Breakout_v1
Hypothesis: Use 1-day high/low price channel with volume confirmation on 12h timeframe.
Long when price breaks above 20-period 1d high with volume > 1.3x 20-period average,
short when breaks below 20-period 1d low with volume > 1.3x 20-period average.
Exit on opposite channel touch or volatility contraction.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drift.
Works in bull via breakouts above resistance, in bear via breakdowns below support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Price_Channel_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for price channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period daily high and low channels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    high_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align daily channels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any data invalid
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume filter
        long_breakout = close[i] > high_20_aligned[i] and vol_ratio[i] > 1.3
        short_breakout = close[i] < low_20_aligned[i] and vol_ratio[i] > 1.3
        
        # Exit conditions: opposite channel touch
        long_exit = close[i] < low_20_aligned[i]
        short_exit = close[i] > high_20_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals