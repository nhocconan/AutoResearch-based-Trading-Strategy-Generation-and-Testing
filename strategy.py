#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_With_Volume_Filter
Hypothesis: Use 1d Donchian channel (20-day) breakouts with volume confirmation on 12h timeframe.
Long when price breaks above upper band with volume > 1.5x 20-bar average.
Short when price breaks below lower band with volume > 1.5x 20-bar average.
Exit when price crosses the 10-day SMA of closing price.
Designed for 12h timeframe to capture multi-day trends with ~15-30 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume filter reduces false breakouts and improves signal quality.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data once for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian channel (20-period)
    upper = np.full_like(high_1d, np.nan)
    lower = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    
    # Shift to align with current day (channels are based on previous 20 days)
    upper = np.roll(upper, 1)
    lower = np.roll(lower, 1)
    upper[0] = np.nan
    lower[0] = np.nan
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # 10-day SMA for exit
    sma_10 = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().values
    sma_10_aligned = align_htf_to_ltf(prices, df_1d, sma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(sma_10_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above upper band + volume confirmation
            if price > upper_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + volume confirmation
            elif price < lower_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 10-day SMA
            if price < sma_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 10-day SMA
            if price > sma_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian_Breakout_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0