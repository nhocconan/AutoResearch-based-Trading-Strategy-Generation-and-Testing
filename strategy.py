#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with Donchian(20) breakout + volume confirmation + 12h EMA34 trend filter.
Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and 12h EMA34 > EMA100.
Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average and 12h EMA34 < EMA100.
Donchian channels capture volatility-based breakouts; volume confirms institutional participation; 12h EMA filter ensures alignment with medium-term trend.
Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag. Uses discrete sizing 0.25.
"""

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
    
    # Calculate Donchian(20) channels
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 and EMA100
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema100_12h = close_12h_series.ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Calculate 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema100_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for Donchian(20) and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_rolling[i]) or np.isnan(low_rolling[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(ema100_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian(20) high with volume and bullish trend (EMA34 > EMA100)
            if (close[i] > high_rolling[i] and 
                volume_confirmed and 
                ema34_12h_aligned[i] > ema100_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low with volume and bearish trend (EMA34 < EMA100)
            elif (close[i] < low_rolling[i] and 
                  volume_confirmed and 
                  ema34_12h_aligned[i] < ema100_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian(20) midpoint or trend turns bearish
            donchian_mid = (high_rolling[i] + low_rolling[i]) / 2
            if (close[i] < donchian_mid or 
                ema34_12h_aligned[i] < ema100_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian(20) midpoint or trend turns bullish
            donchian_mid = (high_rolling[i] + low_rolling[i]) / 2
            if (close[i] > donchian_mid or 
                ema34_12h_aligned[i] > ema100_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMA"
timeframe = "4h"
leverage = 1.0