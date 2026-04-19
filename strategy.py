#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_Confirm
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter
Donchian channels capture breakout momentum, volume confirms institutional participation,
and 1d EMA50 filter ensures alignment with higher-timeframe trend to avoid false breakouts.
Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year).
Works in bull/bear via trend filter and volatility-adjusted breakouts.
"""

name = "4h_1d_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4h Donchian(20) channels
    # Upper band: 20-period high
    # Lower band: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend
            if (close[i] > donchian_high[i] and 
                volume_confirm[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_confirm[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or trend reverses
            if (close[i] < donchian_low[i]) or (not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or trend reverses
            if (close[i] > donchian_high[i]) or (not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals