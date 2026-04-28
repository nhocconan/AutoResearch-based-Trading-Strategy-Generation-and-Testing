#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_VolumeFilter
Hypothesis: On daily timeframe, buy when price breaks above 20-day Donchian high with weekly uptrend and volume confirmation; sell when price breaks below 20-day Donchian low with weekly downtrend and volume confirmation. Targets 15-25 trades/year by requiring confluence of price breakout, weekly trend filter, and volume surge to reduce false signals and work in both bull and bear markets.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > weekly EMA50 = bullish, < weekly EMA50 = bearish
        weekly_uptrend = close > ema_50_1w_aligned
        weekly_downtrend = close < ema_50_1w_aligned
        
        # Entry conditions
        long_entry = (close[i] > high_20[i] and 
                     weekly_uptrend[i] and 
                     volume_surge[i])
        
        short_entry = (close[i] < low_20[i] and 
                      weekly_downtrend[i] and 
                      volume_surge[i])
        
        # Exit conditions: reverse on opposite breakout with volume surge
        long_exit = close[i] < low_20[i] and volume_surge[i]
        short_exit = close[i] > high_20[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0