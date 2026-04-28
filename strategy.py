#!/usr/bin/env python3
"""
12h_Donchian20_WeeklyTrend_VolumeS
Hypothesis: 12-hour breakouts at 20-period Donchian channels with weekly trend filter and volume confirmation. Targets 15-35 trades/year by requiring strong breakouts, weekly trend alignment, and volume surge to reduce false signals and work in both bull and bear markets.
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
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels from previous 12h bar
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all higher timeframe data to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Trend filter: price > EMA34 = bullish, < EMA34 = bearish
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above Donchian high + weekly uptrend + volume surge
        long_entry = (close[i] > donchian_high_aligned[i] and 
                     trend_up[i] and 
                     volume_surge[i])
        
        # Short: price breaks below Donchian low + weekly downtrend + volume surge
        short_entry = (close[i] < donchian_low_aligned[i] and 
                      trend_down[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < donchian_low_aligned[i] and volume_surge[i]
        short_exit = close[i] > donchian_high_aligned[i] and volume_surge[i]
        
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

name = "12h_Donchian20_WeeklyTrend_VolumeS"
timeframe = "12h"
leverage = 1.0