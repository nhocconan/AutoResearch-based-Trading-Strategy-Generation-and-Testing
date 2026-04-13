#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Volume_Confirmation
Hypothesis: Donchian channel breakouts capture strong momentum moves, while volume confirmation filters false breakouts.
Uses 1-day EMA as trend filter to avoid counter-trend trades. Works in both bull (breakouts above) and bear (breakouts below).
Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Donchian channel (20-period) on 4h data
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper_channel = rolling_max(high, 20)
    lower_channel = rolling_min(low, 20)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above upper Donchian channel
        # 2. Price above daily EMA20 (uptrend filter)
        # 3. Volume expansion
        breakout_long = close[i] > upper_channel[i]
        price_above_ema = close[i] > ema_20_1d_aligned[i]
        long_condition = breakout_long and price_above_ema and volume_expansion[i]
        
        # Short conditions:
        # 1. Price breaks below lower Donchian channel
        # 2. Price below daily EMA20 (downtrend filter)
        # 3. Volume expansion
        breakout_short = close[i] < lower_channel[i]
        price_below_ema = close[i] < ema_20_1d_aligned[i]
        short_condition = breakout_short and price_below_ema and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0