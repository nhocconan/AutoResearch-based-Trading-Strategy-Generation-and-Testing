#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout + 12h trend filter + volume confirmation.
Uses 4-hour Donchian channel (20-period) for breakout entries, confirmed by
12-hour EMA trend direction and volume spikes to filter false breakouts.
Designed to work in both bull and bear markets by requiring alignment with
higher timeframe trend. Targets 20-50 trades per year to minimize fee drag.
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
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4-hour Donchian channel (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 12h data for trend filter (EMA 21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume confirmation
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        trend_up = close[i] > ema_12h_aligned[i]
        trend_down = close[i] < ema_12h_aligned[i]
        vol_confirm = vol_spike[i]
        
        long_entry = breakout_up and trend_up and vol_confirm
        short_entry = breakout_down and trend_down and vol_confirm
        
        # Exit when price crosses back through Donchian channel middle
        donchian_middle = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
        exit_long = position == 1 and close[i] < donchian_middle
        exit_short = position == -1 and close[i] > donchian_middle
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_trend_volume"
timeframe = "4h"
leverage = 1.0