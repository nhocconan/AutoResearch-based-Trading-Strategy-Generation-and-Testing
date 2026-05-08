#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA21 trend filter + volume confirmation
# Long when price breaks above 20-day high and 1w EMA21 is rising
# Short when price breaks below 20-day low and 1w EMA21 is falling
# Volume confirmation: current volume > 1.5x 20-day average volume
# Designed for low-frequency trading (target: 30-100 trades over 4 years)
# Works in both bull and bear markets via trend filter and breakout logic

name = "1d_Donchian20_1wEMA21_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate EMA21 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema21_1w_val = ema21_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        # Calculate EMA21 slope for trend direction
        if i >= 21:
            ema21_prev = ema21_1w_aligned[i-1]
            ema21_rising = ema21_1w_val > ema21_prev
            ema21_falling = ema21_1w_val < ema21_prev
        else:
            ema21_rising = False
            ema21_falling = False
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 1w uptrend, volume confirmation
            if high_val > donchian_high_val and ema21_rising and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 1w downtrend, volume confirmation
            elif low_val < donchian_low_val and ema21_falling and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or 1w trend turns down
            if low_val < donchian_low_val or not ema21_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or 1w trend turns up
            if high_val > donchian_high_val or not ema21_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals