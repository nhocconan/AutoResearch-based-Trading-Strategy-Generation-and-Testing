#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Long when price breaks above upper Donchian channel (20-period high) and 1d EMA50 rising
# Short when price breaks below lower Donchian channel (20-period low) and 1d EMA50 falling
# Volume confirmation: current volume > 1.5x 20-period average
# Donchian channels provide clear breakout levels, EMA50 filters trend direction
# Volume confirmation reduces false breakouts
# Targets 50-150 total trades over 4 years (12-37/year) for optimal fee drag

name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)  # Previous value for trend direction
    ema50_1d_prev[0] = ema50_1d[0]  # Handle first element
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window, 50)  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        ema50_1d_val = ema50_1d_aligned[i]
        ema50_rising_val = ema50_rising_aligned[i]
        ema50_falling_val = ema50_falling_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel, 1d EMA50 rising, volume confirmation
            if close_val > upper_val and ema50_rising_val and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel, 1d EMA50 falling, volume confirmation
            elif close_val < lower_val and ema50_falling_val and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel or 1d EMA50 falling
            if close_val < lower_val or ema50_falling_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel or 1d EMA50 rising
            if close_val > upper_val or ema50_rising_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals