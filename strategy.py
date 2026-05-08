#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Long when price breaks above 20-period high and 1d EMA34 rising
# Short when price breaks below 20-period low and 1d EMA34 falling
# Volume confirmation: current volume > 1.2x 20-period average
# Donchian provides clear breakout levels, EMA34 filters trend direction, volume confirms strength
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)

name = "4h_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_prev = np.roll(ema34_1d, 1)
    ema34_1d_prev[0] = np.nan
    ema34_1d_rising = ema34_1d > ema34_1d_prev
    ema34_1d_falling = ema34_1d < ema34_1d_prev
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema34_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_rising)
    ema34_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_falling)
    
    # Volume confirmation: current volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1d_rising_aligned[i]) or 
            np.isnan(ema34_1d_falling_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema34_1d_val = ema34_1d_aligned[i]
        rising_val = ema34_1d_rising_aligned[i]
        falling_val = ema34_1d_falling_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: breakout above high_20, 1d uptrend, volume confirmation
            if close_val > high_20_val and rising_val and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below low_20, 1d downtrend, volume confirmation
            elif close_val < low_20_val and falling_val and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below low_20 or 1d trend turns down
            if close_val < low_20_val or not rising_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above high_20 or 1d trend turns up
            if close_val > high_20_val or not falling_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals