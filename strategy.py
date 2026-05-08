#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# In strong trends, price breaks out of the 20-period Donchian channel and continues.
# The 1d EMA34 provides a higher-timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts are supported by participation.
# Exit when price returns to the middle of the Donchian channel (mean reversion within the trend).
# Targets 20-40 trades per year (~80-160 total over 4 years) to minimize fee drag.

name = "12h_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel: 20-period high and low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0  # middle of the channel for exit
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_slope = ema34_1d[1:] - ema34_1d[:-1]  # slope: positive = uptrend
    ema34_1d_slope = np.concatenate([[0], ema34_1d_slope])  # align length
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_slope)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_20[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high_20[i]
        low_val = low_20[i]
        mid_val = mid_20[i]
        ema34_val = ema34_1d_aligned[i]
        ema34_slope = ema34_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band, 1d uptrend, volume confirmation
            if close_val > high_val and ema34_slope > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band, 1d downtrend, volume confirmation
            elif close_val < low_val and ema34_slope < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle of Donchian channel or 1d trend turns down
            if close_val < mid_val or ema34_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle of Donchian channel or 1d trend turns up
            if close_val > mid_val or ema34_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals