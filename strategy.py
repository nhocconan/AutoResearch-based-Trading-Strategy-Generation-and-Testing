#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction with 1d EMA50 trend filter and volume spike confirmation
# Uses higher timeframe (4h/1d) for signal direction to reduce trade frequency, 1h only for entry timing precision
# Volume confirmation (>1.5x 20 EMA) filters low-quality breakouts. Discrete sizing 0.20 limits risk and fee drag.
# Works in bull/bear markets: trend filter prevents counter-trend entries. Target: 60-150 trades over 4 years.

name = "1h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian channels to 1h timeframe (completed 4h bar only)
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside trading session
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above 4h Donchian upper + uptrend + volume spike
            if close[i] > highest_high_aligned[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below 4h Donchian lower + downtrend + volume spike
            elif close[i] < lowest_low_aligned[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h Donchian midpoint OR trend changes OR volume drops
            midpoint = (highest_high_aligned[i] + lowest_low_aligned[i]) / 2.0
            if (close[i] < midpoint or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h Donchian midpoint OR trend changes OR volume drops
            midpoint = (highest_high_aligned[i] + lowest_low_aligned[i]) / 2.0
            if (close[i] > midpoint or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals