#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + ADX(14) > 25 + volume spike
# Short when price breaks below Donchian(20) low + ADX(14) > 25 + volume spike
# Exits when price returns to Donchian midpoint or ADX drops below 20
# Designed to capture strong trends with low trade frequency in both bull and bear markets
# Target: 80-150 total trades over 4 years = 20-38/year

name = "4h_Donchian_1dADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(val, period):
        smoothed = np.full_like(val, np.nan, dtype=float)
        if len(val) >= period:
            smoothed[period-1] = np.nansum(val[:period])
            for i in range(period, len(val)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + val[i]
        return smoothed
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth(dx, 14)
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels on 4h data
    donchian_period = 20
    upper_channel = np.full_like(high, np.nan, dtype=float)
    lower_channel = np.full_like(low, np.nan, dtype=float)
    
    for i in range(donchian_period-1, len(high)):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_period)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        middle = middle_channel[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel + strong trend + volume spike
            if (close[i] > upper and adx_val > 25 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel + strong trend + volume spike
            elif (close[i] < lower and adx_val > 25 and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle channel OR trend weakens
            if close[i] < middle or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle channel OR trend weakens
            if close[i] > middle or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals