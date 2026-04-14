#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour ADX trend strength filter with 1-day Donchian(20) breakout and volume confirmation.
# The 1-day ADX (>25) identifies strong trends, while the 1-day Donchian breakout captures momentum in the trend direction.
# Volume > 1.8x the 20-period average confirms institutional participation and reduces false breakouts.
# Exit occurs when ADX weakens (<20) or price returns to the midpoint of the Donchian channel.
# This combination aims for 15-25 trades per year per symbol (60-100 total over 4 years), staying within the optimal range to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ADX and Donchian
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX (14 periods)
    adx_len = 14
    if len(df_1d) < adx_len:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initial smoothing (first 14 periods)
    atr[adx_len-1] = np.mean(tr[:adx_len])
    plus_dm_smooth[adx_len-1] = np.mean(plus_dm[:adx_len])
    minus_dm_smooth[adx_len-1] = np.mean(minus_dm[:adx_len])
    
    # Wilder's smoothing
    for i in range(adx_len, len(tr)):
        atr[i] = (atr[i-1] * (adx_len-1) + tr[i]) / adx_len
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (adx_len-1) + plus_dm[i]) / adx_len
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (adx_len-1) + minus_dm[i]) / adx_len
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX smoothing
    adx = np.full_like(dx, np.nan)
    adx[2*adx_len-1] = np.mean(dx[adx_len-1:2*adx_len-1])
    for i in range(2*adx_len, len(dx)):
        adx[i] = (adx[i-1] * (adx_len-1) + dx[i]) / adx_len
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d Donchian channel (20 periods)
    dc_len = 20
    if len(df_1d) < dc_len:
        return np.zeros(n)
    
    dc_upper = pd.Series(high_1d).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low_1d).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    dc_mid = (dc_upper + dc_lower) / 2
    
    dc_upper_aligned = align_htf_to_ltf(prices, df_1d, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1d, dc_lower)
    dc_mid_aligned = align_htf_to_ltf(prices, df_1d, dc_mid)
    
    # Volume confirmation: 1.8x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(2*adx_len, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(dc_upper_aligned[i]) or
            np.isnan(dc_lower_aligned[i]) or
            np.isnan(dc_mid_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: ADX > 25 + breakout above Donchian upper + volume
            if (adx_aligned[i] > 25 and 
                close[i] > dc_upper_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                position = 1
                signals[i] = position_size
            # Enter short: ADX > 25 + breakdown below Donchian lower + volume
            elif (adx_aligned[i] > 25 and 
                  close[i] < dc_lower_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: ADX < 20 or price returns to Donchian midpoint
            if adx_aligned[i] < 20 or close[i] < dc_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: ADX < 20 or price returns to Donchian midpoint
            if adx_aligned[i] < 20 or close[i] > dc_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_ADX_Donchian_Volume_v1"
timeframe = "12h"
leverage = 1.0