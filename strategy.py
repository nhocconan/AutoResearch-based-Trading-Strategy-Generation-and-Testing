#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# Donchian breakout captures trend continuation; ADX filters for trending markets only (ADX > 25)
# Volume confirmation ensures breakouts have institutional participation
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets
# Target: 25-50 trades/year to minimize fee drag while capturing significant moves

name = "4h_Donchian20_1dADX_Trend_Volume"
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
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper_channel = rolling_max(high, 20)
    lower_channel = rolling_min(low, 20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (14-period)
    def smoothed_avg(arr, period):
        res = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return res
        # First value: simple average
        res[period-1] = np.nanmean(arr[1:period]) if np.isnan(arr[0]) else np.mean(arr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + (arr[i] if not np.isnan(arr[i]) else 0)) / period
        return res
    
    atr = smoothed_avg(tr, 14)
    plus_di = 100 * smoothed_avg(plus_dm, 14) / atr
    minus_di = 100 * smoothed_avg(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smoothed_avg(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        adx_val = adx_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel, ADX > 25 (trending), volume confirmation
            if close_val > upper_val and adx_val > 25 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel, ADX > 25 (trending), volume confirmation
            elif close_val < lower_val and adx_val > 25 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel or ADX drops below 20 (losing trend)
            if close_val < lower_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel or ADX drops below 20 (losing trend)
            if close_val > upper_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals