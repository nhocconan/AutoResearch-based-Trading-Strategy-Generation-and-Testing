#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 AND 4h volume > 1.3x average AND 1d ADX > 20.
# Short when price breaks below S1 AND 4h volume > 1.3x average AND 1d ADX > 20.
# Exit when price crosses back to Camarilla pivot point (close).
# Uses Camarilla levels for intraday structure, volume for confirmation, ADX to avoid ranging.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.
# Session filter: 08-20 UTC to avoid low-volume Asian session.

name = "1h_Camarilla_4hVol_1dADX"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot and Camarilla levels from prior day
    # Using previous day's OHLC to avoid look-ahead
    prev_day_high = np.roll(high, 24)  # 24 hours in 1h data
    prev_day_low = np.roll(low, 24)
    prev_day_close = np.roll(close, 24)
    prev_day_open = np.roll(prices['open'].values, 24)
    
    # First valid value after roll
    prev_day_high[0] = high[0]
    prev_day_low[0] = low[0]
    prev_day_close[0] = close[0]
    prev_day_open[0] = prices['open'].values[0]
    
    # Pivot point
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    
    # Camarilla levels
    range_val = prev_day_high - prev_day_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # 4h volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = vol_4h / vol_ma_4h
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # 1d ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx[np.isnan(dx)] = 0
    
    # Align 1d ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(vol_4h_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        in_session = (8 <= hours[i] <= 20)
        
        if position == 0 and in_session:
            # Long conditions: break above R1, volume spike, ADX > 20
            long_cond = (close[i] > r1[i]) and (vol_4h_aligned[i] > 1.3) and (adx_aligned[i] > 20)
            # Short conditions: break below S1, volume spike, ADX > 20
            short_cond = (close[i] < s1[i]) and (vol_4h_aligned[i] > 1.3) and (adx_aligned[i] > 20)
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below pivot
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above pivot
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals