#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout + weekly trend filter + volume confirmation.
Uses weekly ADX for trend strength and daily Donchian channels for entry.
Long when price breaks above upper Donchian channel in strong uptrend (ADX>25),
short when price breaks below lower Donchian channel in strong downtrend (ADX>25).
Volume must be above 20-period average to confirm breakout.
Exit when price crosses opposite Donchian band or ADX falls below 20.
Designed for low trade frequency with strong trend filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_weekly_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY ADX TREND STRENGTH (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Calculate ADX
    # True Range
    tr1 = w_high - w_low
    tr2 = np.abs(w_high - np.roll(w_close, 1))
    tr3 = np.abs(w_low - np.roll(w_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((w_high - np.roll(w_high, 1)) > (np.roll(w_low, 1) - w_low), 
                       np.maximum(w_high - np.roll(w_high, 1), 0), 0)
    dm_minus = np.where((np.roll(w_low, 1) - w_low) > (w_high - np.roll(w_high, 1)), 
                        np.maximum(np.roll(w_low, 1) - w_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(arr)):
            result[i] = result[i-1] * (1 - 1/period) + arr[i] * (1/period)
        return result
    
    atr_period = 14
    tr_smooth = wilder_smooth(tr, atr_period)
    dm_plus_smooth = wilder_smooth(dm_plus, atr_period)
    dm_minus_smooth = wilder_smooth(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, dm_plus_smooth / tr_smooth * 100, 0)
    di_minus = np.where(tr_smooth != 0, dm_minus_smooth / tr_smooth * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilder_smooth(dx, atr_period)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === DAILY DONCHIAN CHANNELS (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Donchian channels (20-period)
    upper_channel = np.full_like(d_high, np.nan)
    lower_channel = np.full_like(d_low, np.nan)
    
    for i in range(len(d_high)):
        if i >= 19:  # 20-period lookback
            upper_channel[i] = np.max(d_high[i-19:i+1])
            lower_channel[i] = np.min(d_low[i-19:i+1])
    
    # Align to 12h timeframe (use previous day's channels)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(adx_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Trend strength filter
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: price crosses below lower channel OR trend weakens
            if close[i] < lower_aligned[i] or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel OR trend weakens
            if close[i] > upper_aligned[i] or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation and strong trend
            if volume[i] <= vol_ma[i] or not strong_trend:
                signals[i] = 0.0
                continue
            
            # Entry: breakout in direction of trend
            if close[i] > upper_aligned[i]:  # Break above upper channel -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lower_aligned[i]:  # Break below lower channel -> short
                position = -1
                signals[i] = -0.25
    
    return signals