#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout + Volume Spike + 1d ADX Trend Filter
Based on Camarilla pivot levels calculated from daily high/low/close.
Breakout above R1 with volume spike and bullish trend (ADX>25) -> long.
Breakdown below S1 with volume spike and bearish trend (ADX>25) -> short.
Uses 1d ADX as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency with clear breakout edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (using previous day's H/L/C)
    # R1 = C + (H-L)*1.1/2
    # S1 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 2
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate daily ADX for trend strength
    # +DM = max(0, high[t] - high[t-1]) if high[t] - high[t-1] > low[t-1] - low[t] else 0
    # -DM = max(0, low[t-1] - low[t]) if low[t-1] - low[t] > high[t] - high[t-1] else 0
    # TR = max(high[t] - low[t], abs(high[t] - close[t-1]), abs(low[t] - close[t-1]))
    # ADX = smoothed DX over period
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    def smooth_series(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    tr_smoothed = smooth_series(tr, atr_period)
    plus_dm_smoothed = smooth_series(plus_dm, atr_period)
    minus_dm_smoothed = smooth_series(minus_dm, atr_period)
    
    # DI values
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_series(dx, atr_period)
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: breakout above R1, volume spike, strong trend
            if price > r1_aligned[i] and volume_spike[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1, volume spike, strong trend
            elif price < s1_aligned[i] and volume_spike[i] and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks back below R1 or trend weakens
            if price < r1_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks back above S1 or trend weakens
            if price > s1_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeSpike_1dADX"
timeframe = "12h"
leverage = 1.0