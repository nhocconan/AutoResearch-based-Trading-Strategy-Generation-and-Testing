#!/usr/bin/env python3
"""
1D Weekly Range Breakout with Volume Spike and ADX Trend Filter
Long: Price breaks above prior weekly high + volume > 2x 1d volume mean + ADX > 25
Short: Price breaks below prior weekly low + volume > 2x 1d volume mean + ADX > 25
Exit: Opposite break of prior weekly level
Uses ADX to filter choppy markets and volume spike to confirm breakout strength
Target: 10-20 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for prior high/low
    df_weekly = get_htf_data(prices, '1w')
    prior_weekly_high = df_weekly['high'].shift(1)  # Prior week's high
    prior_weekly_low = df_weekly['low'].shift(1)    # Prior week's low
    
    prior_weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, prior_weekly_high.values)
    prior_weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, prior_weekly_low.values)
    
    # ADX(14) on 1d timeframe for trend strength
    df_1d = get_htf_data(prices, '1d')
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
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_series(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_series(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume mean on 1d for spike detection
    volume_1d = df_1d['volume'].values
    volume_mean_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(len(volume_1d)):
        if i >= 20:
            volume_mean_1d[i] = np.mean(volume_1d[max(0, i-20):i])
    
    volume_mean_aligned = align_htf_to_ltf(prices, df_1d, volume_mean_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_weekly_high_aligned[i]) or np.isnan(prior_weekly_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_mean_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_mean = volume_mean_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: break above prior weekly high + volume spike + ADX > 25
            if price > prior_weekly_high_aligned[i] and vol > 2.0 * vol_mean and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below prior weekly low + volume spike + ADX > 25
            elif price < prior_weekly_low_aligned[i] and vol > 2.0 * vol_mean and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below prior weekly low
            if price < prior_weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior weekly high
            if price > prior_weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1D_WeeklyRange_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0