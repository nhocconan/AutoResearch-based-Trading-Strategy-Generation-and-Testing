#!/usr/bin/env python3
"""
4h_Combined_Trend_Momentum_Volume
Hypothesis: Combines 4h price action with 1d trend filter and volume spike for high-probability momentum entries.
Uses ADX for trend strength, RSI for momentum, and volume confirmation to filter false breakouts.
Designed for low trade frequency (~20-30 trades/year) to minimize fee decay, working in both bull and bear markets.
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
    
    # 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period+1])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = smooth_wilder(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h RSI for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan
    
    # 4h price channels for breakout detection
    highest_20 = np.full_like(close, np.nan)
    lowest_20 = np.full_like(close, np.nan)
    
    for i in range(20, len(close)):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Volume confirmation
    vol_avg = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_avg[i] = np.mean(volume[i-20:i])
    vol_avg[:20] = np.nan
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    start_idx = max(30, 20)  # ADX and channel lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        rsi_val = rsi[i]
        high_20 = highest_20[i]
        low_20 = lowest_20[i]
        vol_spike = volume_spike[i]
        
        # Only trade when trend is strong enough
        if adx_val < 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish breakout with momentum and volume
            if close[i] > high_20 and rsi_val > 50 and rsi_val < 70 and vol_spike:
                signals[i] = size
                position = 1
            # Short: bearish breakdown with momentum and volume
            elif close[i] < low_20 and rsi_val < 50 and rsi_val > 30 and vol_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: momentum fade or trend weakening
            if rsi_val < 40 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: momentum fade or trend weakening
            if rsi_val > 60 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Combined_Trend_Momentum_Volume"
timeframe = "4h"
leverage = 1.0