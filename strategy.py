#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly ADX trend filter and daily volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly ADX > 25 AND daily volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND weekly ADX > 25 AND daily volume > 1.5x 20-period average.
# Exit when price crosses below Donchian(10) for long positions or above Donchian(10) for short positions.
# This strategy targets trending moves with trend confirmation to avoid whipsaws, suitable for both bull and bear markets.

name = "1d_Donchian20_WeeklyADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate ADX on weekly data (14-period)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # True Range
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_w - np.roll(high_w, 1)) > (np.roll(low_w, 1) - low_w), 
                       np.maximum(high_w - np.roll(high_w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_w, 1) - low_w) > (high_w - np.roll(high_w, 1)), 
                        np.maximum(np.roll(low_w, 1) - low_w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[tr_period-1] = np.mean(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.mean(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.mean(dm_minus[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # DI values
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(plus_di)
    mask = (plus_di + minus_di) != 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = np.mean(dx[tr_period:2*tr_period])
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align weekly ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Daily volume filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Daily Donchian channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2*14)  # Sufficient warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian(20) high, ADX > 25, volume spike
            long_cond = (close[i] > highest_high_20[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: break below Donchian(20) low, ADX > 25, volume spike
            short_cond = (close[i] < lowest_low_20[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian(10) high
            if close[i] < highest_high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian(10) low
            if close[i] > lowest_low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals