#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Breakout_VolumeTrend_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate ADX for trend strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Calculate Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(tr, np.nan)
    minus_dm_smooth = np.full_like(tr, np.nan)
    
    # First value is simple average
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            atr[i] = atr[i-1] * (1 - alpha) + (tr[i] * alpha)
            plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - alpha) + (plus_dm[i] * alpha)
            minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - alpha) + (minus_dm[i] * alpha)
    
    # Calculate Directional Indicators
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(period, len(tr)):
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.full_like(tr, np.nan)
    if len(tr) >= 2 * period - 1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(tr)):
            adx[i] = adx[i-1] * (1 - alpha) + (dx[i] * alpha)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h: Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h: Volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        adx_val = adx_aligned[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        vol_ma_val = vol_ma[i]
        current_close = prices['close'].iloc[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(highest_high_val) or 
            np.isnan(lowest_low_val) or np.isnan(vol_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition
        vol_condition = current_volume > 1.5 * vol_ma_val
        
        if position == 0:
            # Long conditions:
            # 1. ADX > 25 (strong trend)
            # 2. Price breaks above 20-period high
            # 3. Volume confirmation
            if (adx_val > 25 and
                current_close > highest_high_val and
                vol_condition):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. ADX > 25 (strong trend)
            # 2. Price breaks below 20-period low
            # 3. Volume confirmation
            elif (adx_val > 25 and
                  current_close < lowest_low_val and
                  vol_condition):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below 20-period low (trend reversal)
            # 2. ADX falls below 20 (trend weakening)
            if (current_close < lowest_low_val or
                adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above 20-period high (trend reversal)
            # 2. ADX falls below 20 (trend weakening)
            if (current_close > highest_high_val or
                adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals