#!/usr/bin/env python3
"""
1h_4h1d_Breakout_Retest
- Uses 4h/1d price structure (higher highs/lows) for directional bias
- Enters on 1h retest of broken 4h/1d swing levels with volume confirmation
- Filters by 1d ADX > 20 to avoid chop
- Strict entry: max 2 signals per day to control frequency
- Target: 15-30 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_swing_high(low, lookback):
    """Find swing highs: higher than N bars before and after."""
    n = len(low)
    swing_high = np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        if low[i] == np.min(low[i-lookback:i+lookback+1]):
            swing_high[i] = low[i]
    return swing_high

def calculate_swing_low(high, lookback):
    """Find swing lows: lower than N bars before and after."""
    n = len(high)
    swing_low = np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        if high[i] == np.max(high[i-lookback:i+lookback+1]):
            swing_low[i] = high[i]
    return swing_low

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index."""
    if len(high) < period * 2:
        return np.full(len(high), np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM using Wilder's smoothing
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    if len(dm_plus) >= period:
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        for i in range(period, len(dm_plus)):
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = np.full(len(dm_plus), np.nan)
    minus_di = np.full(len(dm_minus), np.nan)
    for i in range(period, len(atr)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
    
    # DX and ADX
    dx = np.full(len(plus_di), np.nan)
    for i in range(period, len(plus_di)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.full(len(dx), np.nan)
    if len(dx) >= 2 * period - 1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for structure
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h swing points (structure)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    swing_high_4h = calculate_swing_high(low_4h, 3)  # 3-bar lookback
    swing_low_4h = calculate_swing_low(high_4h, 3)
    
    # Calculate 1d swing points (stronger structure)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    swing_high_1d = calculate_swing_high(low_1d, 2)
    swing_low_1d = calculate_swing_low(high_1d, 2)
    
    # Calculate 1d ADX filter
    adx_14_1d = calculate_adx(high_1d, low_1d, df_1d['close'].values, 14)
    
    # Align 4h/1d structures to 1h
    swing_high_4h_1h = align_htf_to_ltf(prices, df_4h, swing_high_4h)
    swing_low_4h_1h = align_htf_to_ltf(prices, df_4h, swing_low_4h)
    swing_high_1d_1h = align_htf_to_ltf(prices, df_1d, swing_high_1d)
    swing_low_1d_1h = align_htf_to_ltf(prices, df_1d, swing_low_1d)
    adx_14_1d_1h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Volume filter: 1h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > 1.5 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track daily signals to limit frequency
    dates = pd.DatetimeIndex(prices['open_time']).date
    daily_signal_count = {}
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any data unavailable
        if (np.isnan(swing_high_4h_1h[i]) or np.isnan(swing_low_4h_1h[i]) or
            np.isnan(swing_high_1d_1h[i]) or np.isnan(swing_low_1d_1h[i]) or
            np.isnan(adx_14_1d_1h[i])):
            signals[i] = 0.0
            continue
        
        # Check session and volume
        if not (session_filter[i] and vol_filter[i]):
            signals[i] = 0.0
            continue
        
        # Check daily signal limit (max 2 per day)
        current_date = dates[i]
        daily_count = daily_signal_count.get(current_date, 0)
        if daily_count >= 2:
            signals[i] = 0.0
            continue
        
        # Determine structure bias from 4h/1d swings
        # Bullish: price above recent swing lows
        # Bearish: price below recent swing highs
        structure_bullish = close[i] > swing_low_4h_1h[i] and close[i] > swing_low_1d_1h[i]
        structure_bearish = close[i] < swing_high_4h_1h[i] and close[i] < swing_high_1d_1h[i]
        
        if position == 0:
            # Long: bullish structure + ADX > 20
            if structure_bullish and adx_14_1d_1h[i] > 20:
                signals[i] = 0.20
                position = 1
                daily_signal_count[current_date] = daily_count + 1
            # Short: bearish structure + ADX > 20
            elif structure_bearish and adx_14_1d_1h[i] > 20:
                signals[i] = -0.20
                position = -1
                daily_signal_count[current_date] = daily_count + 1
        
        elif position == 1:
            # Long exit: structure turns bearish or ADX weak
            if not structure_bullish or adx_14_1d_1h[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: structure turns bullish or ADX weak
            if not structure_bearish or adx_14_1d_1h[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Breakout_Retest"
timeframe = "1h"
leverage = 1.0