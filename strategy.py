#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h volume spike and 1d ADX trend filter.
# Long when price breaks above 1h Donchian upper(20) AND 4h volume > 1.5x 20-period average AND 1d ADX > 20.
# Short when price breaks below 1h Donchian lower(20) AND 4h volume > 1.5x 20-period average AND 1d ADX > 20.
# Exit when price returns to 1h Donchian midpoint.
# Uses discrete position size 0.20. Volume confirmation reduces false signals, 1d ADX ensures trending regime.
# Session filter: 08-20 UTC to avoid low-volume Asian session noise.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1h data once before loop for Donchian calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    # === 1h Indicators: Donchian channels (20-period) ===
    upper_20 = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1h, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_1h, middle_20)
    
    # Get 4h data once before loop for volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # === 4h Indicators: Volume spike filter ===
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Get 1d data once before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) for trend filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        middle_val = middle_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol_4h = volume[i // 4]  # Approximate 4h volume from 1h data (every 4th bar)
        
        # Volume filter: current 1h volume > 1.5x 20-period 4h average volume (scaled)
        # Scale 4h MA to 1h: 4h volume represents 4x 1h bars, so divide by 4
        vol_filter = vol > 1.5 * (vol_ma_val / 4.0) if vol_ma_val > 0 else False
        
        # Trend filter: 1d ADX > 20 (trending regime)
        trend_filter = adx_val > 20
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Donchian middle
            if price <= middle_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Donchian middle
            if price >= middle_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper with volume and trend confirmation
            if price > upper_val and vol_filter and trend_filter:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower with volume and trend confirmation
            elif price < lower_val and vol_filter and trend_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Donchian20_4hVolumeSpike_1dADXTrend_Session_V1"
timeframe = "1h"
leverage = 1.0