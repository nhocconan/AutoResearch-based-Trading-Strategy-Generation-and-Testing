#!/usr/bin/env python3
"""
Hypothesis: 1-hour Camarilla pivot breakout with 4-hour trend filter and volume confirmation.
Long when price breaks above Camarilla R3, 4-hour ADX > 25, and volume > 1.5x average.
Short when price breaks below Camarilla S3, 4-hour ADX > 25, and volume > 1.5x average.
Exit when price returns to Camarilla pivot point or ADX < 20.
Designed for 1h timeframe with tight entries (target: 60-150 trades over 4 years) using
4h for signal direction and 1h for entry timing. Session filter (08-20 UTC) reduces noise.
Works in both bull and bear markets by requiring trend confirmation (ADX > 25) for breakouts.
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
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load 4-hour data for ADX - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour ADX (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h),
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)),
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Calculate Camarilla pivot points on 1-hour timeframe - ONCE before loop
    # Using previous bar's high, low, close (standard Camarilla calculation)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar: use current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4.0)  # R3 = pivot + 1.1*(H-L)/4
    s3 = pivot - (range_hl * 1.1 / 4.0)  # S3 = pivot - 1.1*(H-L)/4
    r4 = pivot + (range_hl * 1.1 / 2.0)  # R4 = pivot + 1.1*(H-L)/2
    s4 = pivot - (range_hl * 1.1 / 2.0)  # S4 = pivot - 1.1*(H-L)/2
    
    # Align HTF indicators to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_values)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(pivot[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        r3_val = r3[i]
        s3_val = s3[i]
        pivot_val = pivot[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, strong trend (ADX > 25), volume confirmation
            if (close_val > r3_val and
                adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3, strong trend (ADX > 25), volume confirmation
            elif (close_val < s3_val and
                  adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Camarilla pivot OR trend weakening (ADX < 20)
                if close_val <= pivot_val or adx_val < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Camarilla pivot OR trend weakening (ADX < 20)
                if close_val >= pivot_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3_S3_4hADX_Volume_Breakout"
timeframe = "1h"
leverage = 1.0