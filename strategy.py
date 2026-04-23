#!/usr/bin/env python3
"""
Hypothesis: 1-day Bollinger Band squeeze breakout with 1-week ADX trend filter.
Long when price breaks above upper BB with weekly ADX > 25 and volume > 1.5x average.
Short when price breaks below lower BB with weekly ADX > 25 and volume > 1.5x average.
Exit when price returns to middle BB or weekly ADX < 20.
Designed for low trade frequency (~10-25/year) to capture breakouts in trending markets.
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
    
    # Load 1-week data for ADX - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
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
    
    # Calculate Bollinger Bands on 1-day - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Middle Band (20-period SMA)
    middle_bb = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Standard Deviation (20-period)
    std_dev = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    
    # Upper and Lower Bands (2 standard deviations)
    upper_bb = middle_bb + 2 * std_dev
    lower_bb = middle_bb - 2 * std_dev
    
    # Align HTF indicators to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_bb)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        middle_val = middle_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Price breaks above upper BB, strong trend (ADX > 25), volume confirmation
            if (close_val > upper_val and
                adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB, strong trend (ADX > 25), volume confirmation
            elif (close_val < lower_val and
                  adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle BB OR trend weakening (ADX < 20)
                if close_val <= middle_val or adx_val < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle BB OR trend weakening (ADX < 20)
                if close_val >= middle_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_BB_Squeeze_1wADX_Volume"
timeframe = "1d"
leverage = 1.0