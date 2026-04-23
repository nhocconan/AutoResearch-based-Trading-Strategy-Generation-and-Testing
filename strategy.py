#!/usr/bin/env python3
"""
Hypothesis: 12-hour ATR breakout with 1-day trend filter.
Long when price breaks above close + 2.5 * ATR(14), 1-day ADX > 25, and volume > 1.5x average.
Short when price breaks below close - 2.5 * ATR(14), 1-day ADX > 25, and volume > 1.5x average.
Exit when price crosses the 50-period SMA or ATR multiplier drops below 1.0.
Designed for low trade frequency (~20-30/year) to capture trend continuations in strong trends.
Works in both bull and bear markets by requiring trend confirmation (ADX > 25) for breakout entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
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
    
    # Calculate ATR and SMA on 12h timeframe - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # ATR (14-period) on 12h
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 50-period SMA on 12h
    sma_50_12h = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    sma_50_aligned = align_htf_to_ltf(prices, df_12h, sma_50_12h)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(sma_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        atr_val = atr_aligned[i]
        sma_50_val = sma_50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Price breaks above close + 2.5 * ATR, strong trend (ADX > 25), volume confirmation
            if (close_val > close[i-1] + 2.5 * atr_val and
                adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below close - 2.5 * ATR, strong trend (ADX > 25), volume confirmation
            elif (close_val < close[i-1] - 2.5 * atr_val and
                  adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 50-period SMA OR ATR multiplier drops below 1.0
                if close_val < sma_50_val or (close_val - close[i-1]) < 1.0 * atr_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 50-period SMA OR ATR multiplier drops below 1.0
                if close_val > sma_50_val or (close[i-1] - close_val) < 1.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_ATR_Breakout_1dADX_Volume"
timeframe = "12h"
leverage = 1.0