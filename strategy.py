#!/usr/bin/env python3
"""
Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with 1-day ADX regime filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), and volume > 1.5x average.
Short when Bull Power < 0, Bear Power > 0, ADX > 25 (trending), and volume > 1.5x average.
Exit when trend weakens (ADX < 20) or Elder Ray signals reverse.
Designed for low trade frequency (~15-30/year) to capture strong trends while minimizing whipsaws.
Works in both bull and bear markets by requiring strong trend confirmation (ADX > 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA13 and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1-day ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
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
    
    # Align HTF indicators to lower timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_val = ema13_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        # Calculate Elder Ray components
        bull_power = close[i] - ema13_val
        bear_power = ema13_val - close[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, strong trend (ADX > 25), volume confirmation
            if (bull_power > 0 and bear_power < 0 and
                adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, strong trend (ADX > 25), volume confirmation
            elif (bull_power < 0 and bear_power > 0 and
                  adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Elder Ray turns bearish OR trend weakening (ADX < 20)
                if (bull_power <= 0 or bear_power >= 0) or adx_val < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Elder Ray turns bullish OR trend weakening (ADX < 20)
                if (bull_power >= 0 or bear_power <= 0) or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dADX_Volume_Trend"
timeframe = "6h"
leverage = 1.0