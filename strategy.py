#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R combined with 1-week ADX trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold), ADX > 25 (trending), and volume > 1.5x average.
Short when Williams %R crosses below -20 (overbought), ADX > 25 (trending), and volume > 1.5x average.
Exit when Williams %R reverses or ADX < 20 (trend weakening).
Designed for low trade frequency (~15-30/year) to capture mean reversion in strong trends.
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
    
    # Calculate 12-hour Williams %R (14-period)
    high_12h = df_1w['high'].values  # Reuse 1w data for high/low (will be aligned)
    low_12h = df_1w['low'].values
    
    # Get actual 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align HTF indicators to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        williams_r_val = williams_r_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold), strong trend (ADX > 25), volume confirmation
            if (williams_r_val > -80 and williams_r_aligned[i-1] <= -80 and
                adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought), strong trend (ADX > 25), volume confirmation
            elif (williams_r_val < -20 and williams_r_aligned[i-1] >= -20 and
                  adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -50 (momentum loss) OR trend weakening (ADX < 20)
                if (williams_r_val < -50 and williams_r_aligned[i-1] >= -50) or adx_val < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses above -50 (momentum loss) OR trend weakening (ADX < 20)
                if (williams_r_val > -50 and williams_r_aligned[i-1] <= -50) or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1wADX_Volume_Trend"
timeframe = "12h"
leverage = 1.0