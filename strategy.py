#!/usr/bin/env python3
"""
Hypothesis: Use 4-hour Donchian channel breakout with 1-day ADX for trend strength and volume confirmation. 
This strategy targets trending markets by entering on breakouts only when the daily trend is strong (ADX > 25) 
and volume confirms the breakout. Exits occur on opposite Donchian band touch or when trend weakens (ADX < 20). 
Designed for 4h timeframe with ~20-40 trades/year to avoid overtrading and work in both bull and bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
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
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    plus_di = 100 * dm_plus_14 / (tr14 + 1e-10)
    minus_di = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4-hour average volume for confirmation
    volume_4h = prices['volume'].values
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = prices['high'].iloc[i]
        low_val = prices['low'].iloc[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        adx_val = adx_aligned[i]
        vol_val = volume_4h[i]
        vol_avg_val = vol_avg_20[i]
        
        # Skip if any value is NaN
        if (np.isnan(donch_high) or np.isnan(donch_low) or 
            np.isnan(adx_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, strong trend (ADX > 25), volume above average
            if close_val > donch_high and adx_val > 25 and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, strong trend (ADX > 25), volume above average
            elif close_val < donch_low and adx_val > 25 and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches Donchian low or trend weakens (ADX < 20)
            if low_val <= donch_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches Donchian high or trend weakens (ADX < 20)
            if high_val >= donch_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_Donchian_ADX_Volume
# Uses 4-hour Donchian channel breakout (20-period) for entry
# Filters by 1-day ADX > 25 for strong trend and volume confirmation
# Exits on opposite Donchian band touch or when ADX < 20 (trend weakening)
# Designed for 4h timeframe with ~20-40 trades/year
name = "4h_Donchian_ADX_Volume"
timeframe = "4h"
leverage = 1.0