#!/usr/bin/env python3
"""
4h_1d_supertrend_v2
Strategy: 4h Supertrend with 1-day trend filter and volume confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Combines Supertrend (ATR-based trend following) on 4h with 1-day ADX trend strength filter and volume expansion to capture strong trends while avoiding whipsaws. Designed for moderate trade frequency (20-40/year) to balance signal quality and fee drag. Works in both bull and bear markets by following the trend direction as defined by higher timeframe strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_supertrend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Supertrend calculation ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[0] = upper_band[0] if not np.isnan(upper_band[0]) else close[0]
    direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
        else:
            if close[i] <= supertrend[i-1]:
                # Downtrend
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                # Uptrend
                direction[i] = 1
                supertrend[i] = lower_band[i]
    
    # === 1-day ADX (trend strength filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ADX
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr_1d_smooth = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr_1d_smooth != 0, 100 * dm_plus_smooth / atr_1d_smooth, 0)
    di_minus = np.where(atr_1d_smooth != 0, 100 * dm_minus_smooth / atr_1d_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1-day ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend[i]) or np.isnan(direction[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        adx_value = adx_aligned[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_expanded = volume_current > 1.5 * vol_ma
        
        # Trend strength filter: ADX > 25 indicates strong trend
        strong_trend = adx_value > 25
        
        # Long conditions: price above Supertrend (uptrend) + strong trend + volume expansion
        long_signal = (direction[i] == 1) and strong_trend and volume_expanded
        
        # Short conditions: price below Supertrend (downtrend) + strong trend + volume expansion
        short_signal = (direction[i] == -1) and strong_trend and volume_expanded
        
        # Exit when trend changes or volume dries up
        exit_long = position == 1 and (direction[i] == -1 or not volume_expanded)
        exit_short = position == -1 and (direction[i] == 1 or not volume_expanded)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals