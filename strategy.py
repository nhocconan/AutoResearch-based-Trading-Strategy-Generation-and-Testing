#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d ADX25 regime filter and volume confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend) AND volume spike (>1.5x 20-period volume MA).
# Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 AND volume spike.
# Uses 1d EMA13 for Bull/Bear Power calculation (Elder Ray standard).
# ADX filter ensures we only trade in strong trending markets, avoiding whipsaws in ranging conditions.
# Volume spike confirms institutional participation. Designed for 6h timeframe to achieve 50-150 total trades over 4 years.

name = "6h_ElderRay_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:  # Need enough data for ADX
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Bull/Bear Power
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values), 
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)), 
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)  # Avoid division by zero
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade in strong trending markets (ADX > 25)
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND strong trend AND volume spike
            if bull_val > 0 and bear_val < 0 and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND strong trend AND volume spike
            elif bear_val < 0 and bull_val > 0 and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR trend weakens
            if bull_val <= 0 or bear_val >= 0 or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bull Power <= 0 OR trend weakens
            if bear_val >= 0 or bull_val <= 0 or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals