#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# In trending markets (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# In ranging markets (ADX < 20): fade extremes - short when Bull Power > 0.5*ATR, long when Bear Power < -0.5*ATR
# Volume confirmation: require volume > 1.2x average to avoid false signals
# Designed for low trade frequency (target 15-35/year) with clear regime adaptation
# Works in bull (trend following) and bear (mean reversion in ranges) markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA13 for Elder Ray calculation (13-period)
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13
    
    # ADX for regime detection (14-period)
    # True Range
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            continue
        
        # Regime detection
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        # Volume filter
        vol_ok = volume[i] > 1.2 * vol_avg_aligned[i]
        
        if is_trending and vol_ok:
            # Trending market: trend following
            # Long when Bull Power > 0 and increasing
            if (bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and 
                position <= 0):
                position = 1
                signals[i] = base_size
            
            # Short when Bear Power < 0 and decreasing
            elif (bear_power_aligned[i] < 0 and 
                  bear_power_aligned[i] < bear_power_aligned[i-1] and 
                  position >= 0):
                position = -1
                signals[i] = -base_size
        
        elif is_ranging and vol_ok:
            # Ranging market: mean reversion at extremes
            # Short when Bull Power > 0.5 * ATR (overbought)
            if (bull_power_aligned[i] > 0.5 * atr_aligned[i] and 
                position >= 0):
                position = -1
                signals[i] = -base_size * 0.7  # Smaller size for mean reversion
            
            # Long when Bear Power < -0.5 * ATR (oversold)
            elif (bear_power_aligned[i] < -0.5 * atr_aligned[i] and 
                  position <= 0):
                position = 1
                signals[i] = base_size * 0.7  # Smaller size for mean reversion
        
        # Exit conditions
        if position == 1 and (bull_power_aligned[i] <= 0 or 
                              bear_power_aligned[i] >= -0.2 * atr_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_aligned[i] >= 0 or 
                                 bull_power_aligned[i] <= 0.2 * atr_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_ADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0