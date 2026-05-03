#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d ADX regime filter. 
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In strong trends (ADX > 25 on 1d), trade in direction of Elder Ray power: 
# Long when Bull Power > 0 and Bear Power < 0 (bullish imbalance), 
# Short when Bear Power < 0 and Bull Power < 0 (bearish imbalance).
# Uses 1d ADX for regime filtering to avoid whipsaws in ranging markets.
# Designed to capture sustained moves in both bull and bear markets by aligning with 1d trend.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "6h_ElderRay_1dADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for regime filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.concatenate([[np.nan], np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                                                 np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)])
    dm_minus = np.concatenate([[np.nan], np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                                                  np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di_1d = 100 * dm_plus_smoothed / atr_1d
    minus_di_1d = 100 * dm_minus_smoothed / atr_1d
    
    # Directional Index (DX) and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    # Handle division by zero when both DI are zero
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray power
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_6h = high - ema13_6h
    bear_power_6h = low - ema13_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime filter: only trade when ADX > 25 (trending market)
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish imbalance) AND trending regime AND session
            if bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 (bearish imbalance) AND trending regime AND session
            elif bear_power_6h[i] < 0 and bull_power_6h[i] < 0 and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 (loss of bullish imbalance) OR reverse signal
            if bull_power_6h[i] <= 0 or bear_power_6h[i] >= 0 or (bear_power_6h[i] < 0 and bull_power_6h[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bull Power >= 0 (loss of bearish imbalance) OR reverse signal
            if bear_power_6h[i] >= 0 or bull_power_6h[i] >= 0 or (bull_power_6h[i] > 0 and bear_power_6h[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals