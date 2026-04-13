#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray with 1d volume confirmation and 1w trend filter.
Williams Alligator (Jaw/Teeth/Lips) defines market structure - when aligned (bullish/bearish) it indicates trending conditions.
Elder Ray (Bull Power/Bear Power) measures trend strength relative to EMA13.
1d volume surge confirms institutional participation in the trend direction.
1-week ADX > 25 ensures we only trade in strong trending markets.
This combination works in both bull (captures strong uptrends) and bear (captures strong downtrends) markets by requiring multiple confirmation layers.
Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume surge (volume > 2.0x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume_1d > (vol_ma_20 * 2.0)
    vol_surge_aligned = align_htf_to_ltf(prices, df_1d, vol_surge.astype(float))
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX for 1w
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Strong trend: ADX > 25
    strong_trend = adx > 25
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Aligator alignment conditions
    # Bullish: Lips > Teeth > Jaw
    # Bearish: Lips < Teeth < Jaw
    bullish_aligned = (lips > teeth) & (teeth > jaw)
    bearish_aligned = (lips < teeth) & (teeth < jaw)
    
    # Elder Ray (using 13-period EMA as reference)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Elder Ray conditions
    # Strong bullish: Bull Power > 0 and increasing
    # Strong bearish: Bear Power < 0 and decreasing
    bull_power_prev = np.roll(bull_power, 1)
    bull_power_prev[0] = bull_power[0]
    bear_power_prev = np.roll(bear_power, 1)
    bear_power_prev[0] = bear_power[0]
    
    strong_bull = (bull_power > 0) & (bull_power > bull_power_prev)
    strong_bear = (bear_power < 0) & (bear_power < bear_power_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(strong_trend_aligned[i]) or 
            np.isnan(vol_surge_aligned[i]) or 
            np.isnan(bullish_aligned[i]) if i < len(bullish_aligned) else True or
            np.isnan(bearish_aligned[i]) if i < len(bearish_aligned) else True):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Alligator alignment + Elder Ray + volume surge + strong trend
        long_condition = (bullish_aligned[i] and 
                         strong_bull[i] and 
                         vol_surge_aligned[i] > 0.5 and 
                         strong_trend_aligned[i] > 0.5)
        
        short_condition = (bearish_aligned[i] and 
                          strong_bear[i] and 
                          vol_surge_aligned[i] > 0.5 and 
                          strong_trend_aligned[i] > 0.5)
        
        # Exit when Alligator alignment breaks (market losing trend)
        exit_long = position == 1 and not bullish_aligned[i]
        exit_short = position == -1 and not bearish_aligned[i]
        
        # Execute signals
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_alligator_elder_ray_volume_trend"
timeframe = "4h"
leverage = 1.0