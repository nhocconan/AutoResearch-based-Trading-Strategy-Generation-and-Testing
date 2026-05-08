#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels. In ranging markets, reversals at extremes work well.
# In trending markets (12h ADX > 25), we take reversals in the direction of the trend.
# Volume confirmation ensures genuine interest. Works in both bull and bear markets by adapting to regime.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "6h_WilliamsR_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams %R(14) on 6h data
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    williams_r = -100 * (highest_high - close) / denominator
    
    # 12h data for trend and volume filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h ADX(14) for trend strength
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_12h[0] - low_12h[0]
    
    # Directional Movement
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_12h
    minus_di = 100 * minus_dm_smooth / atr_12h
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_12h[np.isnan(adx_12h)] = 0
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 12h volume filter: current volume > 1.2x 20-period average
    volume_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.2 * vol_ma20_12h)
    volume_filter_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(williams_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) AND trending up (ADX > 25) AND volume confirmation
            long_cond = (williams_r[i] < -80) and (adx_12h_aligned[i] > 25) and volume_filter_aligned[i]
            # Short conditions: Williams %R overbought (> -20) AND trending down (ADX > 25) AND volume confirmation
            short_cond = (williams_r[i] > -20) and (adx_12h_aligned[i] > 25) and volume_filter_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns above -50 (momentum fading) OR ADX weakens (< 20)
            if williams_r[i] > -50 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns below -50 (momentum fading) OR ADX weakens (< 20)
            if williams_r[i] < -50 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals