#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), and volume > 1.5x average.
Short when Bear Power < 0, Bull Power > 0, ADX > 25 (trending), and volume > 1.5x average.
Exit when ADX < 20 (range) or power signals reverse. Uses 6h timeframe targeting 50-150 total trades over 4 years.
Elder Ray measures bull/bear strength via EMA(13), ADX filters for trending markets, volume confirms conviction.
Designed to capture strong trending moves in both bull and bear regimes while avoiding chop.
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
    
    # Load 6h data for EMA13 calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate EMA13 on 6h
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 6h
    bull_power_6h = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power_6h = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    # Load 12h data for ADX calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    tr_sum = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    dm_plus_sum = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False).mean().values
    dm_minus_sum = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_12h = pd.Series(dx).ewm(span=tr_period, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, ADX > 25 (trending), volume > 1.5x average
            if (bull_val > 0 and bear_val < 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, ADX > 25 (trending), volume > 1.5x average
            elif (bear_val < 0 and bull_val > 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX < 20 (range) OR power signals reverse
                if (adx_val < 20 or bull_val <= 0 or bear_val >= 0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX < 20 (range) OR power signals reverse
                if (adx_val < 20 or bull_val >= 0 or bear_val <= 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_12hADX_Volume"
timeframe = "6h"
leverage = 1.0