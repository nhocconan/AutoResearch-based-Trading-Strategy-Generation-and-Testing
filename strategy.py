#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Adaptive
Hypothesis: Uses 6h Elder Ray (Bull/Bear Power) with 1w regime filter (ADX) to adapt strategy:
- In trending regime (1w ADX > 25): Trend follow - long when Bear Power turns up from below zero, short when Bull Power turns down from above zero
- In ranging regime (1w ADX <= 25): Mean revert - long when Bull Power crosses below -0.5*ATR, short when Bear Power crosses above 0.5*ATR
- Volume confirmation required for all entries
Designed for 6h timeframe to achieve 50-150 total trades over 4 years with low fee drag.
Works in both bull and bear markets by adapting to regime.
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
    
    # Get 1w data for regime filter (ADX) and EMA22 for EMA13 in Elder Ray
    df_1w = get_htf_data(prices, '1w')
    
    # 1w ADX for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(low_1w)
    tr = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0)
        minus_dm[i] = max(low_1w[i-1] - low_1w[i], 0)
        tr[i] = max(high_1w[i] - low_1w[i], 
                   abs(high_1w[i] - close_1w[i-1]), 
                   abs(low_1w[i] - close_1w[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[1:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    tr_smooth = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, 
                  (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1w EMA22 for Elder Ray EMA13 (approximation)
    ema_22_1w = pd.Series(close_1w).ewm(span=22, adjust=False, min_periods=22).mean().values
    ema_22_aligned = align_htf_to_ltf(prices, df_1w, ema_22_1w)
    
    # Calculate 6h Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema_22_aligned
    bear_power = low - ema_22_aligned
    
    # 6h ATR for regime-based thresholds
    tr_6h = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.append([np.nan], close[:-1])),
                                 np.abs(low - np.append([np.nan], close[:-1]))))
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1w ADX (14+14=28), EMA22 (22), 6h ATR (14), volume avg (20)
    start_idx = max(28, 22, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_22_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr_6h[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Regime-adaptive entry logic
            if adx_val > 25:  # Trending regime
                # Trend follow: long when Bear Power turns up from below zero
                #           short when Bull Power turns down from above zero
                long_condition = (bear_val > 0) and (bear_power[i-1] <= 0) and vol_conf
                short_condition = (bull_val < 0) and (bull_power[i-1] >= 0) and vol_conf
            else:  # Ranging regime
                # Mean revert: long when Bull Power crosses below -0.5*ATR (oversold)
                #            short when Bear Power crosses above 0.5*ATR (overbought)
                long_condition = (bull_val < -0.5 * atr_val) and (bull_power[i-1] >= -0.5 * atr_val) and vol_conf
                short_condition = (bear_val > 0.5 * atr_val) and (bear_power[i-1] <= 0.5 * atr_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when Bull Power crosses above zero (momentum fading)
            exit_condition = bull_val > 0
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when Bear Power crosses below zero (momentum fading)
            exit_condition = bear_val < 0
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0