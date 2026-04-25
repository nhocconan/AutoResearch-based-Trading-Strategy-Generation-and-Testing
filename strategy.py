#!/usr/bin/env python3
"""
6h Elder Ray Power + 12h SuperTrend + Volume Spike
Hypothesis: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13.
12h SuperTrend filters trend direction. Volume spike confirms institutional participation.
Works in bull/bear markets: Long when Bull Power > 0, Bear Power < 0, price > SuperTrend (uptrend).
Short when Bull Power < 0, Bear Power > 0, price < SuperTrend (downtrend). Volume spike ensures
moves have conviction. 6h timeframe balances trade frequency and responsiveness.
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
    
    # Get 12h data for SuperTrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h SuperTrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first element NaN
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # SuperTrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + 3.0 * atr_10
    lower_band = hl2 - 3.0 * atr_10
    
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr_10[i-1]) or np.isnan(close_12h[i-1]):
            continue
            
        if close_12h[i-1] > supertrend[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        if close_12h[i] > upper_band[i]:
            direction[i] = 1
            supertrend[i] = upper_band[i]
        elif close_12h[i] < lower_band[i]:
            direction[i] = -1
            supertrend[i] = lower_band[i]
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = lower_band[i]
    
    # Align SuperTrend and direction to 6h
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Elder Ray calculation
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
        if i >= 13:
            ema_13 = np.mean(close[i-12:i+1])
        else:
            ema_13 = np.mean(close[:i+1])
        
        bull_power = curr_high - ema_13
        bear_power = curr_low - ema_13
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        supertrend_val = supertrend_aligned[i]
        trend_dir = direction_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure), Bear Power < 0 (no selling pressure),
            # price above SuperTrend (uptrend), volume spike
            long_condition = (bull_power > 0) and (bear_power < 0) and (curr_close > supertrend_val) and volume_spike
            # Short: Bull Power < 0 (no buying pressure), Bear Power > 0 (selling pressure),
            # price below SuperTrend (downtrend), volume spike
            short_condition = (bull_power < 0) and (bear_power > 0) and (curr_close < supertrend_val) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative or price breaks below SuperTrend
            if bull_power <= 0 or curr_close < supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns negative or price breaks above SuperTrend
            if bear_power <= 0 or curr_close > supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_12hSuperTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0