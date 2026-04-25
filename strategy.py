#!/usr/bin/env python3
"""
6h Elder Ray Power + 12h SuperTrend + Volume Spike
Hypothesis: Elder Ray (Bull/Bear Power) measures trend strength relative to EMA13; 
combined with 12h SuperTrend filter and volume confirmation, it captures strong 
momentum moves in both bull and bear markets while avoiding whipsaws. 6h timeframe 
targets ~12-25 trades/year to minimize fee drag.
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
    
    # Get 12h data for SuperTrend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h SuperTrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR calculation for 12h
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # SuperTrend calculation
    hl2_12h = (high_12h + low_12h) / 2
    upper_12h = hl2_12h + 3.0 * atr_12h
    lower_12h = hl2_12h - 3.0 * atr_12h
    
    supertrend_12h = np.full_like(close_12h, np.nan)
    direction_12h = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_12h)):
        if i == 0:
            supertrend_12h[i] = upper_12h[i]
            direction_12h[i] = 1
        else:
            if supertrend_12h[i-1] == upper_12h[i-1]:
                supertrend_12h[i] = lower_12h[i] if close_12h[i] > upper_12h[i-1] else upper_12h[i]
                direction_12h[i] = -1 if supertrend_12h[i] == lower_12h[i] else 1
            else:
                supertrend_12h[i] = upper_12h[i] if close_12h[i] < lower_12h[i-1] else lower_12h[i]
                direction_12h[i] = 1 if supertrend_12h[i] == upper_12h[i] else -1
    
    # Align SuperTrend direction to 6h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Elder Ray calculation on 6h data (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA13 and SuperTrend warmup
    start_idx = max(13, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(supertrend_dir_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        supertrend_dir = supertrend_dir_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (strong bullish momentum) AND 12h SuperTrend uptrend AND volume spike
            long_condition = (curr_bull_power > 0) and (supertrend_dir == 1) and curr_volume_spike
            # Short: Bear Power < 0 (strong bearish momentum) AND 12h SuperTrend downtrend AND volume spike
            short_condition = (curr_bear_power < 0) and (supertrend_dir == -1) and curr_volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: Bull Power <= 0 (momentum fading) OR 12h SuperTrend turns down
            if curr_bull_power <= 0 or supertrend_dir == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (momentum fading) OR 12h SuperTrend turns up
            if curr_bear_power >= 0 or supertrend_dir == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_12hSuperTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0