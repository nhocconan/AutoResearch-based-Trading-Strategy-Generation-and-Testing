#!/usr/bin/env python3
"""
6h Elder Ray + SuperTrend(12h) + Volume Spike
Hypothesis: Elder Ray (Bull/Bear power) identifies institutional buying/selling pressure.
SuperTrend(12h) filters for higher-timeframe trend direction. Volume spike confirms participation.
Long when Bull Power > 0, Bear Power < 0, price above SuperTrend(12h), and volume spike.
Short when Bear Power < 0, Bull Power < 0, price below SuperTrend(12h), and volume spike.
Works in bull/bear markets by requiring alignment of short-term power, higher-TF trend, and volume.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 12h data for SuperTrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate SuperTrend on 12h: ATR(10), multiplier=3.0
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first TR is NaN
    
    # ATR(10)
    atr_12h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper/Lower Bands
    hl2 = (high_12h + low_12h) / 2.0
    upper_basic = hl2 + 3.0 * atr_12h
    lower_basic = hl2 - 3.0 * atr_12h
    
    # Final Upper/Lower Bands
    upper_final = np.full_like(close_12h, np.nan)
    lower_final = np.full_like(close_12h, np.nan)
    upper_final[0] = upper_basic[0]
    lower_final[0] = lower_basic[0]
    
    for i in range(1, len(close_12h)):
        if close_12h[i-1] <= upper_final[i-1]:
            upper_final[i] = min(upper_basic[i], upper_final[i-1])
        else:
            upper_final[i] = upper_basic[i]
            
        if close_12h[i-1] >= lower_final[i-1]:
            lower_final[i] = max(lower_basic[i], lower_final[i-1])
        else:
            lower_final[i] = lower_basic[i]
    
    # SuperTrend direction: 1 = uptrend, -1 = downtrend
    supertrend_dir = np.full_like(close_12h, np.nan)
    supertrend_dir[0] = 1
    
    for i in range(1, len(close_12h)):
        if supertrend_dir[i-1] == -1 and close_12h[i] > upper_final[i-1]:
            supertrend_dir[i] = 1
        elif supertrend_dir[i-1] == 1 and close_12h[i] < lower_final[i-1]:
            supertrend_dir[i] = -1
        else:
            supertrend_dir[i] = supertrend_dir[i-1]
    
    # Align SuperTrend to 6h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, supertrend_dir)
    
    # Calculate Elder Ray on 6h: EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13 (13) + volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if SuperTrend not ready
        if np.isnan(supertrend_dir_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        st_dir = supertrend_dir_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry conditions
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, SuperTrend uptrend, volume spike
            long_condition = (curr_bull > 0) and (curr_bear < 0) and (st_dir == 1) and vol_spike
            # Short: Bear Power < 0, Bull Power < 0, SuperTrend downtrend, volume spike
            short_condition = (curr_bear < 0) and (curr_bull < 0) and (st_dir == -1) and vol_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or SuperTrend turns down
            if curr_bull <= 0 or st_dir == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or SuperTrend turns up
            if curr_bear >= 0 or st_dir == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_SuperTrend12h_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0