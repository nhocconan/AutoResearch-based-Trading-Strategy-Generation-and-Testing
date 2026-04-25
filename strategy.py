#!/usr/bin/env python3
"""
6h Elder Ray Power + 12h SuperTrend + Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure.
Combined with 12h SuperTrend for trend direction and volume spike for confirmation, this captures strong
momentum moves in both bull and bear markets. 6h timeframe provides sufficient noise reduction while
catching multi-day trends. Discrete position sizing (0.25) minimizes fee churn.
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
    
    # Get 12h data for SuperTrend trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h SuperTrend (ATR=10, mult=3.0)
    hl2 = (df_12h['high'] + df_12h['low']) / 2
    atr = pd.Series(df_12h['high'] - df_12h['low']).rolling(window=10, min_periods=10).mean()
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    supertrend = np.full(len(df_12h), np.nan)
    direction = np.full(len(df_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(df_12h)):
        if i == 0:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upper_band[i-1]:
                supertrend[i] = lower_band[i] if close[i] > upper_band[i-1] else upper_band[i]
                direction[i] = -1 if supertrend[i] == upper_band[i] else 1
            else:
                supertrend[i] = upper_band[i] if close[i] < lower_band[i-1] else lower_band[i]
                direction[i] = 1 if supertrend[i] == lower_band[i] else -1
    
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Elder Ray EMA13 warmup
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(supertrend_direction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Elder Ray: EMA13 of close
        if i >= 13:
            ema13 = np.mean(close[i-12:i+1])
        else:
            ema13 = np.mean(close[:i+1])
        
        bull_power = curr_high - ema13
        bear_power = ema13 - curr_low
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        supertrend_val = supertrend_aligned[i]
        supertrend_dir = supertrend_direction_aligned[i]
        
        # Entry logic
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND price above SuperTrend (uptrend) AND volume spike
            long_condition = (bull_power > 0) and (curr_close > supertrend_val) and volume_spike
            # Short: Bear Power > 0 (selling pressure) AND price below SuperTrend (downtrend) AND volume spike
            short_condition = (bear_power > 0) and (curr_close < supertrend_val) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power > 0 (selling pressure takes over) OR price breaks below SuperTrend
            if (bear_power > 0) or (curr_close < supertrend_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 (buying pressure takes over) OR price breaks above SuperTrend
            if (bull_power > 0) or (curr_close > supertrend_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_SuperTrend12h_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0