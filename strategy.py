#!/usr/bin/env python3
"""
1d_1D_VolumeBreakout_Supertrend_Filter
Hypothesis: On daily timeframe, use Supertrend for trend direction and volume breakout for entry.
Supertrend filters trend direction, while volume spikes confirm breakout momentum.
Designed to work in both bull and bear markets by following the trend.
Target: 15-25 trades/year per symbol, focusing on high-probability breakouts.
"""

name = "1d_1D_VolumeBreakout_Supertrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR calculation for Supertrend
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_period = 10
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    multiplier = 3.0
    upper_band = (high_1w + low_1w) / 2 + multiplier * atr
    lower_band = (high_1w + low_1w) / 2 - multiplier * atr
    
    supertrend = np.zeros_like(close_1w)
    supertrend_dir = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    supertrend_dir[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            supertrend_dir[i] = 1
        else:
            supertrend[i] = upper_band[i]
            supertrend_dir[i] = -1
            
        # Adjust bands
        if supertrend_dir[i] == 1:
            if lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
        else:
            if upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
                
        if supertrend_dir[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend direction to daily timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir)
    
    # Daily price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume breakout: current volume > 2.0x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_breakout = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(supertrend_dir_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from Supertrend direction
        uptrend = supertrend_dir_aligned[i] == 1
        downtrend = supertrend_dir_aligned[i] == -1
        
        if position == 0:
            # Long entry: uptrend AND volume breakout
            if uptrend and volume_breakout[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend AND volume breakout
            elif downtrend and volume_breakout[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend changes to downtrend
            if not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend changes to uptrend
            if not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals