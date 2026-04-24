#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend + Volume Spike
- Primary timeframe: 6h for execution, HTF: 1d for EMA50 trend filter.
- Williams %R(14): Long when < -80 (oversold), Short when > -20 (overbought) in 6h timeframe.
- Trend filter: Only take longs when 1d EMA50 is rising, shorts when 1d EMA50 is falling.
- Volume confirmation: Current 6h volume > 2.0x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying oversold pullbacks in uptrend, in bear via selling overbought rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)  # Williams %R(14) + EMA50 buffer + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i-1]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Williams %R extreme + 1d EMA50 trend + volume spike
            ema50_slope = ema_50_1d_aligned[i] - ema_50_1d_aligned[i-1]
            
            if williams_r[i] < -80 and ema50_slope > 0 and volume_spike[i]:
                # Oversold + uptrend + volume: long
                signals[i] = 0.25
                position = 1
            elif williams_r[i] > -20 and ema50_slope < 0 and volume_spike[i]:
                # Overbought + downtrend + volume: short
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns above -50 or opposite extreme
            if williams_r[i] > -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns below -50 or opposite extreme
            if williams_r[i] < -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0