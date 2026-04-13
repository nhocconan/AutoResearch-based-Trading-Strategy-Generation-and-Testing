#!/usr/bin/env python3
"""
4h_1d_4h_Camarilla_Pullback_Strategy
Hypothesis: Daily Camarilla pivot levels (S3/R3) act as strong support/resistance.
On 4h, we look for pullbacks to these levels with volume confirmation and alignment with
4h EMA20 trend. This structure works in bull markets (pullbacks to support in uptrend)
and bear markets (pullbacks to resistance in downtrend) by using price action confirmation.
Uses 4h EMA20 as trend filter and 4h volume spike for confirmation to avoid false signals.
Target: 20-40 trades per year per symbol.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d
    r4_1d = close_1d + 1.5 * range_1d
    s4_1d = close_1d - 1.5 * range_1d
    
    # Align daily Camarilla levels to 4h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Pullback to S3 with volume expansion (support bounce)
        # 2. Must be above 4h EMA20 for trend alignment
        pullback_long = (low[i] <= s3_1d_aligned[i]) and (close[i] > s3_1d_aligned[i]) and volume_expansion[i]
        long_condition = pullback_long and (close[i] > ema_20_4h_aligned[i])
        
        # Short conditions:
        # 1. Pullback to R3 with volume expansion (resistance rejection)
        # 2. Must be below 4h EMA20 for trend alignment
        pullback_short = (high[i] >= r3_1d_aligned[i]) and (close[i] < r3_1d_aligned[i]) and volume_expansion[i]
        short_condition = pullback_short and (close[i] < ema_20_4h_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_4h_Camarilla_Pullback_Strategy"
timeframe = "4h"
leverage = 1.0