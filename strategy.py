#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot R1/S1 levels from 1d timeframe act as strong support/resistance.
# Breakout above R1 with bullish 1d trend (price > EMA34) triggers long; breakdown below S1 with bearish 1d trend triggers short.
# Volume confirmation (volume > 1.5x 20-period EMA) filters false breakouts.
# Works in bull markets via breakouts above resistance and in bear via breakdowns below support.
# Target: 20-50 trades/year on 4h timeframe.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + (range_val * 1.1 / 2)
    r3 = c + (range_val * 1.1/4)
    r2 = c + (range_val * 1.1/6)
    r1 = c + (range_val * 1.1/12)
    s1 = c - (range_val * 1.1/12)
    s2 = c - (range_val * 1.1/6)
    s3 = c - (range_val * 1.1/4)
    s4 = c - (range_val * 1.1/2)
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r4_1d, r3_1d, r2_1d, r1_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 4h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1) + EMA34 (34) + vol EMA (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 1d trend bullish (price > EMA34) AND volume confirmation
            if close[i] > r1_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND 1d trend bearish (price < EMA34) AND volume confirmation
            elif close[i] < s1_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 (reversal signal)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 (reversal signal)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals