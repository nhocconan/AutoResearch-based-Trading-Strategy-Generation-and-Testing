#!/usr/bin/env python3
"""
4h_PivotPoint_Breakout_Volume_Spike_TrendFilter
Hypothesis: Use daily pivot points (PP, R1, S1) as key support/resistance levels on 4h chart. Breakouts above R1 or below S1 with volume spike and aligned trend (via 1w EMA200) capture institutional breakout moves. Works in bull/bear via 1w trend filter, limiting trades to 20-40/year via strict breakout conditions.
"""

name = "4h_PivotPoint_Breakout_Volume_Spike_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    
    # Align pivot levels to 4h (they change only once per day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get 4h data for volume and price
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 200 EMA and 20 MA
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1w EMA200
        uptrend_1w = close[i] > ema200_1w_aligned[i]
        downtrend_1w = close[i] < ema200_1w_aligned[i]
        
        # Volume filter: current 4h volume > 2x 20-period MA
        volume_spike = volume[i] > vol_ma20_4h_aligned[i] * 2.0
        
        # Breakout conditions
        breakout_long = high[i] > r1_aligned[i]  # Break above R1
        breakout_short = low[i] < s1_aligned[i]  # Break below S1
        
        if position == 0:
            # Long: break above R1 in uptrend with volume spike
            if breakout_long and uptrend_1w and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 in downtrend with volume spike
            elif breakout_short and downtrend_1w and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below PP or trend fails
            if low[i] < pp_aligned[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above PP or trend fails
            if high[i] > pp_aligned[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals