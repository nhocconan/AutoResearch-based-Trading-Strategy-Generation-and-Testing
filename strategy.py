#!/usr/bin/env python3
"""
6h_WeeklyPivot_Pullback_1dTrend_Volume
Hypothesis: Uses weekly pivot points (PP, R1, S1) from prior week for pullback entries in direction of 1d trend, 
confirmed by volume spike. Designed for 6h timeframe to capture institutional-level support/resistance bounces 
with low trade frequency (target: 15-30 trades/year). Works in both bull and bear markets by following 1d trend 
direction, avoiding counter-trend trades. Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "6h_WeeklyPivot_Pullback_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for pivot points (prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = (2 * pp) - weekly_low
    s1 = (2 * pp) - weekly_high
    
    # Align weekly pivot points to 6h timeframe (use prior week's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume filter: volume > 2.0x 30-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 30)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Long entry: price pulls back to S1 + above 1d EMA + volume spike
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price pulls back to R1 + below 1d EMA + volume spike
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or reaches PP or volume drops
            if (close[i] < s1_aligned[i] or close[i] > pp_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or reaches PP or volume drops
            if (close[i] > r1_aligned[i] or close[i] < pp_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals