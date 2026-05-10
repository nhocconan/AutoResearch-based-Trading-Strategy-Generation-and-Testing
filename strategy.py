#!/usr/bin/env python3

# 6H_WeeklyPivot_Direction_1dTrend_Filter
# Hypothesis: Uses weekly pivot points from 1w to establish major support/resistance zones,
# combined with 1d trend filter (EMA50) and volume confirmation on 6h timeframe.
# In bull markets: buy near weekly S1/S2 with 1d uptrend, in bear markets: sell near weekly R1/R2 with 1d downtrend.
# Weekly pivots provide stronger institutional levels than daily pivots, reducing false breaks.
# Target: 15-30 trades per year on 6h timeframe with position size 0.25 to minimize fee drag.
# Designed to work in both bull and bear markets by aligning with 1d trend and using weekly structure.

name = "6H_WeeklyPivot_Direction_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Standard pivot: P = (H + L + C) / 3
    # Support 1: S1 = (2 * P) - H
    # Resistance 1: R1 = (2 * P) - L
    # Support 2: S2 = P - (H - L)
    # Resistance 2: R2 = P + (H - L)
    wk_high = df_1w['high']
    wk_low = df_1w['low']
    wk_close = df_1w['close']
    
    pivot = (wk_high + wk_low + wk_close) / 3
    s1 = (2 * pivot) - wk_high
    r1 = (2 * pivot) - wk_low
    s2 = pivot - (wk_high - wk_low)
    r2 = pivot + (wk_high - wk_low)
    
    # Align weekly pivot levels to 6h timeframe (use prior week's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    
    # Volume filter: volume > 1.3x 50-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_threshold = vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or \
           np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price near weekly support + above 1d EMA50 + volume spike
            # Enter when price touches or breaks above S1/S2 with uptrend bias
            if ((close[i] >= s1_aligned[i] * 0.998 and close[i] <= s1_aligned[i] * 1.002) or
                (close[i] >= s2_aligned[i] * 0.998 and close[i] <= s2_aligned[i] * 1.002)) and \
               price_above_ema and \
               volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price near weekly resistance + below 1d EMA50 + volume spike
            elif ((close[i] >= r1_aligned[i] * 0.998 and close[i] <= r1_aligned[i] * 1.002) or
                  (close[i] >= r2_aligned[i] * 0.998 and close[i] <= r2_aligned[i] * 1.002)) and \
                 price_below_ema and \
                 volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S2 or trend turns down
            if close[i] < s2_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly R2 or trend turns up
            if close[i] > r2_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals