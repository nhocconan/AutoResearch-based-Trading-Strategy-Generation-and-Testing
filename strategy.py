#!/usr/bin/env python3
# 6H_1W_Camarilla_R4_S4_Breakout_Trend
# Hypothesis: On 6h timeframe, enter long when price breaks above weekly Camarilla R4 level with bullish trend (price > 50-period EMA),
# and short when price breaks below weekly Camarilla S4 level with bearish trend (price < 50-period EMA).
# Uses weekly trend filter to avoid counter-trend trades and Camarilla levels from weekly for precise entries.
# Target: 12-37 trades/year per symbol (50-150 total over 4 years).

name = "6H_1W_Camarilla_R4_S4_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for weekly: R4, S4 based on previous week
    # Typical price = (high + low + close) / 3
    typical_price = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    # Camarilla R4 = close + (range * 1.1/2)
    # Camarilla S4 = close - (range * 1.1/2)
    camarilla_r4 = close_1w + (range_1w * 1.1 / 2)
    camarilla_s4 = close_1w - (range_1w * 1.1 / 2)
    
    # Weekly trend: EMA(50) on close
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1w > ema_50
    
    # Align weekly indicators to 6h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R4 + weekly uptrend
            if close[i] > camarilla_r4_aligned[i] and trend_up_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S4 + weekly downtrend
            elif close[i] < camarilla_s4_aligned[i] and not trend_up_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla S4 (reversal) or trend changes
            if close[i] < camarilla_s4_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Camarilla R4 (reversal) or trend changes
            if close[i] > camarilla_r4_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals