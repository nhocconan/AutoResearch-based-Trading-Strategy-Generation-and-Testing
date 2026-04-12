#!/usr/bin/env python3
# 12h_1w_camarilla_breakout_volume
# Hypothesis: 12-hour strategy using weekly Camarilla pivot levels for breakout signals,
# with volume confirmation and 12h EMA50 for trend direction.
# Weekly pivots provide strong support/resistance levels that work across market regimes.
# Volume confirmation reduces false breakouts. EMA50 filter ensures alignment with higher timeframe trend.
# Target: 12-37 trades per year (50-150 total over 4 years) to minimize fee drag.

name = "12h_1w_camarilla_breakout_volume"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 12h EMA50 for trend direction
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Previous weekly bar's range for Camarilla calculation
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    range_1w = prev_high_1w - prev_low_1w
    # Resistance levels
    r3 = prev_close_1w + range_1w * 1.1 / 2
    r4 = prev_close_1w + range_1w * 1.1
    # Support levels
    s3 = prev_close_1w - range_1w * 1.1 / 2
    s4 = prev_close_1w - range_1w * 1.1
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > EMA50 (uptrend) AND close breaks above R4 with volume
        if (close[i] > ema50_12h_aligned[i] and close[i] > r4_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < EMA50 (downtrend) AND close breaks below S4 with volume
        elif (close[i] < ema50_12h_aligned[i] and close[i] < s4_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals