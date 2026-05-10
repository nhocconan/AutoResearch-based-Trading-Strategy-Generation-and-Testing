#!/usr/bin/env python3
# 1h_Camarilla_Pivot_Breakout_4hTrend_Volume
# Hypothesis: Use Camarilla pivot points (S3/R3) on 1h for breakout entries, filtered by 4h EMA50 trend and volume spikes.
# Camarilla levels provide high-probability reversal/breakout zones. 4h EMA50 ensures alignment with intermediate trend.
# Volume confirmation filters low-conviction breakouts. Designed for 1h timeframe with controlled trade frequency.
# Works in bull/bear by following 4h trend direction; range-bound markets filtered by volume.
# Target: 15-35 trades/year to stay within optimal range for 1h.

name = "1h_Camarilla_Pivot_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for previous period
    # Using previous bar's high, low, close for current bar's levels (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels: S3, S2, S1, R1, R2, R3
    s3 = prev_close - (range_val * 1.1 / 2)
    s2 = prev_close - (range_val * 1.1 / 4)
    s1 = prev_close - (range_val * 1.1 / 6)
    r1 = prev_close + (range_val * 1.1 / 6)
    r2 = prev_close + (range_val * 1.1 / 4)
    r3 = prev_close + (range_val * 1.1 / 2)
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price crosses above R3 with volume, 4h EMA uptrend
            if close[i] > r3[i] and close[i-1] <= r3[i-1] and volume_filter[i] and ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price crosses below S3 with volume, 4h EMA downtrend
            elif close[i] < s3[i] and close[i-1] >= s3[i-1] and volume_filter[i] and ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price crosses below R1 or 4h EMA turns down
            if close[i] < r1[i] or ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price crosses above S1 or 4h EMA turns up
            if close[i] > s1[i] or ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals