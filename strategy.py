#!/usr/bin/env python3
# 6h_Price_Action_Range_Breaker
# Hypothesis: In both bull and bear markets, price respects prior day's high-low range.
# Breakouts above prior day's high with volume and 12h trend confirmation lead to continuation,
# while breakdowns below prior day's low lead to reversals. Uses 12h EMA50 for trend filter
# and volume spike for confirmation. Designed for low trade frequency (15-25/year) to minimize fee drag.

name = "6h_Price_Action_Range_Breaker"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend (smooth, lag-appropriate)
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get daily data for prior day's high-low range
    df_1d = get_htf_data(prices, '1d')
    # Prior day's high and low (use previous day's values)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    # Align daily levels to 6h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation (20-period average on 6h = ~5 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above prior day's high with volume, above 12h EMA50 (uptrend)
            if close[i] > prev_high_aligned[i] and volume_confirm and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior day's low with volume, below 12h EMA50 (downtrend)
            elif close[i] < prev_low_aligned[i] and volume_confirm and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below prior day's low or breaks below 12h EMA50
            if close[i] < prev_low_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above prior day's high or breaks above 12h EMA50
            if close[i] > prev_high_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals