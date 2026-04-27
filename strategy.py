#!/usr/bin/env python3
"""
12h_VolumeBreakout_TrendFollowing
Hypothesis: On 12h timeframe, price breaking above/below 20-period high/low with volume confirmation and trend filter captures trends in both bull and bear markets.
- Long: price > highest(high,20) + volume > 1.5x average + 1d EMA50 up (close > EMA50)
- Short: price < lowest(low,20) + volume > 1.5x average + 1d EMA50 down (close < EMA50)
- Exit: opposite breakout or trend reversal
- Designed to catch sustained moves with institutional volume
- Target: 15-35 trades/year (60-140 total over 4 years)
"""

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
    
    # 20-period highest high and lowest low
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Volume spike: 1.5x average
    volume_spike = volume > (vol_ma * 1.5)
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: break above 20-period high with volume and uptrend
            if (close[i] > highest_high[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: break below 20-period low with volume and downtrend
            elif (close[i] < lowest_low[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: break below 20-period low or trend reversal
            if (close[i] < lowest_low[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above 20-period high or trend reversal
            if (close[i] > highest_high[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolumeBreakout_TrendFollowing"
timeframe = "12h"
leverage = 1.0