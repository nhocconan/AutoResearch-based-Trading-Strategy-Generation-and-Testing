#!/usr/bin/env python3
"""
6h_WoodyCCI_Divergence_With_Volume
Hypothesis: Trade divergences between price and Woody CCI on 6h timeframe, confirmed by volume spikes and aligned with 1d trend.
Long when bullish divergence (price makes lower low, CCI makes higher low) with volume spike and 1d uptrend.
Short when bearish divergence (price makes higher high, CCI makes lower high) with volume spike and 1d downtrend.
Uses Woody CCI (typically more responsive than standard CCI) with period 14.
Designed for 6h to capture medium-term reversals with confirmation, reducing false signals.
Works in bull/bear: 1d trend filter ensures trades align with higher timeframe momentum.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
"""

name = "6h_WoodyCCI_Divergence_With_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema20_1d = ema(close_1d, 20)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate Woody CCI on 6h (Typical Price = (H+L+C)/3)
    tp = (high + low + close) / 3.0
    # CCI = (TP - SMA(TP,20)) / (0.015 * Mean Deviation)
    sma_tp = np.full_like(tp, np.nan)
    md = np.full_like(tp, np.nan)
    cci = np.full_like(tp, np.nan)
    
    for i in range(19, len(tp)):
        sma_tp[i] = np.mean(tp[i-19:i+1])
        mean_dev = np.mean(np.abs(tp[i-19:i+1] - sma_tp[i]))
        if mean_dev > 0:
            cci[i] = (tp[i] - sma_tp[i]) / (0.015 * mean_dev)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(cci[i-1]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(close[i-1]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, CCI makes higher low
            bull_div = (close[i] < close[i-1]) and (low[i] < low[i-1]) and (cci[i] > cci[i-1])
            # Bearish divergence: price makes higher high, CCI makes lower high
            bear_div = (close[i] > close[i-1]) and (high[i] > high[i-1]) and (cci[i] < cci[i-1])
            
            if bull_div and volume_filter[i] and close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif bear_div and volume_filter[i] and close[i] < ema20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish divergence OR price breaks below EMA20
            bear_div = (close[i] > close[i-1]) and (high[i] > high[i-1]) and (cci[i] < cci[i-1])
            if bear_div or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish divergence OR price breaks above EMA20
            bull_div = (close[i] < close[i-1]) and (low[i] < low[i-1]) and (cci[i] > cci[i-1])
            if bull_div or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals