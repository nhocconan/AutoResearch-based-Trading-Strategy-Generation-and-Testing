#!/usr/bin/env python3
# 1D_1W_CCI_Trend_Follow
# Hypothesis: Use CCI on weekly timeframe to determine primary trend (bullish > +100, bearish < -100) and enter on daily breakouts in trend direction.
# Weekly CCI filters out daily noise and provides strong trend bias, reducing whipsaws. Works in bull by catching uptrends, in bear by following downtrends.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

name = "1D_1W_CCI_Trend_Follow"
timeframe = "1d"
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
    
    # Get weekly data for CCI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Commodity Channel Index (CCI) on weekly data
    # CCI = (Typical Price - SMA(TP, 20)) / (0.015 * Mean Deviation)
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Calculate SMA of typical price
    sma_tp = np.full_like(tp_1w, np.nan)
    for i in range(19, len(tp_1w)):
        sma_tp[i] = np.mean(tp_1w[i-19:i+1])
    
    # Calculate mean deviation
    mean_dev = np.full_like(tp_1w, np.nan)
    for i in range(19, len(tp_1w)):
        dev = np.abs(tp_1w[i-19:i+1] - sma_tp[i])
        mean_dev[i] = np.mean(dev)
    
    # Calculate CCI
    cci_1w = np.full_like(tp_1w, np.nan)
    for i in range(19, len(tp_1w)):
        if mean_dev[i] != 0:
            cci_1w[i] = (tp_1w[i] - sma_tp[i]) / (0.015 * mean_dev[i])
    
    # Define trend based on CCI levels
    bullish_trend = cci_1w > 100
    bearish_trend = cci_1w < -100
    
    # Align weekly CCI trend to daily timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend.astype(float))
    
    # Calculate daily Donchian channels for breakout signals
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(50, lookback-1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly bullish trend + price breaks above daily Donchian high
            if bullish and close[i] > highest_high[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly bearish trend + price breaks below daily Donchian low
            elif bearish and close[i] < lowest_low[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns bearish or price breaks below daily Donchian low
            if bearish or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns bullish or price breaks above daily Donchian high
            if bullish or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals