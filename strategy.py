#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA trend + 12-hour Bollinger Bands squeeze + volume confirmation
# Long when KAMA slope > 0, price > KAMA, BB width < 20th percentile, volume > 1.5x average
# Short when KAMA slope < 0, price < KAMA, BB width < 20th percentile, volume > 1.5x average
# Exit when price crosses KAMA or BB width expands above 50th percentile
# KAMA adapts to market noise, Bollinger squeeze identifies low volatility breakouts,
# Volume confirms breakout strength. Works in bull (trend continuation) and bear (mean reversion after squeeze).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate KAMA on 4h (ER=10, FAST=2, SLOW=30)
    close_4h = df_4h['close'].values
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.sum(np.abs(np.diff(close_4h, prepend=close_4h[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Calculate Bollinger Bands on 12h (20, 2)
    close_12h = df_12h['close'].values
    bb_mid = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate average volume on 12h
    vol_12h = df_12h['volume'].values
    avg_volume = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    bb_mid_aligned = align_htf_to_ltf(prices, df_12h, bb_mid)
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    avg_volume_aligned = align_htf_to_ltf(prices, df_12h, avg_volume)
    
    # Calculate percentile ranks for BB width (using expanding window to avoid look-ahead)
    bb_width_percentile = np.zeros_like(bb_width_aligned)
    for i in range(len(bb_width_aligned)):
        if i < 20:
            bb_width_percentile[i] = 50.0  # neutral until enough data
        else:
            # Use data up to i for percentile calculation (no look-ahead)
            historical_width = bb_width_aligned[:i+1]
            current_width = bb_width_aligned[i]
            if len(historical_width) > 0:
                percentile = (np.sum(historical_width <= current_width) / len(historical_width)) * 100
                bb_width_percentile[i] = percentile
            else:
                bb_width_percentile[i] = 50.0
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(bb_mid_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Calculate KAMA slope (change over 3 periods)
        if i >= 3:
            kama_slope = kama_aligned[i] - kama_aligned[i-3]
        else:
            kama_slope = 0
        
        if position == 0:
            # Long setup: KAMA up, price > KAMA, BB squeeze (width < 20th percentile), volume > 1.5x avg
            if (kama_slope > 0 and price > kama_aligned[i] and 
                bb_width_percentile[i] < 20 and vol > 1.5 * avg_volume_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: KAMA down, price < KAMA, BB squeeze (width < 20th percentile), volume > 1.5x avg
            elif (kama_slope < 0 and price < kama_aligned[i] and 
                  bb_width_percentile[i] < 20 and vol > 1.5 * avg_volume_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA OR BB width expands above 50th percentile
            if price < kama_aligned[i] or bb_width_percentile[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA OR BB width expands above 50th percentile
            if price > kama_aligned[i] or bb_width_percentile[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_12hBB_Squeeze_Volume"
timeframe = "4h"
leverage = 1.0