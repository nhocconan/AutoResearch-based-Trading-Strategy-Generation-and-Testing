#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action near weekly pivot points with weekly trend filter and volume confirmation.
# Uses weekly pivot levels (support/resistance) for mean reversion in ranging markets and breakout in trending markets.
# Weekly trend (EMA50) determines bias: above EMA50 = look for longs at support, below EMA50 = look for shorts at resistance.
# Volume confirms conviction at pivot levels. Reduces false signals and improves edge in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly pivot points (using prior week's HLC)
    # Pivot = (H + L + C) / 3
    # Support1 = (2 * Pivot) - H
    # Resistance1 = (2 * Pivot) - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    support1 = (2 * pivot) - high_1w
    resistance1 = (2 * pivot) - low_1w
    
    # Align all data to 12-hour timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    support1_aligned = align_htf_to_ltf(prices, df_1w, support1)
    resistance1_aligned = align_htf_to_ltf(prices, df_1w, resistance1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(support1_aligned[i]) or np.isnan(resistance1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x average of last 6 periods (3 days)
        if i >= 6:
            vol_avg = np.mean(volume[i-6:i])
        else:
            vol_avg = 0
        volume_condition = volume[i] > (vol_avg * 1.5)
        
        # Determine bias based on weekly EMA50
        bias_long = close[i] > ema50_1w_aligned[i]  # Above weekly EMA50 = bullish bias
        bias_short = close[i] < ema50_1w_aligned[i]  # Below weekly EMA50 = bearish bias
        
        # Entry conditions: mean reversion at pivot levels with volume and bias alignment
        # Long when price near support1 with bullish bias and volume
        # Short when price near resistance1 with bearish bias and volume
        near_support = low[i] <= support1_aligned[i] * 1.005  # Within 0.5% of support
        near_resistance = high[i] >= resistance1_aligned[i] * 0.995  # Within 0.5% of resistance
        
        if position == 0:
            if near_support and bias_long and volume_condition:
                position = 1
                signals[i] = position_size
            elif near_resistance and bias_short and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches pivot or breaks below support with volume
            if high[i] >= pivot_aligned[i] * 0.995 or (low[i] < support1_aligned[i] and volume_condition):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches pivot or breaks above resistance with volume
            if low[i] <= pivot_aligned[i] * 1.005 or (high[i] > resistance1_aligned[i] and volume_condition):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Pivot_Point_Bias_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0