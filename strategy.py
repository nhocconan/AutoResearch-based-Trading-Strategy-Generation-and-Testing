#!/usr/bin/env python3
"""
1d_1w_HMA_Trend_Pullback_v1
Hypothesis: Uses 1-day Hull Moving Average (HMA) for trend direction and 1-week HMA for higher timeframe confirmation.
Enters on pullbacks to the 1d HMA when the 1w HMA confirms trend direction, with volume confirmation.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "1d_1w_HMA_Trend_Pullback_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def weighted_wma(values, window):
    """Weighted Moving Average with weights 1,2,3,...,window"""
    if len(values) < window:
        return np.full_like(values, np.nan)
    weights = np.arange(1, window + 1)
    return np.convolve(values, weights, mode='valid') / weights.sum()

def hull_moving_average(values, window):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    half_window = window // 2
    sqrt_window = int(np.sqrt(window))
    
    wma_half = weighted_wma(values, half_window)
    wma_full = weighted_wma(values, window)
    
    # Align arrays: wma_half starts at index (window - half_window)
    # wma_full starts at index (window - 1)
    raw_wma = 2 * wma_half - wma_full
    
    # Pad raw_wma to match original length
    padded_raw = np.full_like(values, np.nan)
    start_idx = window - 1
    end_idx = start_idx + len(raw_wma)
    if end_idx <= len(values):
        padded_raw[start_idx:end_idx] = raw_wma
    
    # Final WMA of sqrt_window on raw_wma
    hma = weighted_wma(padded_raw, sqrt_window)
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day HMA (21-period)
    hma_21 = hull_moving_average(close, 21)
    
    # Get weekly data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    hma_21_1w = hull_moving_average(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Volume confirmation: current volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if any critical value is NaN
        if (np.isnan(hma_21[i]) or np.isnan(hma_21_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price pulls back to 1d HMA from above, weekly HMA confirms uptrend
            if (close[i] <= hma_21[i] * 1.01 and  # Within 1% above HMA (pullback)
                close[i] > hma_21[i] and          # Still above HMA
                hma_21_1w_aligned[i] > hma_21_1w_aligned[max(0, i-5)] and  # Weekly HMA rising
                volume[i] > vol_ma[i]):           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to 1d HMA from below, weekly HMA confirms downtrend
            elif (close[i] >= hma_21[i] * 0.99 and  # Within 1% below HMA (pullback)
                  close[i] < hma_21[i] and          # Still below HMA
                  hma_21_1w_aligned[i] < hma_21_1w_aligned[max(0, i-5)] and  # Weekly HMA falling
                  volume[i] > vol_ma[i]):           # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below 1d HMA or weekly trend turns down
            if close[i] < hma_21[i] or hma_21_1w_aligned[i] < hma_21_1w_aligned[max(0, i-5)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above 1d HMA or weekly trend turns up
            if close[i] > hma_21[i] or hma_21_1w_aligned[i] > hma_21_1w_aligned[max(0, i-5)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals