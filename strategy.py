#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Williams %R with 14-period and 1-week SMA(50) for trend filter.
# Williams %R identifies overbought/oversold conditions, while weekly SMA(50) filters trend direction.
# Long when Williams %R < -80 (oversold) and price > weekly SMA(50) (uptrend).
# Short when Williams %R > -20 (overbought) and price < weekly SMA(50) (downtrend).
# Exit when Williams %R crosses back to -50 (mean reversion).
# Designed for low trade frequency (12-20/year) to avoid fee drag. Works in trending markets via trend filter.

name = "12h_1dWilliamsR_1wSMA50_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    
    # For Williams %R, we need the highest high and lowest low over the last 14 periods
    highest_high_14 = np.full_like(high_1d, np.nan)
    lowest_low_14 = np.full_like(low_1d, np.nan)
    
    for i in range(13, len(high_1d)):
        highest_high_14[i] = np.max(high_1d[i-13:i+1])
        lowest_low_14[i] = np.min(low_1d[i-13:i+1])
    
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14),
        -50
    )
    
    # Get 1w data for SMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50_1w = np.convolve(close_1w, np.ones(50)/50, mode='same')
    
    # Align indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for Williams %R and SMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) and price > weekly SMA(50) (uptrend)
            if williams_r_aligned[i] < -80 and close[i] > sma_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) and price < weekly SMA(50) (downtrend)
            elif williams_r_aligned[i] > -20 and close[i] < sma_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals