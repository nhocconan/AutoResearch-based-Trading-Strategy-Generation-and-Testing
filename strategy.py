#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with 1-week SMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; trend filter ensures trades align with higher timeframe direction.
# Long when Williams %R < -80 (oversold) and price > 1w SMA (uptrend) with volume confirmation.
# Short when Williams %R > -20 (overbought) and price < 1w SMA (downtrend) with volume confirmation.
# Exit when Williams %R crosses back to -50 (mean reversion) or trend changes.
# Designed for low trade frequency (20-30/year) to avoid fee decay. Works in both trending and ranging markets via trend filter.

name = "4h_1dWilliamsR_1wSMA_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Williams %R (14-period)
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    # For true rolling window, we need to look back 14 periods
    hh_14 = np.full_like(high_1d, np.nan)
    ll_14 = np.full_like(low_1d, np.nan)
    
    for i in range(13, len(high_1d)):
        hh_14[i] = np.max(high_1d[i-13:i+1])
        ll_14[i] = np.min(low_1d[i-13:i+1])
    
    williams_r = -100 * (hh_14 - close_1d) / (hh_14 - ll_14)
    williams_r[:13] = np.nan  # Not enough data
    
    # Get 1w data for SMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1-week SMA (9-period)
    sma_9 = np.convolve(close_1w, np.ones(9)/9, mode='same')
    for i in range(len(close_1w)):
        if i < 4 or i >= len(close_1w) - 4:
            sma_9[i] = np.nan
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    sma_9_aligned = align_htf_to_ltf(prices, df_1w, sma_9)
    
    # Volume confirmation: 4h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for Williams %R and SMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(sma_9_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) and price > 1w SMA (uptrend) with volume confirmation
            if (williams_r_aligned[i] < -80 and 
                close[i] > sma_9_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) and price < 1w SMA (downtrend) with volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < sma_9_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 or trend turns down
            if williams_r_aligned[i] > -50 or close[i] < sma_9_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 or trend turns up
            if williams_r_aligned[i] < -50 or close[i] > sma_9_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals