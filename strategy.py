#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with 1-week SMA50 trend filter and volume confirmation.
# Williams %R (14) identifies overbought/oversold conditions; > -20 = overbought, < -80 = oversold.
# Trend filter: 1-week SMA50 > price = bullish, < price = bearish.
# Long when Williams %R < -80 (oversold) and trend is bullish with volume confirmation.
# Short when Williams %R > -20 (overbought) and trend is bearish with volume confirmation.
# Exit when Williams %R crosses back to -50 (neutral).
# Designed for low trade frequency (20-40/year) to avoid fee drag. Works in both trending and ranging markets via trend filter.

name = "4h_1dWilliamsR_1wSMA50_TrendFilter"
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
    
    # Calculate 1d Williams %R (14-period)
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    # For true Williams %R, we need the highest high and lowest low over the last 14 periods
    # Using rolling window approach
    williams_r = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        highest_high_14 = np.max(high_1d[i-13:i+1])
        lowest_low_14 = np.min(low_1d[i-13:i+1])
        if highest_high_14 != lowest_low_14:
            williams_r[i] = (highest_high_14 - close_1d[i]) / (highest_high_14 - lowest_low_14) * -100
        else:
            williams_r[i] = -50  # Avoid division by zero
    
    # Get 1w data for SMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w SMA50
    sma_50 = np.full_like(close_1w, np.nan)
    for i in range(49, len(close_1w)):
        sma_50[i] = np.mean(close_1w[i-49:i+1])
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    # Volume confirmation: 4h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(sma_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) and price above 1w SMA50 (bullish trend) with volume confirmation
            if (williams_r_aligned[i] < -80 and 
                close[i] > sma_50_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) and price below 1w SMA50 (bearish trend) with volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < sma_50_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 or price falls below 1w SMA50
            if williams_r_aligned[i] > -50 or close[i] < sma_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 or price rises above 1w SMA50
            if williams_r_aligned[i] < -50 or close[i] > sma_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals