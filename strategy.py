#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike.
# Uses daily EMA for trend direction, Camarilla levels from daily pivot for entries,
# and volume surge for confirmation. Works in both bull (breakouts above R3) and bear
# (breakdowns below S3). Target: 50-150 total trades over 4 years to avoid fee drag.
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot, EMA, and volume calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily pivot and Camarilla levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    r3_1d = close_1d + 1.1 * range_1d * 1.1 / 4
    s3_1d = close_1d - 1.1 * range_1d * 1.1 / 4
    
    # Align 1d indicators to 6h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (strict threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 to be valid
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above R3 + daily close > EMA34 (uptrend) + volume spike
            if (price > r3_aligned[i] and close_1d[-1] > ema34_1d[-1] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + daily close < EMA34 (downtrend) + volume spike
            elif (price < s3_aligned[i] and close_1d[-1] < ema34_1d[-1] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R3 or trend turns down
            if price < r3_aligned[i] or close_1d[-1] < ema34_1d[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S3 or trend turns up
            if price > s3_aligned[i] or close_1d[-1] > ema34_1d[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals