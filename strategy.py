#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot S3/R3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above R3 (resistance level) AND price > 1w EMA200 AND volume > 1.5x 20-period average.
# Short when price breaks below S3 (support level) AND price < 1w EMA200 AND volume > 1.5x 20-period average.
# Exit when price retests the pivot point (PP) or opposite S1/R1 level.
# Uses Camarilla pivot levels for institutional support/resistance with trend filter to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "12h_Camarilla_R3S3_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot point
    pp = typical_price.iloc[-1]  # Most recent completed daily bar
    # Ranges
    range_hl = df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1]
    # Camarilla levels
    r4 = pp + range_hl * 1.1 / 2
    r3 = pp + range_hl * 1.1 / 4
    r2 = pp + range_hl * 1.1 / 6
    r1 = pp + range_hl * 1.1 / 12
    s1 = pp - range_hl * 1.1 / 12
    s2 = pp - range_hl * 1.1 / 6
    s3 = pp - range_hl * 1.1 / 4
    s4 = pp - range_hl * 1.1 / 2
    
    # Align pivot levels to 12h timeframe (constant until new daily bar)
    pp_array = np.full(n, pp)
    r3_array = np.full(n, r3)
    s3_array = np.full(n, s3)
    r1_array = np.full(n, r1)
    s1_array = np.full(n, s1)
    
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_array)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_array)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_array)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_array)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_array)
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, above 1w EMA200, volume spike
            long_cond = (close[i] > r3_aligned[i]) and (close[i-1] <= r3_aligned[i-1]) and \
                       (close[i] > ema200_1w_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below S3, below 1w EMA200, volume spike
            short_cond = (close[i] < s3_aligned[i]) and (close[i-1] >= s3_aligned[i-1]) and \
                        (close[i] < ema200_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price retests pivot point or S1 level
            if close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retests pivot point or R1 level
            if close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals