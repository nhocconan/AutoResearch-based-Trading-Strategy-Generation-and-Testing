#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_chop_v1
# Hypothesis: On 12h timeframe, price retracement to Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts) with volume expansion and low choppiness (trending regime) captures high-probability mean-reversion entries. Works in bull/bear via regime filter.
# Entry: Long when price <= S3 + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
# Entry: Short when price >= R3 + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
# Exit: Price crosses opposite pivot level (S3/R3) or CHOP < 38.2 (trend) or volume fails
# Position sizing: 0.25 long, -0.25 short

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and CHOP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on prior day)
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    hl_range = high_1d - low_1d
    r3 = close_1d + 1.1 * hl_range
    s3 = close_1d - 1.1 * hl_range
    
    # Align pivots to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Choppiness Index (CHOP) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    chop = np.where(denominator != 0, 100 * np.log10(np.sum(tr[-14:]) / denominator) / np.log10(14), 50)
    # For rolling calculation, we need full array
    chop_full = np.zeros(len(close_1d))
    for i in range(13, len(close_1d)):
        tr_sum = np.sum(tr[i-13:i+1])
        denom = highest_high[i] - lowest_low[i]
        chop_full[i] = 100 * np.log10(tr_sum / denom) / np.log10(14) if denom != 0 else 50
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_full)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price >= R3 OR CHOP < 38.2 (trending) OR volume filter fails
            if (close[i] >= r3_aligned[i]) or (chop_aligned[i] < 38.2) or not volume_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price <= S3 OR CHOP < 38.2 (trending) OR volume filter fails
            if (close[i] <= s3_aligned[i]) or (chop_aligned[i] < 38.2) or not volume_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price <= S3 + volume + CHOP > 61.8 (range)
            if (close[i] <= s3_aligned[i]) and volume_filter[i] and (chop_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short entry: price >= R3 + volume + CHOP > 61.8 (range)
            elif (close[i] >= r3_aligned[i]) and volume_filter[i] and (chop_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals