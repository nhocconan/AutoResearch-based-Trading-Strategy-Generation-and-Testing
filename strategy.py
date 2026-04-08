#!/usr/bin/env python3
# 12h_camarilla_pivot_daily_trend_volume_v2
# Hypothesis: Camarilla pivot levels from daily timeframe combined with daily trend filter and volume confirmation on 12h timeframe.
# Works in bull markets by buying dips to S3/S4 in uptrend, and in bear markets by selling rallies to R3/R4 in downtrend.
# Uses volume spike to confirm institutional interest at key levels.
# Target: 15-25 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_daily_trend_volume_v2"
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
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels based on previous day's range
    range_1d = high_1d - low_1d
    # Resistance levels
    r1 = close_1d + (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    r3 = close_1d + (range_1d * 1.1 / 4)
    r4 = close_1d + (range_1d * 1.1 / 2)
    # Support levels
    s1 = close_1d - (range_1d * 1.1 / 12)
    s2 = close_1d - (range_1d * 1.1 / 6)
    s3 = close_1d - (range_1d * 1.1 / 4)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all daily data to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation on 12h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 or trend changes
            if close[i] <= s1_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 or trend changes
            if close[i] >= r1_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: price touches S3/S4 in uptrend (buy the dip)
                if daily_uptrend and (close[i] <= s3_aligned[i] or close[i] <= s4_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches R3/R4 in downtrend (sell the rally)
                elif daily_downtrend and (close[i] >= r3_aligned[i] or close[i] >= r4_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals