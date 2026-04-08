#!/usr/bin/env python3

"""
12h_camarilla_pivot_daily_trend_volume_v1
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
In uptrend (price > daily EMA200), buy at S3/S4 levels with volume confirmation.
In downtrend (price < daily EMA200), sell at R3/R4 levels with volume confirmation.
Uses 12h timeframe for lower frequency trading to reduce fee drag.
Works in both bull and bear markets by trading mean reversion within the trend.
Target: 12-30 trades per year to minimize fee impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_daily_trend_volume_v1"
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
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    daily_range = high_1d - low_1d
    r4 = close_1d + 1.5 * daily_range
    r3 = close_1d + 1.1 * daily_range
    s3 = close_1d - 1.1 * daily_range
    s4 = close_1d - 1.5 * daily_range
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are warmed up
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema200_1d_aligned[i]
        daily_downtrend = close[i] < ema200_1d_aligned[i]
        
        # Volume confirmation (at least 1.5x average volume)
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit when price reaches R3 (take profit) or trend changes
            if close[i] >= r3_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches S3 (take profit) or trend changes
            if close[i] <= s3_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry at S4 in uptrend with volume confirmation
                if daily_uptrend and close[i] <= s4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry at R4 in downtrend with volume confirmation
                elif daily_downtrend and close[i] >= r4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals