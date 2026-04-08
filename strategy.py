#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_trend_volume_v1
# Hypothesis: Uses daily Camarilla pivot levels with 1-week trend filter and volume confirmation.
# Enters long when price touches pivot support level in uptrend with volume spike; short when touches resistance level in downtrend with volume spike.
# Exits on opposite pivot touch or trend reversal. Designed for low trade frequency (~15-30/year) to minimize fee drag.
# Works in both bull and bear markets via weekly trend filter that aligns with higher timeframe direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels (using previous day's OHLC)
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    r1 = close_1d + (range_hl * 1.1 / 12)
    r2 = close_1d + (range_hl * 1.1 / 6)
    r3 = close_1d + (range_hl * 1.1 / 4)
    r4 = close_1d + (range_hl * 1.1 / 2)
    s1 = close_1d - (range_hl * 1.1 / 12)
    s2 = close_1d - (range_hl * 1.1 / 6)
    s3 = close_1d - (range_hl * 1.1 / 4)
    s4 = close_1d - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly trend filter: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * avg_volume[i]
        
        # Price proximity to pivot levels (within 0.2%)
        proximity = 0.002
        near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < proximity
        near_r2 = abs(close[i] - r2_aligned[i]) / close[i] < proximity
        near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < proximity
        near_s2 = abs(close[i] - s2_aligned[i]) / close[i] < proximity
        
        if position == 1:  # Long position
            # Exit: Price near resistance or trend change
            if (near_r1 or near_r2) or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price near support or trend change
            if (near_s1 or near_s2) or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: Price near support in uptrend
                if weekly_uptrend and (near_s1 or near_s2):
                    position = 1
                    signals[i] = 0.25
                # Short entry: Price near resistance in downtrend
                elif weekly_downtrend and (near_r1 or near_r2):
                    position = -1
                    signals[i] = -0.25
    
    return signals