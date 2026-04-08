#!/usr/bin/env python3
# 1d_camarilla_pivot_weekly_trend_volume_v1
# Hypothesis: Uses weekly trend filter and daily Camarilla pivot levels for mean-reversion entries.
# Goes long when price rebounds from S3/S4 in weekly uptrend with volume confirmation.
# Goes short when price rebounds from R3/R4 in weekly downtrend with volume confirmation.
# Designed for 1d timeframe to reduce trade frequency and avoid fee drag. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly trend filter: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    hl_range = high_1d - low_1d
    r4 = close_1d + 1.5 * hl_range
    r3 = close_1d + 1.0 * hl_range
    s3 = close_1d - 1.0 * hl_range
    s4 = close_1d - 1.5 * hl_range
    
    # Align daily data to daily timeframe (no alignment needed, but using for consistency)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation on daily
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(r4[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(s4[i]) or np.isnan(ema50_1w[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 or trend changes
            if close[i] <= s3_aligned[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R3 or trend changes
            if close[i] >= r3_aligned[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long entry: price bounces from S4/S3 in uptrend
                if weekly_uptrend and close[i] > s4_aligned[i] and close[i-1] <= s4_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price bounces from R4/R3 in downtrend
                elif weekly_downtrend and close[i] < r4_aligned[i] and close[i-1] >= r4_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals