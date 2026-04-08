#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend + Volume Spike Strategy
Hypothesis: Weekly pivot points (R3/S3) act as key support/resistance levels. 
Trades in direction of daily EMA trend with volume confirmation capture institutional flow.
Designed for 15-25 trades/year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_wk = df_1w['high'].values
    low_wk = df_1w['low'].values
    close_wk = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot_wk = (high_wk + low_wk + close_wk) / 3.0
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3_wk = high_wk + 2.0 * (pivot_wk - low_wk)
    s3_wk = low_wk - 2.0 * (high_wk - pivot_wk)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_wk_aligned = align_htf_to_ltf(prices, df_1w, pivot_wk)
    r3_wk_aligned = align_htf_to_ltf(prices, df_1w, r3_wk)
    s3_wk_aligned = align_htf_to_ltf(prices, df_1w, s3_wk)
    
    # Volume filter: current volume > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_wk_aligned[i]) or 
            np.isnan(r3_wk_aligned[i]) or
            np.isnan(s3_wk_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly S3 OR trend reverses
            if (close[i] < s3_wk_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly R3 OR trend reverses
            if (close[i] > r3_wk_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs daily EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price crosses above weekly S3 + uptrend + volume spike
            if (close[i] > s3_wk_aligned[i] and 
                close[i-1] <= s3_wk_aligned[i-1] and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price crosses below weekly R3 + downtrend + volume spike
            elif (close[i] < r3_wk_aligned[i] and 
                  close[i-1] >= r3_wk_aligned[i-1] and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals