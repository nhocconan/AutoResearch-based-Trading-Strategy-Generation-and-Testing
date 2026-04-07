#!/usr/bin/env python3
"""
6h_camarilla_pivot_1w_ema_volume_v1
Hypothesis: On 6-hour timeframe, use weekly Camarilla pivot levels with daily EMA filter and volume confirmation.
Long when price breaks above R3 with daily EMA(50) trending up and volume > 1.5x 20-period average.
Short when price breaks below S3 with daily EMA(50) trending down and volume > 1.5x 20-period average.
Exit when price returns to the weekly pivot point.
Designed for 10-25 trades/year to minimize fee drag while capturing institutional level breaks.
Weekly pivots provide strong support/resistance that work in both bull and bear markets as they adapt to price action.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1w_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (P) = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    # Weekly range
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    r3_1w = close_1w + range_1w * 1.1 / 2
    s3_1w = close_1w - range_1w * 1.1 / 2
    pivot_1w_val = pivot_1w  # for exit
    
    # Align to 6h timeframe (use previous week's levels)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w_val)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine daily trend direction (using EMA slope)
    daily_trend_up = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    for i in range(1, len(ema_50_1d_aligned)):
        if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
            daily_trend_up[i] = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            daily_trend_down[i] = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly pivot
            if close[i] <= pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly pivot
            if close[i] >= pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price breaks above R3 with daily uptrend
                if (close[i] > r3_1w_aligned[i] and close[i-1] <= r3_1w_aligned[i-1] and 
                    daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S3 with daily downtrend
                elif (close[i] < s3_1w_aligned[i] and close[i-1] >= s3_1w_aligned[i-1] and 
                      daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals