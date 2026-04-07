#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v2
Hypothesis: On 6-hour timeframe, use Camarilla pivot levels from daily timeframe with EMA trend filter and volume confirmation. 
In uptrend (price > daily EMA50): long at S3/S4 reversal, short at R3/R4 breakout. 
In downtrend (price < daily EMA50): short at R3/R4 reversal, long at S3/S4 breakout. 
This captures both mean reversion in strong trends and breakout continuations. 
Volume > 1.3x average confirms institutional interest. 
Designed for low frequency (12-30 trades/year) to avoid fee drag while working in both bull (trend continuations) and bear (mean reversion in trends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    c = close
    r4 = c + (range_val * 1.1 / 2)
    r3 = c + (range_val * 1.1 / 4)
    r2 = c + (range_val * 1.1 / 6)
    r1 = c + (range_val * 1.1 / 12)
    s1 = c - (range_val * 1.1 / 12)
    s2 = c - (range_val * 1.1 / 6)
    s3 = c - (range_val * 1.1 / 4)
    s4 = c - (range_val * 1.1 / 2)
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    d_ema50 = pd.Series(d_close).ewm(span=50, adjust=False).mean().values
    d_ema50_aligned = align_htf_to_ltf(prices, df_1d, d_ema50)
    
    # Pre-calculate Camarilla levels for each day
    r4_arr = np.full(len(d_high), np.nan)
    r3_arr = np.full(len(d_high), np.nan)
    r2_arr = np.full(len(d_high), np.nan)
    r1_arr = np.full(len(d_high), np.nan)
    s1_arr = np.full(len(d_high), np.nan)
    s2_arr = np.full(len(d_high), np.nan)
    s3_arr = np.full(len(d_high), np.nan)
    s4_arr = np.full(len(d_high), np.nan)
    
    for i in range(len(d_high)):
        r4, r3, r2, r1, s1, s2, s3, s4 = calculate_camarilla(d_high[i], d_low[i], d_close[i])
        r4_arr[i] = r4
        r3_arr[i] = r3
        r2_arr[i] = r2
        r1_arr[i] = r1
        s1_arr[i] = s1
        s2_arr[i] = s2
        s3_arr[i] = s3
        s4_arr[i] = s4
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_arr)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_arr)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_arr)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_arr)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_arr)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_arr)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_arr)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_arr)
    
    # Volume confirmation: 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if daily EMA50 not available
        if np.isnan(d_ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA50
        uptrend = close[i] > d_ema50_aligned[i]
        downtrend = close[i] < d_ema50_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit conditions: reverse signal or stop at opposite Camarilla level
            if uptrend:
                # In uptrend, exit long if price reaches R3 (take profit) or breaks below S3 (reverse)
                if close[i] >= r3_aligned[i] or close[i] <= s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # downtrend
                # In downtrend, exit long if price reaches S3 (take profit) or breaks above R3 (reverse)
                if close[i] <= s3_aligned[i] or close[i] >= r3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions: reverse signal or stop at opposite Camarilla level
            if uptrend:
                # In uptrend, exit short if price reaches R3 (take profit) or breaks above R3 (reverse)
                if close[i] >= r3_aligned[i] or close[i] <= s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # downtrend
                # In downtrend, exit short if price reaches R3 (take profit) or breaks above R3 (reverse)
                if close[i] >= r3_aligned[i] or close[i] <= s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_confirm:
                if uptrend:
                    # In uptrend: look for mean reversion entries at support or breakout at resistance
                    long_entry = (close[i] <= s3_aligned[i] and close[i] >= s4_aligned[i]) or \
                                 (close[i] >= r4_aligned[i])
                    short_entry = (close[i] >= r3_aligned[i] and close[i] <= r4_aligned[i]) or \
                                  (close[i] <= s4_aligned[i])
                else:  # downtrend
                    # In downtrend: look for mean reversion entries at resistance or breakdown at support
                    short_entry = (close[i] >= r3_aligned[i] and close[i] <= r4_aligned[i]) or \
                                  (close[i] <= s4_aligned[i])
                    long_entry = (close[i] <= s3_aligned[i] and close[i] >= s4_aligned[i]) or \
                                 (close[i] >= r4_aligned[i])
                
                if long_entry:
                    position = 1
                    signals[i] = 0.25
                elif short_entry:
                    position = -1
                    signals[i] = -0.25
    
    return signals