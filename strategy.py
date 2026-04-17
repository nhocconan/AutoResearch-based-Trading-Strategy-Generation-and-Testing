#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align pivot levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Get 4h ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h EMA21 for trend filter
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s1_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(ema21[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA21 for long, below for short
        long_trend = close[i] > ema21[i]
        short_trend = close[i] < ema21[i]
        
        # Volatility filter: avoid extremely low volatility periods
        if i >= 50:
            vol_ma = np.nanmean(atr[i-50:i])
            vol_filter = atr[i] > 0.5 * vol_ma
        else:
            vol_filter = True
        
        if position == 0:
            # Long: price breaks above S3 with trend alignment
            if long_trend and vol_filter and close[i] > s3_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R3 with trend alignment
            elif short_trend and vol_filter and close[i] < r3_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S2
            if close[i] < s2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R2
            if close[i] > r2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_S2S3_R2R3_Breakout_EMA21"
timeframe = "4h"
leverage = 1.0