#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly close for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    # Weekly EMA(34) for trend filter
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly high/low/close for pivot points
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w_vals = pd.Series(df_1w['close'].values)
    
    # Weekly pivot points: PP, R3, S3
    pp_1w = (high_1w + low_1w + close_1w_vals) / 3
    r3_1w = close_1w_vals + (high_1w - low_1w) * 1.1
    s3_1w = close_1w_vals - (high_1w - low_1w) * 1.1
    
    # Align pivot points to daily timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w.values)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w.values)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w.values)
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike and above weekly EMA trend
            if close[i] > r3_1w_aligned[i] and vol_ok and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume spike and below weekly EMA trend
            elif close[i] < s3_1w_aligned[i] and vol_ok and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S3 (mean reversion)
            if close[i] < s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above R3
            if close[i] > r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals