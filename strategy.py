#!/usr/bin/env python3
name = "1d_WKLY_Pivot_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_w = df_w['close'].values
    ema_w = pd.Series(close_w).ewm(span=20, min_periods=20).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Weekly high/low/close for pivot calculation (previous week)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot levels
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    r1_w = close_w + range_w * 1.1 / 12
    s1_w = close_w - range_w * 1.1 / 12
    r2_w = close_w + range_w * 1.1 / 6
    s2_w = close_w - range_w * 1.1 / 6
    r3_w = close_w + range_w * 1.1 / 4
    s3_w = close_w - range_w * 1.1 / 4
    
    # Align weekly pivot levels to daily
    r1_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Volume spike: current volume > 2x 20-period average (~1 month)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 AND above weekly EMA20 (uptrend) AND volume spike
            if close[i] > r2_aligned[i] and close[i] > ema_w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 AND below weekly EMA20 (downtrend) AND volume spike
            elif close[i] < s2_aligned[i] and close[i] < ema_w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S2 OR below weekly EMA20 (trend change)
            if close[i] < s2_aligned[i] or close[i] < ema_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R2 OR above weekly EMA20 (trend change)
            if close[i] > r2_aligned[i] or close[i] > ema_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals