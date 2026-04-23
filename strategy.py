#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Long: Close breaks above Camarilla R3 + price > 1w EMA50 (bullish trend) + volume > 2.0x 24-period avg
- Short: Close breaks below Camarilla S3 + price < 1w EMA50 (bearish trend) + volume > 2.0x 24-period avg
- Exit: Close crosses Camarilla H6/L6 levels (extreme mean reversion)
- Uses Camarilla pivot levels from daily HTF for structure, 1w EMA50 for trend filter, and volume confirmation
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with trend) and bear markets (mean reversion at extreme pivots)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 24-period average (24 * 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from 1d HTF data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_range = high_1d - low_1d
    camarilla_h4 = close_1d + camarilla_range * 1.1 / 4
    camarilla_l4 = close_1d - camarilla_range * 1.1 / 4
    camarilla_h3 = close_1d + camarilla_range * 1.1 / 6
    camarilla_l3 = close_1d - camarilla_range * 1.1 / 6
    camarilla_h6 = close_1d + camarilla_range * 1.1
    camarilla_l6 = close_1d - camarilla_range * 1.1
    camarilla_r3 = camarilla_h3  # R3 = H3
    camarilla_s3 = camarilla_l3  # S3 = L3
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
    camarilla_l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # Need 50 for EMA50, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_h6_aligned[i]) or
            np.isnan(camarilla_l6_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Camarilla R3 + bullish trend + volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 + bearish trend + volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Camarilla L6 (extreme mean reversion)
            if close[i] < camarilla_l6_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Camarilla H6 (extreme mean reversion)
            if close[i] > camarilla_h6_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0