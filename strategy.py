#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Use daily Camarilla pivot levels (R3/S3) as breakout triggers on 6h timeframe, filtered by 1d EMA trend and volume spikes. This combines proven Camarilla breakout edge with trend filtering to work in both bull (breakout continuation) and bear (mean reversion at extremes) markets. Target: 15-35 trades/year.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's values (no look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First value will be invalid (rolled from last), handle below
    
    # Calculate Camarilla levels for previous day
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 6h timeframe (wait for 1d bar to close)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: 24-period average on 6h (4 days)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.divide(volume, vol_ma24, out=np.zeros_like(volume), where=vol_ma24!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready (first rolled value invalid, or alignment not ready)
        if (i == 0 or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend determination
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume spike in uptrend
            if (close[i] > R3_6h[i] and 
                trend_1d_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with volume spike in downtrend
            elif (close[i] < S3_6h[i] and 
                  trend_1d_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or trend changes
            if close[i] < R3_6h[i] or not trend_1d_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or trend changes
            if close[i] > S3_6h[i] or not trend_1d_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals