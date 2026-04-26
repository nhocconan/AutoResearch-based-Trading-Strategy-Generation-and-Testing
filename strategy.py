#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 breakout with weekly trend filter and volume confirmation on 12h timeframe.
In bull markets: price breaks above R3 with weekly uptrend → long. 
In bear markets: price breaks below S3 with weekly downtrend → short.
Uses discrete sizing (0.25) and volume confirmation to reduce false breakouts.
Target: 50-150 trades over 4 years (12-37/year). Works in both regimes by requiring alignment with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need 20 for volume median
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load daily data for Camarilla pivot levels (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_range = (high_1d - low_1d)
    r3_1d = close_1d + 1.1 * camarilla_range / 2
    s3_1d = close_1d - 1.1 * camarilla_range / 2
    
    # Align daily Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Weekly EMA26 for trend filter
    ema_26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_26_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 20 for volume median)
    start_idx = 20
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_26_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Camarilla breakout conditions
        close_val = close[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        
        # Long logic: price breaks above R3 with volume spike and weekly uptrend
        long_condition = close_val > r3_val and volume_spike[i] and (close_val > ema_26_1w_aligned[i])
        # Short logic: price breaks below S3 with volume spike and weekly downtrend
        short_condition = close_val < s3_val and volume_spike[i] and (close_val < ema_26_1w_aligned[i])
        
        # Exit logic: price re-enters Camarilla range or weekly trend reversal
        exit_long = close_val < r3_val or close_val < ema_26_1w_aligned[i]
        exit_short = close_val > s3_val or close_val > ema_26_1w_aligned[i]
        
        # Minimum holding period: 2 bars to reduce churn
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0