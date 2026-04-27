#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 12h timeframe with 1-week EMA50 trend filter and volume confirmation. Uses wider Camarilla levels (R3/S3) for fewer false breakouts and 1-week trend filter to capture major moves while avoiding chop. Designed for low trade frequency (12-37/year) with discrete position sizing (0.25) to minimize fee drag. Works in bull markets by following 1w uptrend for longs, and in bear markets by following 1w downtrend for shorts. Volume spike confirms institutional interest at breakout.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels (R3/S3)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 from 1d OHLC
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # 1-week EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 12h timeframe (completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    
    # Warmup: need 1w EMA50 (50) + 1d Camarilla + volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with 1-week EMA50 trend filter and volume spike
            # Long: price closes above R3 AND above 1w EMA50 (1w uptrend)
            long_condition = (close_val > r3_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S3 AND below 1w EMA50 (1w downtrend)
            short_condition = (close_val < s3_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions:
            # 1. Price touches S3 (opposite Camarilla level)
            # 2. 1-week EMA50 turns bearish (price below EMA)
            exit_condition = (close_val < s3_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. Price touches R3 (opposite Camarilla level)
            # 2. 1-week EMA50 turns bullish (price above EMA)
            exit_condition = (close_val > r3_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0