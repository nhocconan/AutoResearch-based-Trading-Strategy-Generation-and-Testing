#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume
Hypothesis: 1h timeframe with 4h trend filter and 1d volume confirmation captures breakouts with controlled frequency. 
Uses Camarilla R3/S3 levels from 1h for entry, 4h EMA50 for trend, 1d volume spike for confirmation. 
Session filter (08-20 UTC) reduces noise. Target: 60-150 trades over 4 years.
"""

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
    
    # Get 1h data for Camarilla levels
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    range_1h = high_1h - low_1h
    camarilla_r3 = close_1h + 1.125 * range_1h
    camarilla_s3 = close_1h - 1.125 * range_1h
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for volume confirmation (volume > 2.0 * 20-period average)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (2.0 * vol_avg_1d)
    
    # Align all indicators to primary timeframe (1h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need Camarilla (1), EMA50 (50), volume avg (20)
    start_idx = max(1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or
            not session_mask[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema50 = ema50_4h_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs 4h EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf:
                # Long bias: long when price breaks above R3 with volume confirmation
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short bias: short when price breaks below S3 with volume confirmation
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price touches S3 (opposite Camarilla level)
            if close_val < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price touches R3 (opposite Camarilla level)
            if close_val > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0