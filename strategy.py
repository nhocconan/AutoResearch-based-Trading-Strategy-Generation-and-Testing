#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Camarilla R3/S3 breakouts aligned with 12h EMA21 trend and volume confirmation capture high-probability moves. 
Works in both bull and bear markets: trend filter ensures we only trade in the direction of the 12h trend, 
while volume confirmation and breakout logic capture momentum bursts. Discrete sizing (0.25) limits drawdown.
Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Get 1d data for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.125 * range_1d
    camarilla_s3 = close_1d - 1.125 * range_1d
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators to 4h timeframe
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA21 (21), Camarilla (1), volume avg (20)
    start_idx = max(21, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema21_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema21 = ema21_12h_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine 12h trend: price vs EMA21
            uptrend = close_val > ema21
            downtrend = close_val < ema21
            
            if uptrend and vol_conf:
                # Long bias: long when price breaks above R3 with volume
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short bias: short when price breaks below S3 with volume
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: price crosses below EMA21 (trend change) or touches S3
            if close_val < ema21 or close_val < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: price crosses above EMA21 (trend change) or touches R3
            if close_val > ema21 or close_val > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0