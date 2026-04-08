#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Hypothesis: 12-hour Camarilla pivot level reversals with volume confirmation and daily trend filter.
# Long: price touches or crosses below S3 (strong support) with volume spike AND price > daily EMA200.
# Short: price touches or crosses above R3 (strong resistance) with volume spike AND price < daily EMA200.
# Exit: price returns to daily EMA200 or opposite R/S3 touch with volume.
# Designed to capture mean-reversion bounces at strong intraday levels in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Previous day's Camarilla pivot levels (R3, S3)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_200 = np.full(len(close_1d), np.nan)
    ema_1d_200[199] = np.mean(close_1d[:200])
    for i in range(200, len(close_1d)):
        ema_1d_200[i] = close_1d[i] * (2/201) + ema_1d_200[i-1] * (199/201)
    
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate Camarilla levels using previous day's OHLC
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
            range_val = prev_high - prev_low
            camarilla_r3[i] = prev_close + range_val * 1.1 / 2
            camarilla_s3[i] = prev_close - range_val * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        vol = volume[i]
        
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        ema_200 = ema_1d_200_aligned[i]
        
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_200):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2x 20-period average
        if i >= 20:
            avg_vol = np.mean(volume[i-20:i])
            vol_spike = vol > 2.0 * avg_vol
        else:
            vol_spike = False
        
        # Price proximity to S3/R3 (within 0.1%)
        near_s3 = abs(price - s3) / s3 < 0.001
        near_r3 = abs(price - r3) / r3 < 0.001
        
        if position == 1:  # Long position
            # Exit: price returns to EMA200 or touches R3 with volume
            if abs(price - ema_200) / ema_200 < 0.005 or (near_r3 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to EMA200 or touches S3 with volume
            if abs(price - ema_200) / ema_200 < 0.005 or (near_s3 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long at S3 with volume spike and price above EMA200
            if near_s3 and vol_spike and price > ema_200:
                position = 1
                signals[i] = 0.25
            # Enter short at R3 with volume spike and price below EMA200
            elif near_r3 and vol_spike and price < ema_200:
                position = -1
                signals[i] = -0.25
    
    return signals