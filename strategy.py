#!/usr/bin/env python3
"""
4H_Camarilla_R3_S3_Breakout_1dTrend_Volume
Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 and above 1d EMA50 with volume > 1.5x avg.
Short when price breaks below S3 and below 1d EMA50 with volume > 1.5x avg.
Exit when price returns to H4/L4 levels or trend filter fails.
Target: 20-40 trades/year per symbol.
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
    
    # Calculate Camarilla levels using previous day's range
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    # Calculate daily pivot and levels
    for i in range(1, n):
        # Previous day's high, low, close
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Range
        range_val = prev_high - prev_low
        
        # Camarilla levels
        camarilla_r3[i] = prev_close + range_val * 1.1 / 4.0
        camarilla_s3[i] = prev_close - range_val * 1.1 / 4.0
        camarilla_h4[i] = prev_close + range_val * 1.1 / 2.0
        camarilla_l4[i] = prev_close - range_val * 1.1 / 2.0
    
    # Volume spike filter: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_spike = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            volume_spike[i] = volume[i] > (vol_ma[i] * 1.5)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla levels, volume MA, and EMA1d
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        ema1d_val = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R3, above 1d EMA50, with volume spike
            if price > r3 and price > ema1d_val and vol_spike:
                signals[i] = size
                position = 1
            # Short: price breaks below S3, below 1d EMA50, with volume spike
            elif price < s3 and price < ema1d_val and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to H4 or trend fails
            if price < h4 or price < ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to L4 or trend fails
            if price > l4 or price > ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0