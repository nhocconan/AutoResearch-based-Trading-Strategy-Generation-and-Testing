#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    open_price = prices['open'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d high, low, close for Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous 1d
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels: R3 = C + (H-L)*1.1/2, R4 = C + (H-L)*1.1
    # Support levels: S3 = C - (H-L)*1.1/2, S4 = C - (H-L)*1.1
    r3 = close_1d + range_1d * 1.1 / 2
    s3 = close_1d - range_1d * 1.1 / 2
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume average for volume confirmation
    vol_ma10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma10_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 (support hold), price > 1d EMA34 (uptrend), volume > 1.2x average
            if (close[i] > s3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > vol_ma10_1d_aligned[i] * 1.2):
                signals[i] = 0.25
                position = 1
            # Short: price below R3 (resistance hold), price < 1d EMA34 (downtrend), volume > 1.2x average
            elif (close[i] < r3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > vol_ma10_1d_aligned[i] * 1.2):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below pivot (trend change) or volume drops
            if close[i] < pivot_aligned[i] or volume[i] < vol_ma10_1d_aligned[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above pivot (trend change) or volume drops
            if close[i] > pivot_aligned[i] or volume[i] < vol_ma10_1d_aligned[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals