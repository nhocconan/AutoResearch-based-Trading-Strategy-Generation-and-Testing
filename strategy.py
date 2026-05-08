#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_ = high_1d - low_1d
    r3 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 2)
    r4 = pivot + (range_ * 1.1)
    s4 = pivot - (range_ * 1.1)
    
    # Align daily Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 4h trend filter: EMA(50) on close
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Long: price breaks above S4 with 4h uptrend and volume
            if (close[i] > s4_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below R3 with 4h downtrend and volume
            elif (close[i] < r3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or reverses below 4h EMA
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 or reverses above 4h EMA
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals