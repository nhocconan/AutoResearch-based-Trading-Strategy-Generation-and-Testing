#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R3, S3)
    # Based on previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w) / 3
    # Range = H - L
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    # R3 = H + 2*(H-L)/1.1
    r3 = high_1w + 2 * range_1w / 1.1
    # S3 = L - 2*(H-L)/1.1
    s3 = low_1w - 2 * range_1w / 1.1
    
    # Align weekly levels to daily timeframe (use previous week's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily trend filter: EMA(50)
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R3 + above EMA50 + volume spike
            if (close[i] > r3_aligned[i] and
                close[i] > ema_50[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 + below EMA50 + volume spike
            elif (close[i] < s3_aligned[i] and
                  close[i] < ema_50[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below S3 or below EMA50
            if (close[i] < s3_aligned[i] or
                close[i] < ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above R3 or above EMA50
            if (close[i] > r3_aligned[i] or
                close[i] > ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals